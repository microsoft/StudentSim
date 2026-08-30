"""Training the two heads the tutor RL reward reads.

The backbone is frozen, so a head is a single linear layer over a pooled hidden
state and training it costs a matmul per step. That makes the work two phases:
run the backbone once over the labelled examples and keep the pooled features,
then fit the heads on those features for as many epochs as needed.

The perception head predicts, for each of six ways a tutor message can
misdescribe the board, whether the message does it. Classes are rare and uneven,
so the loss weights positives per class and weights examples by how much the
label is trusted.

The style head is distilled from a classifier over the four guidance styles: it
learns the classifier's distribution rather than its argmax, so the gate reads a
calibrated probability instead of a hard vote.

The two heads are fitted together but need not be labelled together. A row with
no style target contributes nothing to the style loss, so a set labelled only
for board errors trains the perception head and leaves the style head as it
was.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from studentsim.core.seeds import DATA_SAMPLER_SEED
from studentsim.tutor_rl.multihead import (
    ERROR_TYPES,
    HIDDEN_DIM,
    PER_CLASS_F1,
    STYLE_LABELS,
)

EPOCHS: Final = 100
LEARNING_RATE: Final = 1e-3
WEIGHT_DECAY: Final = 0.01
STYLE_LOSS_WEIGHT: Final = 1.0
PATIENCE: Final = 20
BATCH_SIZE: Final = 256


@dataclass
class LabelledExample:
    """One tutor message, with what the board-error and style labels say.

    ``perception`` is a 0/1 flag per error type. ``style`` is a distribution
    over the four styles, not a single label, because the head is distilled
    from it. ``weight`` scales this example's perception loss by how much its
    labels are trusted.
    """

    fen: str
    wrong_move: str
    tutor_text: str
    perception: Sequence[float]
    style: Sequence[float]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if len(self.perception) != len(ERROR_TYPES):
            raise ValueError(
                f"perception needs {len(ERROR_TYPES)} flags, got {len(self.perception)}"
            )
        if len(self.style) != len(STYLE_LABELS):
            raise ValueError(f"style needs {len(STYLE_LABELS)} weights, got {len(self.style)}")


def read_examples(path: str | Path) -> list[LabelledExample]:
    """Read the labelled set written by the dataset builder."""
    examples = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                LabelledExample(
                    fen=row["fen"],
                    wrong_move=row["wrong_move"],
                    tutor_text=row["tutor_text"],
                    perception=row["perception"],
                    style=row["style"],
                    weight=float(row.get("weight", 1.0)),
                )
            )
    return examples


def positive_weights(examples: Sequence[LabelledExample]) -> list[float]:
    """How much to scale each error class's positives.

    A class that fires on a tenth of the examples gets nine times the weight,
    so a head cannot do well by calling everything clean.
    """
    weights = []
    for index in range(len(ERROR_TYPES)):
        positives = sum(1 for example in examples if example.perception[index] > 0.5)
        negatives = len(examples) - positives
        weights.append(negatives / positives if positives else 1.0)
    return weights


@dataclass
class TrainingReport:
    """What a head-training run produced."""

    epochs_run: int
    best_epoch: int
    best_recall: float
    per_class_f1: list[float] = field(default_factory=list)
    style_accuracy: float = 0.0

    def metrics_payload(self) -> dict:
        """The metrics file the perception gate reads its weights from."""
        return {
            PER_CLASS_F1: self.per_class_f1,
            "style_accuracy": self.style_accuracy,
            "best_epoch": self.best_epoch,
            "best_recall": self.best_recall,
        }


def pooled_features(
    simulator,
    examples: Sequence[LabelledExample],
    *,
    batch_size: int = 32,
    progress_every: int = 0,
):
    """Run the backbone once and keep the pooled hidden state per example.

    Training reads these instead of the model, which is what makes fitting the
    heads cheap. This is the only part of head training that touches a GPU, and
    on a full labelled set it is the part that takes the time, so pass
    ``progress_every`` to hear about it.
    """
    import torch

    from studentsim.tutor_rl.multihead import MAX_PROMPT_TOKENS, simulator_chat

    features = []
    tokenizer = simulator.tokenizer
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                simulator_chat(e.fen, e.wrong_move, e.tutor_text),
                tokenize=False,
                add_generation_prompt=True,
            )
            for e in batch
        ]
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        ).to(simulator.device)
        with torch.inference_mode():
            forward = simulator.model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                # Only the hidden states are wanted. Scoring every position
                # against a 150k vocabulary would cost more memory than the
                # rest of the pass put together.
                logits_to_keep=1,
            )
            pooled = simulator._pool_tutor_turn(
                encoded.input_ids, encoded.attention_mask, forward.hidden_states[-1].float()
            )
        features.append(pooled.cpu())
        done = start + len(batch)
        if progress_every and (done % (progress_every * batch_size) < batch_size):
            print(f"  pooled {done:,}/{len(examples):,}", flush=True)
    return torch.cat(features) if features else torch.zeros(0, HIDDEN_DIM)


def train_heads(
    features,
    examples: Sequence[LabelledExample],
    *,
    validation_fraction: float = 0.2,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    style_loss_weight: float = STYLE_LOSS_WEIGHT,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    seed: int = DATA_SAMPLER_SEED,
):
    """Fit both heads on precomputed features.

    Returns the state dicts to save and a report carrying the per-class F1 the
    perception gate weights itself by.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(examples), generator=generator)
    split = int(len(examples) * (1.0 - validation_fraction))
    train_index, val_index = order[:split], order[split:]

    perception = torch.tensor([list(e.perception) for e in examples], dtype=torch.float32)
    style = torch.tensor([list(e.style) for e in examples], dtype=torch.float32)
    sample_weight = torch.tensor([e.weight for e in examples], dtype=torch.float32)
    pos_weight = torch.tensor(positive_weights(examples), dtype=torch.float32)

    perception_head = nn.Linear(features.size(1), len(ERROR_TYPES))
    style_head = nn.Linear(features.size(1), len(STYLE_LABELS))
    optimizer = AdamW(
        list(perception_head.parameters()) + list(style_head.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_recall = -1.0
    best_epoch = 0
    best_state: dict | None = None
    stale = 0
    epoch = 0
    for epoch in range(1, epochs + 1):
        perception_head.train()
        style_head.train()
        for start in range(0, len(train_index), batch_size):
            rows = train_index[start : start + batch_size]
            optimizer.zero_grad()
            pooled = features[rows]
            perception_loss = (bce(perception_head(pooled), perception[rows]).mean(dim=1)
                               * sample_weight[rows]).mean()
            style_loss = F.kl_div(
                F.log_softmax(style_head(pooled), dim=-1), style[rows], reduction="batchmean"
            )
            (perception_loss + style_loss_weight * style_loss).backward()
            optimizer.step()

        perception_head.eval()
        style_head.eval()
        with torch.inference_mode():
            predicted = (torch.sigmoid(perception_head(features[val_index])) > 0.5).float()
        recall = _mean_recall(predicted, perception[val_index])
        # Ties go to the later epoch: recall can top out while the style head
        # is still moving, and selecting strictly on improvement would freeze
        # both heads at whichever epoch first reached the ceiling.
        if recall >= best_recall:
            best_recall, best_epoch, stale = recall, epoch, 0
            best_state = {
                "style_head": {k: v.clone() for k, v in style_head.state_dict().items()},
                "perception_head": {
                    k: v.clone() for k, v in perception_head.state_dict().items()
                },
            }
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:  # pragma: no cover - only when there is no validation data
        raise ValueError("training produced no checkpoint; is the labelled set empty?")

    perception_head.load_state_dict(best_state["perception_head"])
    style_head.load_state_dict(best_state["style_head"])
    with torch.inference_mode():
        predicted = (torch.sigmoid(perception_head(features[val_index])) > 0.5).float()
        style_predicted = style_head(features[val_index]).argmax(dim=-1)
    report = TrainingReport(
        epochs_run=epoch,
        best_epoch=best_epoch,
        best_recall=best_recall,
        per_class_f1=_per_class_f1(predicted, perception[val_index]),
        style_accuracy=float(
            (style_predicted == style[val_index].argmax(dim=-1)).float().mean().item()
        ),
    )
    return best_state, report


def _mean_recall(predicted, actual) -> float:
    recalls = []
    for index in range(actual.size(1)):
        positives = actual[:, index].sum().item()
        if positives:
            recalls.append((predicted[:, index] * actual[:, index]).sum().item() / positives)
    return sum(recalls) / len(recalls) if recalls else 0.0


def _per_class_f1(predicted, actual) -> list[float]:
    scores = []
    for index in range(actual.size(1)):
        true_positive = (predicted[:, index] * actual[:, index]).sum().item()
        predicted_positive = predicted[:, index].sum().item()
        actual_positive = actual[:, index].sum().item()
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return scores


def calibrate(state: dict, features, examples: Sequence[LabelledExample]) -> dict:
    """Fit a per-class scale and shift on the perception logits.

    The head is trained to separate classes, not to report how often it is
    right. The gate multiplies by a probability, so the logits are stretched
    onto the observed rate before they are used.
    """
    import torch
    import torch.nn as nn
    from torch.optim import LBFGS

    head = nn.Linear(features.size(1), len(ERROR_TYPES))
    head.load_state_dict(state["perception_head"])
    head.eval()
    # Not inference_mode: these logits are the input to a fitted optimization,
    # and tensors made there cannot take part in one.
    with torch.no_grad():
        logits = head(features).detach()
    actual = torch.tensor([list(e.perception) for e in examples], dtype=torch.float32)

    scale = torch.ones(len(ERROR_TYPES), requires_grad=True)
    shift = torch.zeros(len(ERROR_TYPES), requires_grad=True)
    optimizer = LBFGS([scale, shift], lr=0.1, max_iter=100)
    loss_fn = nn.BCEWithLogitsLoss()

    def step():
        optimizer.zero_grad()
        loss = loss_fn(logits * scale + shift, actual)
        loss.backward()
        return loss

    optimizer.step(step)
    return {
        **state,
        "perception_platt_a": scale.detach(),
        "perception_platt_b": shift.detach(),
    }
