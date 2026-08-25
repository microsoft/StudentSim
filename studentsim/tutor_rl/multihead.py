"""The simulator backbone with a style head and a perception head on top.

The tutor RL reward reads three signals from one model: the move the simulator
plays after hearing the tutor, how the tutor's message reads as a teaching
style, and whether the message says things about the board that are not true.
All three come from a single backbone, the chess Stage-1 adapter, with two
linear heads pooled off the tutor's turn.

One batch costs one generate call plus one forward. Generation gives the move;
the forward gives hidden states over the prompt, and the heads read a mean over
the tutor-turn span, which is the span they were trained on.

The backbone is frozen. Only the two heads carry learned parameters, and they
are trained separately (see :mod:`studentsim.tutor_rl.heads`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ERROR_TYPES: Final = (
    "wrong_square",
    "wrong_piece",
    "wrong_color",
    "hallucinated_piece",
    "wrong_capture",
    "illegal_move",
)
"""The six ways a tutor message can misdescribe the board."""

PER_CLASS_F1: Final = "per_class_f1"
"""Where the head trainer writes, and the gate reads, one F1 per error class.

Named here because the two ends have to agree on it and neither owns it.
"""

STYLE_LABELS: Final = ("error_remediation", "socratic", "strategic", "comparative")
"""The four guidance styles, in the order the style head scores them."""

HIDDEN_DIM: Final = 2560
MAX_NEW_TOKENS: Final = 32
MAX_PROMPT_TOKENS: Final = 1024

_UCI = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)


def parse_uci(text: str) -> str:
    """Read the move out of a decode, or ``""`` when there is none."""
    if not text:
        return ""
    match = _UCI.search(_THINK_BLOCK.sub("", text.strip()))
    return match.group(1).lower() if match else ""


def simulator_chat(fen: str, wrong_move: str, tutor_text: str) -> list[dict[str, str]]:
    """The exchange the reward shows the simulator.

    Three messages and no assistant turn: the position, the move the student
    played, and what the tutor said. The heads pool over the third one.
    """
    return [
        {
            "role": "user",
            "content": f"You are a chess student. Given the position:\n{fen}\n\nWhat is your move?",
        },
        {"role": "assistant", "content": wrong_move},
        {
            "role": "user",
            "content": f"Your tutor explains:\n{tutor_text}\n\n"
            "Given this feedback, what is your move now?",
        },
    ]


@dataclass(frozen=True)
class HeadSignals:
    """What one batch of rollouts yields."""

    moves: list[str]
    style_probs: list[list[float]]
    perception_probs: list[list[float]]


def _linear_head(hidden_dim: int, n_classes: int):
    import torch.nn as nn

    return nn.Linear(hidden_dim, n_classes)


class MultiHeadSimulator:
    """The Stage-1 adapter plus the two trained heads, held on one device."""

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: str | Path,
        heads_checkpoint: str | Path | None = None,
        device: str = "cuda:0",
        hidden_dim: int = HIDDEN_DIM,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=dtype, attn_implementation="sdpa", trust_remote_code=True
        ).to(self.device)
        self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(self.device)
        self.model.eval()

        # Training the heads needs this class for its pooling and nothing else,
        # and at that point there is no checkpoint to load yet.
        if heads_checkpoint is None:
            self.style_head = self.perception_head = None
            self.platt = None
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            return

        # The checkpoint also carries a path in its metadata, which the default
        # loader will not unpickle.
        checkpoint = torch.load(str(heads_checkpoint), weights_only=False, map_location=self.device)
        self.style_head = _linear_head(hidden_dim, len(STYLE_LABELS)).to(self.device)
        self.style_head.load_state_dict(checkpoint["style_head"])
        self.style_head.eval()
        self.perception_head = _linear_head(hidden_dim, len(ERROR_TYPES)).to(self.device)
        self.perception_head.load_state_dict(checkpoint["perception_head"])
        self.perception_head.eval()

        platt_a = checkpoint.get("perception_platt_a")
        platt_b = checkpoint.get("perception_platt_b")
        self.platt = (
            (platt_a.to(self.device), platt_b.to(self.device))
            if platt_a is not None and platt_b is not None
            else None
        )

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for head in (self.style_head, self.perception_head):
            for parameter in head.parameters():
                parameter.requires_grad = False

    def infer_batch(
        self, fens: list[str], wrong_moves: list[str], tutor_texts: list[str]
    ) -> HeadSignals:
        """Score one batch of rollouts."""
        if self.perception_head is None:
            raise ValueError("this simulator was loaded without heads, so it cannot score")
        if not (len(fens) == len(wrong_moves) == len(tutor_texts)):
            raise ValueError("fens, wrong_moves, and tutor_texts must be the same length")
        torch = self._torch

        texts = [
            self.tokenizer.apply_chat_template(
                simulator_chat(fen, move, tutor), tokenize=False, add_generation_prompt=True
            )
            for fen, move, tutor in zip(fens, wrong_moves, tutor_texts)
        ]
        # Left padding keeps batched generation aligned to the prompt end.
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        ).to(self.device)
        prompt_length = encoded.input_ids.size(1)

        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
            completions = self.tokenizer.batch_decode(
                generated.sequences[:, prompt_length:], skip_special_tokens=True
            )
            # A separate forward for the hidden states: reading them off the
            # generate call is unreliable across transformers versions.
            forward = self.model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            pooled = self._pool_tutor_turn(
                encoded.input_ids, encoded.attention_mask, forward.hidden_states[-1].float()
            )
            style = torch.softmax(self.style_head(pooled), dim=-1)
            perception_logits = self.perception_head(pooled)
            if self.platt is not None:
                perception_logits = perception_logits * self.platt[0] + self.platt[1]
            perception = torch.sigmoid(perception_logits)

        return HeadSignals(
            moves=[parse_uci(c) for c in completions],
            style_probs=style.cpu().tolist(),
            perception_probs=perception.cpu().tolist(),
        )

    def _pool_tutor_turn(self, input_ids, attention_mask, hidden):
        """Mean the hidden states over the tutor's message.

        The tutor turn is the last chat turn in the prompt, so it runs from the
        final ``<|im_start|>`` to the final ``<|im_end|>``. This is the span the
        heads were trained to read; a prompt without those markers falls back to
        the last real token.
        """
        torch = self._torch
        start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        pooled = torch.zeros(
            input_ids.size(0), hidden.size(-1), dtype=hidden.dtype, device=hidden.device
        )
        for row in range(input_ids.size(0)):
            ids = input_ids[row]
            mask = attention_mask[row].bool()
            last_real = int(mask.sum()) - 1
            ends = ((ids == end_id) & mask).nonzero(as_tuple=True)[0]
            if ends.numel() == 0:
                pooled[row] = hidden[row, last_real]
                continue
            end_index = int(ends[-1].item())
            starts = ((ids == start_id) & mask).nonzero(as_tuple=True)[0]
            starts = starts[starts < end_index]
            if starts.numel() == 0 or end_index <= int(starts[-1].item()):
                pooled[row] = hidden[row, last_real]
                continue
            pooled[row] = hidden[row, int(starts[-1].item()) : end_index].mean(0)
        return pooled
