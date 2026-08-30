"""The two multipliers that sit on top of the move-quality reward.

Move quality alone rewards any guidance that gets the student to a better move,
including guidance that reaches it by saying false things about the board or by
abandoning the style the tutor was asked for. The gates price both: each is a
number in ``(0, 1]`` that the move-quality term is multiplied by, so a rollout
keeps its full reward only when it is also on-style and factually clean.

    reward = move_quality * style_gate * perception_gate

Both fall off exponentially, which keeps a confident failure expensive while
leaving a small doubt nearly free.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from studentsim.tutor_rl.multihead import ERROR_TYPES, PER_CLASS_F1, STYLE_LABELS

ALPHA_STYLE: Final = 2.0
ALPHA_PERCEPTION: Final = 3.0


def style_gate(style_probs: Sequence[float], *, preferred: str, alpha: float = ALPHA_STYLE) -> float:
    """How much of the reward survives the tutor writing in the wrong style.

    Full credit when the style head is certain the message is the requested
    style, decaying as that certainty drops.
    """
    if preferred not in STYLE_LABELS:
        raise ValueError(f"preferred style {preferred!r} is not one of {list(STYLE_LABELS)}")
    if len(style_probs) != len(STYLE_LABELS):
        raise ValueError(f"expected {len(STYLE_LABELS)} style probabilities, got {len(style_probs)}")
    preferred_probability = float(style_probs[STYLE_LABELS.index(preferred)])
    return math.exp(-alpha * (1.0 - preferred_probability))


def perception_weights(metrics_path: str | Path) -> tuple[float, ...]:
    """Per-class weights for the perception gate, read from the head metrics.

    Each error class is weighted by the F1 the perception head reaches on it,
    normalized to sum to one, so classes the head reads reliably carry more of
    the penalty than classes it guesses at.
    """
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    if PER_CLASS_F1 not in metrics:
        raise KeyError(
            f"{metrics_path} has no {PER_CLASS_F1!r}; is it the metrics file the "
            "head trainer writes?"
        )
    f1 = [float(x) for x in metrics[PER_CLASS_F1]]
    if len(f1) != len(ERROR_TYPES):
        raise ValueError(f"expected {len(ERROR_TYPES)} F1 scores, got {len(f1)}")
    total = sum(f1)
    if total <= 0:
        raise ValueError("per-class F1 scores sum to zero; the gate would be undefined")
    return tuple(score / total for score in f1)


def perception_gate(
    perception_probs: Sequence[float],
    *,
    weights: Sequence[float],
    alpha: float = ALPHA_PERCEPTION,
) -> float:
    """How much of the reward survives the tutor misdescribing the board.

    The six error probabilities are combined into one weighted sum, and the
    gate decays in that sum.
    """
    if len(perception_probs) != len(ERROR_TYPES):
        raise ValueError(
            f"expected {len(ERROR_TYPES)} error probabilities, got {len(perception_probs)}"
        )
    if len(weights) != len(ERROR_TYPES):
        raise ValueError(f"expected {len(ERROR_TYPES)} weights, got {len(weights)}")
    aggregate = sum(float(p) * float(w) for p, w in zip(perception_probs, weights))
    return math.exp(-alpha * aggregate)


def gated_reward(
    move_quality: float,
    *,
    style_probs: Sequence[float],
    perception_probs: Sequence[float],
    preferred_style: str | None,
    weights: Sequence[float],
    alpha_style: float = ALPHA_STYLE,
    alpha_perception: float = ALPHA_PERCEPTION,
) -> float:
    """The scalar one rollout earns.

    ``preferred_style`` of ``None`` leaves the style gate open, so the reward
    is the move-quality term through the perception gate alone.
    """
    style = (
        1.0
        if preferred_style is None
        else style_gate(style_probs, preferred=preferred_style, alpha=alpha_style)
    )
    return (
        float(move_quality)
        * style
        * perception_gate(perception_probs, weights=weights, alpha=alpha_perception)
    )
