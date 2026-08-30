"""Per-domain evaluation protocol.

These values define how a simulator is decoded and scored. They are fixed
rather than tunable, because a metric is only comparable between two
simulators if both were decoded the same way. The turn-2 suffix is appended to
the tutor turn of every multi-turn record before inference; it is what makes
the guidance question a request for an updated answer.

The repetition penalty applies to fidelity decoding only. Guidance decoding
leaves it at 1.0 in every domain. ``mode_field`` names the record field that
carries the guidance mode, which differs by domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class EvalProtocol:
    """How one domain decodes and scores."""

    domain: str
    max_new_tokens: int
    max_batch_size: int
    turn2_suffix: str
    mode_field: str
    fidelity_repetition_penalty: float = 1.0
    guidance_repetition_penalty: float = 1.0
    strip_think: bool = False
    temperature: float = 0.0


CHESS: Final = EvalProtocol(
    domain="chess",
    max_new_tokens=32,
    max_batch_size=64,
    strip_think=True,
    turn2_suffix="Given the above, what is your updated move for this position?",
    mode_field="instruction_type",
)

L2: Final = EvalProtocol(
    domain="l2",
    max_new_tokens=256,
    max_batch_size=32,
    fidelity_repetition_penalty=1.1,
    turn2_suffix=(
        "Reply with only the corrected text — no explanation, "
        "no rewriting the rest."
    ),
    mode_field="style",
)

MATH: Final = EvalProtocol(
    domain="math",
    max_new_tokens=32,
    max_batch_size=32,
    strip_think=True,
    turn2_suffix=(
        "Given the above, what is your updated answer? Respond with the "
        "answer text only (e.g., '27.5', '0.9c'), not a letter."
    ),
    mode_field="style",
)

_PROTOCOLS: Final = {p.domain: p for p in (CHESS, L2, MATH)}


def protocol_for(domain: str) -> EvalProtocol:
    """Return the evaluation protocol for ``domain``."""
    try:
        return _PROTOCOLS[domain]
    except KeyError:
        raise ValueError(f"unknown domain: {domain!r}") from None
