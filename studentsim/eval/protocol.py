"""Per-domain evaluation protocol.

These values define how a simulator is decoded and scored. They are fixed
rather than tunable, because a metric is only comparable between two
simulators if both were decoded the same way. The turn-2 suffix, where a domain
has one, is appended to the tutor turn of every multi-turn record before
inference; it is what makes the guidance question a request for an updated
answer. Math has none, because its tutor turn already ends by asking for a
letter.

The repetition penalty applies to fidelity decoding only. Guidance decoding
leaves it at 1.0 in every domain. ``mode_field`` names the record field that
carries the guidance mode.

Chess and L2 decode guidance with ``swift infer``. Math decodes it with
HuggingFace ``generate`` on the merged adapter, one record at a time, with
the first three messages as the chat; ``guidance_backend`` says which.
``score_decimals`` rounds each student's score before students are averaged,
and ``empty_is_miss`` keeps an empty decode from matching an empty reference.
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
    guidance_backend: str = "swift"
    score_decimals: int | None = None
    empty_is_miss: bool = False


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
    mode_field="instruction_type",
)

MATH: Final = EvalProtocol(
    domain="math",
    max_new_tokens=128,
    max_batch_size=1,
    strip_think=True,
    turn2_suffix="",
    mode_field="instruction_type",
    guidance_backend="hf",
    score_decimals=4,
    empty_is_miss=True,
)

_PROTOCOLS: Final = {p.domain: p for p in (CHESS, L2, MATH)}


def protocol_for(domain: str) -> EvalProtocol:
    """Return the evaluation protocol for ``domain``."""
    try:
        return _PROTOCOLS[domain]
    except KeyError:
        raise ValueError(f"unknown domain: {domain!r}") from None
