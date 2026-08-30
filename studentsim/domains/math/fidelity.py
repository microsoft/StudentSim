"""Math behavioral fidelity: top-1 accuracy on a four-way multiple-choice form.

A record scores 1 when the letter the simulator ranks highest is the letter
the student's own answer maps to under that record's shuffle, and 0 otherwise.
Per-student scores are the mean over their records.

The letter is read from the simulator's log-probabilities rather than from
free text, because a closed model may decline to emit a bare letter while
still ranking one. That also means a provider returning only its top few
tokens omits the rest, so a letter it did not rank is floored rather than
treated as impossible.

:func:`renormalized_log_likelihood` is a diagnostic helper; the score itself
is the argmax above.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from studentsim.core.metric import check_aligned
from studentsim.core.records import SingleTurnRecord

MATH_LETTERS: Final[tuple[str, ...]] = ("A", "B", "C", "D")
"""Canonical four-way alphabet, used by both the metric and the runner."""

LOGPROB_FLOOR: Final[float] = -10.0
"""Letters absent from a baseline's top-k returned logprobs get this floor."""


def argmax_letter(
    logprobs: Mapping[str, float],
    *,
    candidate_letters: Sequence[str] = MATH_LETTERS,
) -> str:
    """Pick the letter in ``candidate_letters`` with the highest logprob.

    By default the argmax is taken over all four :data:`MATH_LETTERS`; pass
    ``candidate_letters`` to restrict it to the letters a record actually
    offers, for records built with fewer distractors. Their order is the
    tie-break order.

    A letter absent from ``logprobs`` is floored at :data:`LOGPROB_FLOOR`
    rather than treated as impossible. A provider that returns only its
    highest-ranked few tokens omits the rest, and reading an omission as
    minus infinity would leave the argmax undefined when all four are missing.
    """
    if not candidate_letters:
        raise ValueError("candidate_letters must be non-empty")
    best_letter = candidate_letters[0]
    best_lp = logprobs.get(best_letter, LOGPROB_FLOOR)
    for letter in candidate_letters[1:]:
        lp = logprobs.get(letter, LOGPROB_FLOOR)
        if lp > best_lp:
            best_lp = lp
            best_letter = letter
    return best_letter


def renormalized_log_likelihood(
    *,
    logprobs: Mapping[str, float],
    target_letter: str,
) -> float:
    """Per-record renormalized log-likelihood (diagnostic helper, not the
    score itself).

    Returns ``log P(target_letter | x) - log sum_{l in MATH_LETTERS} P(l | x)``
    with missing letters floored at :data:`LOGPROB_FLOOR`. The score
    metric is top-1 accuracy via :func:`argmax_letter`; this helper is kept
    for users who want the log-likelihood signal alongside the accuracy.
    """
    if target_letter not in MATH_LETTERS:
        raise ValueError(f"target_letter must be one of {MATH_LETTERS}, got {target_letter!r}")
    target_lp = logprobs.get(target_letter, LOGPROB_FLOOR)
    floored = [logprobs.get(letter, LOGPROB_FLOOR) for letter in MATH_LETTERS]
    log_denom = _logsumexp(floored)
    return target_lp - log_denom


def _logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not values:
        return float("-inf")
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


@dataclass(frozen=True, slots=True)
class MathFidelity:
    """Math fidelity metric: top-1 accuracy on the four-way MC form.

    Predictions are per-record argmax letters ("A"/"B"/"C"/"D") produced by
    the evaluation runner from a
    :class:`Simulator`'s logprobs over the A/B/C/D alphabet. Each record's
    per-instance score is 1 if the prediction equals ``record.response``
    (the recorded letter ``l*``) and 0 otherwise; the population score is
    the mean across all records and ``per_student_score`` is the per-student
    mean.
    """

    name: str = "math_mc_top1_accuracy"

    def score(
        self,
        *,
        records: Sequence[SingleTurnRecord],
        predictions: Sequence[str],
    ) -> float:
        check_aligned(records, predictions)
        if not records:
            return 0.0
        correct = sum(
            1 for rec, pred in zip(records, predictions)
            if _validate_letter(pred) == rec.response
        )
        return correct / len(records)

    def per_student_score(
        self,
        *,
        records: Sequence[SingleTurnRecord],
        predictions: Sequence[str],
    ) -> Mapping[str, float]:
        check_aligned(records, predictions)
        buckets: dict[str, list[int]] = defaultdict(list)
        for rec, pred in zip(records, predictions):
            buckets[rec.student_id].append(
                int(_validate_letter(pred) == rec.response)
            )
        return {sid: sum(vs) / len(vs) for sid, vs in buckets.items()}


def _validate_letter(pred: str) -> str:
    if pred not in MATH_LETTERS:
        raise ValueError(
            f"MathFidelity expects predictions to be letters in {MATH_LETTERS}; "
            f"got {pred!r}"
        )
    return pred
