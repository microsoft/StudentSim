"""L2 fidelity: how alike two essays are in the errors they contain.

Two essays on the same task are never the same string, so they are compared by
the mistakes in them rather than by their words. Each essay is run through
LanguageTool and its issues counted per hundred words, in seven buckets.

An essay scores half for matching the learner's overall error rate and half for
matching the mix across buckets, so writing the right number of mistakes of the
wrong kinds earns about half marks. Each comparison uses the same similarity:
one when two densities are equal, falling towards zero as one dwarfs the other,
and one again when both are zero, since two clean essays agree.

The LanguageTool dependency is injected through the :class:`IssueCounter`
Protocol so the rest of the package and the unit tests can use a fake counter
without requiring a Java runtime. The production counter
(:class:`LanguageToolCounter`) is lazily importable via
:mod:`studentsim.domains.l2.languagetool`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from studentsim.core.metric import check_aligned
from studentsim.core.records import SingleTurnRecord

LT_BUCKETS: Final[tuple[str, ...]] = (
    "misspelling",
    "grammar",
    "typographical",
    "style",
    "uncategorized",
    "whitespace",
    "inconsistency",
)
"""Seven LanguageTool issue-type buckets, in the canonical paper order."""


@dataclass(frozen=True, slots=True)
class IssueCounts:
    """Per-essay LanguageTool counts.

    ``per_bucket`` is keyed by every name in :data:`LT_BUCKETS`; absent buckets
    must be present with value 0 so downstream math sees a fixed-width vector.
    ``word_count`` is the essay's word count (whitespace-delimited).
    """

    per_bucket: Mapping[str, int]
    word_count: int

    def __post_init__(self) -> None:
        missing = set(LT_BUCKETS) - set(self.per_bucket)
        if missing:
            raise ValueError(f"IssueCounts missing buckets: {sorted(missing)}")
        if self.word_count < 0:
            raise ValueError(f"word_count must be non-negative, got {self.word_count}")

    def density_per_100_words(self, bucket: str) -> float:
        """Issues of ``bucket`` per 100 words. Zero if word_count == 0."""
        if self.word_count == 0:
            return 0.0
        return 100.0 * self.per_bucket[bucket] / self.word_count

    def total_density_per_100_words(self) -> float:
        if self.word_count == 0:
            return 0.0
        return 100.0 * sum(self.per_bucket.values()) / self.word_count


@runtime_checkable
class IssueCounter(Protocol):
    """Pluggable LanguageTool wrapper.

    Production: :class:`studentsim.domains.l2.languagetool.LanguageToolCounter`.
    Tests: an in-memory fake that returns canned counts.
    """

    def count(self, text: str) -> IssueCounts: ...


def density_kernel(a: float, b: float) -> float:
    """How alike two densities are: 1 when equal, falling towards 0 as one
    dwarfs the other, and 1 again when both are zero, since two essays with no
    errors of a kind agree about that kind."""
    if a == 0.0 and b == 0.0:
        return 1.0
    denom = max(a, b)
    return 1.0 - abs(a - b) / denom


def per_record_density_match(
    *,
    predicted: IssueCounts,
    reference: IssueCounts,
) -> float:
    """One essay's score: half total-error agreement, half per-category agreement."""
    total_term = density_kernel(
        predicted.total_density_per_100_words(),
        reference.total_density_per_100_words(),
    )
    per_bucket_terms = [
        density_kernel(
            predicted.density_per_100_words(b),
            reference.density_per_100_words(b),
        )
        for b in LT_BUCKETS
    ]
    return 0.5 * total_term + 0.5 / len(LT_BUCKETS) * sum(per_bucket_terms)


@dataclass(frozen=True, slots=True)
class L2Fidelity:
    """Error-pattern-density L2 fidelity metric.

    Parameters
    ----------
    counter
        :class:`IssueCounter` used to count LanguageTool issues. Both the
        predicted essay and the recorded reference essay are counted by the
        same counter, so the comparison is symmetric.
    """

    counter: IssueCounter

    name: str = "l2_lt_issue_density"

    def score(
        self,
        *,
        records: Sequence[SingleTurnRecord],
        predictions: Sequence[str],
    ) -> float:
        check_aligned(records, predictions)
        if not records:
            return 0.0
        total = 0.0
        for rec, pred in zip(records, predictions):
            total += per_record_density_match(
                predicted=self.counter.count(pred),
                reference=self.counter.count(rec.response),
            )
        return total / len(records)

    def per_student_score(
        self,
        *,
        records: Sequence[SingleTurnRecord],
        predictions: Sequence[str],
    ) -> Mapping[str, float]:
        check_aligned(records, predictions)
        buckets: dict[str, list[float]] = defaultdict(list)
        for rec, pred in zip(records, predictions):
            score = per_record_density_match(
                predicted=self.counter.count(pred),
                reference=self.counter.count(rec.response),
            )
            buckets[rec.student_id].append(score)
        return {sid: sum(vs) / len(vs) for sid, vs in buckets.items()}
