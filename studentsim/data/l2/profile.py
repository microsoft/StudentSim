"""The rolling picture of a learner that conditions their prompt.

A learner's profile is computed from the essays they wrote *before* the one
being predicted, so a simulator never sees the future when it is asked to
write as that learner.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

#: Error categories shown in the profile, most frequent first.
TOP_ERRORS = 5

#: Worked wrong-to-right pairs kept per category.
PAIRS_PER_ERROR = 2

#: Pairs longer than this are dropped, to keep the profile block readable.
MAX_PAIR_CHARS = 30


def cefr_for_level(level: int) -> str:
    """EFCAMDAT level 1-15 to its approximate CEFR band."""
    if level <= 3:
        return "A1"
    if level <= 6:
        return "A2"
    if level <= 9:
        return "B1"
    if level <= 12:
        return "B2"
    if level <= 14:
        return "C1"
    return "C2"


@dataclass(frozen=True, slots=True)
class LearnerProfile:
    n_essays: int
    avg_grade: float | None
    avg_wordcount: float
    current_level: int | None
    cefr: str | None
    nationality: str
    common_errors: list[tuple[str, int]] = field(default_factory=list)
    """(symbol, count) for the most frequent categories, most frequent first."""
    error_examples: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    """symbol -> [(what the learner wrote, what it should have been), ...]"""


def compute_profile(earlier_essays: list[dict]) -> LearnerProfile | None:
    """Summarize what a learner has written so far.

    ``None`` when this is the learner's first essay, which the prompt then
    says outright.
    """
    if not earlier_essays:
        return None

    grades = [e["grade"] for e in earlier_essays if e["grade"] > 0]
    wordcounts = [len(e["original"].split()) for e in earlier_essays]
    levels = [e["level"] for e in earlier_essays if e["level"] > 0]

    counts: Counter[str] = Counter()
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for essay in earlier_essays:
        for span in essay["spans"]:
            if not span.symbol:
                continue
            counts[span.symbol] += 1
            wrong = span.selection.strip()
            right = span.correct.strip()
            # A missing-word correction has no wrong text to show, and a very
            # long pair would crowd out the rest of the profile.
            if not wrong or len(wrong) > MAX_PAIR_CHARS or len(right) > MAX_PAIR_CHARS:
                continue
            pairs[span.symbol].append((wrong, right))

    examples: dict[str, list[tuple[str, str]]] = {}
    for symbol, _ in counts.most_common(TOP_ERRORS):
        seen: set[tuple[str, str]] = set()
        picked: list[tuple[str, str]] = []
        for wrong, right in pairs[symbol]:
            key = (wrong.lower(), right.lower())
            if key in seen:
                continue
            seen.add(key)
            picked.append((wrong, right))
            if len(picked) >= PAIRS_PER_ERROR:
                break
        if picked:
            examples[symbol] = picked

    top_level = max(levels) if levels else None
    return LearnerProfile(
        n_essays=len(earlier_essays),
        avg_grade=sum(grades) / len(grades) if grades else None,
        avg_wordcount=sum(wordcounts) / len(wordcounts) if wordcounts else 0.0,
        current_level=top_level,
        cefr=cefr_for_level(top_level) if top_level else None,
        nationality=earlier_essays[-1].get("nationality", ""),
        common_errors=counts.most_common(TOP_ERRORS),
        error_examples=examples,
    )
