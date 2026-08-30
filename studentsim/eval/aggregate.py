"""Aggregating per-student scores into a domain-level number.

Both metrics aggregate as the mean over students, so every student counts the
same regardless of how many held-out records they contributed. ``mean_micro``
carries the record-weighted alternative, which differs only where students
contribute unequal numbers of held-out records.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from studentsim.eval.runner import StudentResult


@dataclass(frozen=True)
class Aggregate:
    """A metric summarized across students."""

    metric: str
    domain: str
    mean: float
    std: float
    n_students: int
    mean_micro: float = 0.0
    per_student: dict[str, float] = field(default_factory=dict)
    per_mode: dict[str, float] = field(default_factory=dict)


def _summarize(
    metric: str,
    domain: str,
    per_student: dict[str, float],
    per_mode: dict[str, list[float]],
    weights: dict[str, int] | None = None,
) -> Aggregate:
    values = list(per_student.values())
    micro = 0.0
    if values and weights:
        total = sum(weights.get(s, 0) for s in per_student)
        if total:
            micro = sum(a * weights.get(s, 0) for s, a in per_student.items()) / total
    return Aggregate(
        metric=metric,
        domain=domain,
        mean=statistics.mean(values) if values else 0.0,
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        mean_micro=micro,
        n_students=len(values),
        per_student=per_student,
        per_mode={mode: statistics.mean(v) for mode, v in sorted(per_mode.items()) if v},
    )


def aggregate_fidelity(results: Sequence[StudentResult], *, domain: str) -> Aggregate:
    """Mean behavioral fidelity over the students that were scored."""
    scored = [r for r in results if r.fidelity is not None]
    per_student = {r.student_id: r.fidelity.accuracy for r in scored}
    weights = {r.student_id: r.fidelity.n_samples for r in scored}
    return _summarize("fidelity", domain, per_student, {}, weights)


def aggregate_responsiveness(results: Sequence[StudentResult], *, domain: str) -> Aggregate:
    """Mean guidance responsiveness over the students that were scored.

    The per-mode entry is the mean over students of each student's score on
    that guidance mode.
    """
    per_student: dict[str, float] = {}
    per_mode: dict[str, list[float]] = {}
    weights: dict[str, int] = {}
    for result in results:
        if result.responsiveness is None:
            continue
        per_student[result.student_id] = result.responsiveness.accuracy
        weights[result.student_id] = result.responsiveness.n_samples
        for mode, score in result.responsiveness.per_mode.items():
            per_mode.setdefault(mode, []).append(score.accuracy)
    return _summarize("responsiveness", domain, per_student, per_mode, weights)


@dataclass(frozen=True)
class CrossSeed:
    """One metric summarized across independent training runs."""

    metric: str
    domain: str
    mean: float
    std: float
    n_seeds: int
    per_seed: dict[str, float] = field(default_factory=dict)
    per_mode_mean: dict[str, float] = field(default_factory=dict)
    per_mode_std: dict[str, float] = field(default_factory=dict)


def aggregate_across_seeds(
    per_seed: Mapping[str, Aggregate], *, micro: bool = False
) -> CrossSeed:
    """Summarize one metric over the seeds that were run.

    A set of seeds is summarised as the mean over seeds of each
    seed's own score, with the sample standard deviation as the run-to-run
    spread. Pass ``micro=True`` to summarize the record-weighted score instead.
    """
    if not per_seed:
        raise ValueError("aggregate_across_seeds needs at least one seed")
    first = next(iter(per_seed.values()))
    scores = {
        tag: (aggregate.mean_micro if micro else aggregate.mean)
        for tag, aggregate in per_seed.items()
    }
    values = list(scores.values())

    modes: dict[str, list[float]] = {}
    for aggregate in per_seed.values():
        for mode, value in aggregate.per_mode.items():
            modes.setdefault(mode, []).append(value)

    return CrossSeed(
        metric=first.metric,
        domain=first.domain,
        mean=statistics.mean(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        n_seeds=len(values),
        per_seed=scores,
        per_mode_mean={m: statistics.mean(v) for m, v in sorted(modes.items())},
        per_mode_std={
            m: (statistics.stdev(v) if len(v) > 1 else 0.0) for m, v in sorted(modes.items())
        },
    )
