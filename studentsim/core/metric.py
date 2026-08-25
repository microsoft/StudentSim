"""What the two metrics have to provide, whatever domain they measure.

Fidelity asks whether the simulator answers as the student did. Responsiveness
asks whether it changes its answer correctly after reading a tutor's message.

A :class:`FidelityMetric` consumes :class:`SingleTurnRecord` plus model predictions
and emits a scalar per student and an aggregate. A :class:`GuidanceMetric` does
the same over :class:`MultiTurnRecord`, plus a breakdown keyed by
guidance mode, so responsiveness can be read per pedagogical style.

Per-domain implementations live under ``studentsim/domains/<domain>/{fidelity,guidance}.py``
and are returned by ``Domain.fidelity_metric()`` / ``Domain.guidance_metric()``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from studentsim.core.records import MultiTurnRecord, SingleTurnRecord


@runtime_checkable
class FidelityMetric(Protocol):
    """How closely the simulator's answer matches the student's own.

    Implementations are stateless: every method depends only on its arguments.
    """

    name: str
    """Short identifier for logging, e.g. ``"chess_uci_exact_match"``."""

    def score(
        self,
        *,
        records: Sequence[SingleTurnRecord],
        predictions: Sequence[str],
    ) -> float:
        """One score over all (record, prediction) pairs.

        The records and predictions are positionally aligned; ``len(records) ==
        len(predictions)`` must hold.
        """
        ...

    def per_student_score(
        self,
        *,
        records: Sequence[SingleTurnRecord],
        predictions: Sequence[str],
    ) -> Mapping[str, float]:
        """One score per student, keyed by ``student_id``.

        This is so a student with more records does not weigh more than one with fewer (average
        across students). Aggregation in :mod:`studentsim.eval.aggregate`
        consume this output.
        """
        ...


@runtime_checkable
class GuidanceMetric(Protocol):
    """How often the simulator updates its answer correctly
    after reading the tutor's message."""

    name: str

    def score(
        self,
        *,
        records: Sequence[MultiTurnRecord],
        predictions: Sequence[str],
    ) -> float: ...

    def per_student_score(
        self,
        *,
        records: Sequence[MultiTurnRecord],
        predictions: Sequence[str],
    ) -> Mapping[str, float]: ...

    def per_mode_score(
        self,
        *,
        records: Sequence[MultiTurnRecord],
        predictions: Sequence[str],
    ) -> Mapping[str, float]:
        """One score per guidance mode, keyed by ``GuidanceMode.name``."""
        ...

    def per_student_per_mode_score(
        self,
        *,
        records: Sequence[MultiTurnRecord],
        predictions: Sequence[str],
    ) -> Mapping[str, Mapping[str, float]]:
        """One score per student per guidance mode.

        Returned as ``{student_id: {mode_name: score}}``.
        """
        ...


def check_aligned(records: Sequence, predictions: Sequence[str]) -> None:
    """Raise if records and predictions are misaligned. Helper for impls."""
    if len(records) != len(predictions):
        raise ValueError(
            f"records and predictions misaligned: "
            f"{len(records)} records vs {len(predictions)} predictions"
        )
