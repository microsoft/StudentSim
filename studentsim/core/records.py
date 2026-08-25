"""Record dataclasses: the lingua franca of the pipeline.

Single-turn records (:class:`SingleTurnRecord`) describe a problem and the actual
student response. Multi-turn records (:class:`MultiTurnRecord`) extend the same
prefix with the student's wrong response, the tutor's message, and the answer
the student should arrive at after reading it. Both share a :class:`StudentProfile` and may
carry per-domain extras in ``meta``.

All record types are frozen dataclasses: immutable, hashable, safe to share
across processes. They are pure Python (no torch / pandas) so they can be
imported in lightweight contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

_EMPTY_META: Mapping[str, Any] = MappingProxyType({})


def _frozen_meta(meta: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a read-only view of a meta dict (or the shared empty view)."""
    if not meta:
        return _EMPTY_META
    if isinstance(meta, MappingProxyType):
        return meta
    return MappingProxyType(dict(meta))


@dataclass(frozen=True, slots=True)
class StudentProfile:
    """Per-student feature block included in every prompt.

    ``features`` is a domain-specific dict and is the only place where chess
    player context, L2 learner CEFR / nationality / history, or math per-skill
    accuracies live. The core never inspects its contents; per-domain prompt
    renderers do.
    """

    student_id: str
    domain: str
    features: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_META)

    def __post_init__(self) -> None:  # normalize features to a read-only view
        object.__setattr__(self, "features", _frozen_meta(self.features))


@dataclass(frozen=True, slots=True)
class GuidanceMode:
    """A pedagogical style label, scoped to a domain.

    The full set of modes a domain supports is declared by ``Domain.modes``.
    A ``GuidanceMode`` is essentially a (name, domain) pair and exists as a
    type-safe alternative to bare strings.
    """

    name: str
    domain: str


@dataclass(frozen=True, slots=True)
class SingleTurnRecord:
    """A (problem, response) pair from one student on one held-out item.

    Used to score fidelity. The ``response`` field is the student's own answer,
    which may be wrong; the metric asks how closely the simulator matches it,
    not whether either of them is correct.
    """

    record_id: str
    student_id: str
    domain: str
    problem: str
    response: str
    profile: StudentProfile
    meta: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_META)

    def __post_init__(self) -> None:
        if self.profile.student_id != self.student_id:
            raise ValueError(
                f"profile.student_id ({self.profile.student_id!r}) "
                f"does not match record.student_id ({self.student_id!r})"
            )
        if self.profile.domain != self.domain:
            raise ValueError(
                f"profile.domain ({self.profile.domain!r}) "
                f"does not match record.domain ({self.domain!r})"
            )
        object.__setattr__(self, "meta", _frozen_meta(self.meta))


@dataclass(frozen=True, slots=True)
class MultiTurnRecord:
    """A (problem, wrong response, tutor guidance, canonical correction) tuple.

    Used to score responsiveness. After reading ``tutor_guidance`` the simulator
    should produce ``reference_response``; the metric asks how often it does.

    The ``mode`` field is the pedagogical style ``tutor_guidance`` was written
    in, which is what lets the score be split by style.
    """

    record_id: str
    student_id: str
    domain: str
    problem: str
    wrong_response: str
    tutor_guidance: str
    reference_response: str
    mode: GuidanceMode
    profile: StudentProfile
    meta: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_META)

    def __post_init__(self) -> None:
        if self.profile.student_id != self.student_id:
            raise ValueError(
                f"profile.student_id ({self.profile.student_id!r}) "
                f"does not match record.student_id ({self.student_id!r})"
            )
        if self.profile.domain != self.domain:
            raise ValueError(
                f"profile.domain ({self.profile.domain!r}) "
                f"does not match record.domain ({self.domain!r})"
            )
        if self.mode.domain != self.domain:
            raise ValueError(
                f"mode.domain ({self.mode.domain!r}) "
                f"does not match record.domain ({self.domain!r})"
            )
        object.__setattr__(self, "meta", _frozen_meta(self.meta))
