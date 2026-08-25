"""The rolling picture of a student that conditions their prompt.

Everything here is computed from the attempts a student made *before* the one
being predicted, so a simulator never sees the future when it is asked to
answer as that student.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

#: How many recent attempts the second accuracy figure covers.
RECENT_WINDOW = 10

#: Worked attempts shown per skill of the current problem.
ATTEMPTS_PER_SKILL = 6


@dataclass(frozen=True, slots=True)
class StudentProfile:
    accuracy: float
    recent_accuracy: float
    n_attempts: int
    skill_attempts: dict[str, int] = field(default_factory=dict)
    skill_correct: dict[str, int] = field(default_factory=dict)


def primary_skill(problem_id: str, skills: dict[str, list[str]]) -> str:
    """The first skill listed for a problem, or a catch-all when it has none."""
    listed = skills.get(problem_id) or []
    return listed[0] if listed else "Other"


def compute_profile(earlier, skills: dict[str, list[str]]) -> StudentProfile | None:
    """Summarize how the student has done so far.

    ``None`` when this is their first recorded attempt, which the prompt then
    simply omits.
    """
    if not earlier:
        return None

    correct = sum(1 for attempt in earlier if attempt.correct)
    recent = earlier[-RECENT_WINDOW:]
    recent_correct = sum(1 for attempt in recent if attempt.correct)

    attempts_by_skill: Counter[str] = Counter()
    correct_by_skill: Counter[str] = Counter()
    for attempt in earlier:
        skill = primary_skill(attempt.problem_id, skills)
        attempts_by_skill[skill] += 1
        if attempt.correct:
            correct_by_skill[skill] += 1

    return StudentProfile(
        accuracy=correct / len(earlier),
        recent_accuracy=recent_correct / len(recent) if recent else correct / len(earlier),
        n_attempts=len(earlier),
        skill_attempts=dict(attempts_by_skill),
        skill_correct=dict(correct_by_skill),
    )


def recent_on_skills(earlier, targets: list[str], skills: dict[str, list[str]]) -> dict[str, list]:
    """The student's latest attempts on each skill the current problem exercises.

    Walking backwards and stopping once every skill has its quota keeps the
    block short on students with long histories.
    """
    wanted = set(targets)
    collected: dict[str, list] = {skill: [] for skill in targets}
    for attempt in reversed(earlier):
        skill = primary_skill(attempt.problem_id, skills)
        if skill in wanted and len(collected[skill]) < ATTEMPTS_PER_SKILL:
            collected[skill].append(attempt)
        if all(len(v) >= ATTEMPTS_PER_SKILL for v in collected.values()):
            break
    for entries in collected.values():
        entries.reverse()
    return collected
