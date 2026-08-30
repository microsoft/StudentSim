"""Rendering a student's prompt.

The prompt shows who the student is, how they have done overall and lately,
how they have done on each skill, and what they wrote on their most recent
attempts at the skills the current problem exercises. The problem itself comes
last, so that everything before it reads as context.
"""

from __future__ import annotations

from studentsim.data.math.problems import ANSWER_TYPE_HINTS
from studentsim.data.math.profile import (
    StudentProfile,
    recent_on_skills,
)

#: How much of a past problem is quoted back, in characters.
PROBLEM_PREVIEW_CHARS = 300


def _render_attempt(attempt, bodies: dict[str, str], keys: dict[str, str]) -> str:
    body = (bodies.get(attempt.problem_id) or "").replace("\n", " | ")[:PROBLEM_PREVIEW_CHARS]
    mark = "✓" if attempt.correct else f"✗ correct={keys.get(attempt.problem_id, '?')}"
    return f"    [{mark}] {body}  →  wrote: {attempt.answer_text!r}"


def _render_recent(collected: dict[str, list], bodies, keys) -> str:
    lines = []
    for skill, attempts in collected.items():
        if not attempts:
            continue
        n_correct = sum(1 for a in attempts if a.correct)
        lines.append(f"  {skill} ({n_correct}/{len(attempts)} correct in most recent attempts):")
        lines.extend(_render_attempt(a, bodies, keys) for a in attempts)
    return "\n".join(lines) if lines else "  (no prior attempts on these skills)"


def build_user_message(
    *,
    student_name: str,
    profile: StudentProfile | None,
    current,
    earlier: list,
    skills: dict[str, list[str]],
    bodies: dict[str, str],
    keys: dict[str, str],
    answer_types: dict[str, str],
) -> str:
    """The prompt a simulator answers as this student."""
    current_skills = skills.get(current.problem_id) or ["Other"]

    lines = ["You are simulating a student doing math practice.", ""]
    if profile is not None:
        lines.append("Student profile:")
        lines.append(f"  Student: {student_name}")
        lines.append(
            f"  Overall accuracy: {profile.accuracy:.1%}  "
            f"(recent 10: {profile.recent_accuracy:.1%})"
        )
        if profile.skill_attempts:
            lines.append("  Per-skill performance (correct / attempted):")
            for skill, attempted in sorted(profile.skill_attempts.items(), key=lambda kv: -kv[1]):
                got = profile.skill_correct.get(skill, 0)
                lines.append(f"    {skill}: {got}/{attempted} ({got / attempted:.0%})")

    collected = recent_on_skills(earlier, current_skills, skills)
    if any(collected.values()):
        lines.append("")
        lines.append("Most recent attempts on the skill(s) of the current problem:")
        lines.append(_render_recent(collected, bodies, keys))

    lines.append("")
    lines.append("Now solve this problem:")
    lines.append(f"[Skill: {', '.join(current_skills)}]")
    lines.append(bodies.get(current.problem_id, ""))
    hint = ANSWER_TYPE_HINTS.get(answer_types.get(current.problem_id, ""), "")
    if hint:
        lines.append(hint)
    lines += ["", "Respond with your answer only."]
    return "\n".join(lines)
