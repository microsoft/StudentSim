"""Turning a free-text problem into the multiple-choice form used for scoring.

The student's own recorded answer is one option and the rest are the answers
other students most often gave on the same problem, so a simulator is asked to
pick this student's answer out of answers real students actually wrote. The
students on the train and eval rosters never contribute options, so nobody's own
history leaks into their own choices.
"""

from __future__ import annotations

import csv
import html
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

LETTERS = "ABCDEFGH"

#: How many options a problem gets when enough distinct answers exist.
CHOICES = 4

#: Below this many the problem is left out, since a choice of one is no choice.
MIN_CHOICES = 2


def clean_candidate(text: str) -> str:
    """An answer as it should appear as an option.

    Markup and spacing differences would otherwise show up as separate
    options, so they are normalized away. Capitalization is kept, because
    students do write the same word differently.
    """
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def collect_answers(
    interactions_csv: Path,
    problems: set[str],
    exclude_students: set[str],
) -> dict[str, Counter]:
    """What every other student wrote on each problem, most common first."""
    by_problem: dict[str, Counter] = defaultdict(Counter)
    with open(interactions_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("user_id") in exclude_students:
                continue
            problem = str(row.get("problem_id"))
            if problem not in problems:
                continue
            answer = (row.get("answer_text") or "").strip()
            if answer:
                by_problem[problem][answer] += 1
    return dict(by_problem)


def build_choices(
    *,
    student_answer: str,
    problem_id: str,
    written_by_others: dict[str, Counter],
    seed: int,
    n_choices: int = CHOICES,
) -> tuple[dict[str, str], str] | None:
    """The lettered options for one record, and which letter is the student's.

    ``None`` when the problem has too few distinct answers to offer a choice.
    """
    correct = clean_candidate(student_answer)
    seen = {correct.lower()}
    distractors: list[str] = []
    for answer, _ in written_by_others.get(problem_id, Counter()).most_common():
        candidate = clean_candidate(answer)
        if not candidate or candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        distractors.append(candidate)

    available = min(n_choices, 1 + len(distractors))
    if available < MIN_CHOICES:
        return None

    options = [correct] + distractors[: available - 1]
    random.Random(seed).shuffle(options)
    lettered = {LETTERS[i]: option for i, option in enumerate(options)}
    return lettered, next(letter for letter, option in lettered.items() if option == correct)


def append_choices(prompt: str, lettered: dict[str, str]) -> str:
    """Put the options and the letter instruction after a free-text prompt."""
    body = re.sub(r"\s*Respond with your answer only\.\s*$", "", prompt).rstrip()
    listed = "\n".join(f"  {letter}) {option}" for letter, option in lettered.items())
    last = LETTERS[len(lettered) - 1]
    return (
        f"{body}\n\nMultiple-choice options:\n{listed}\n\n"
        f"Pick the letter (A-{last}) you would answer with. Respond with only the letter."
    )
