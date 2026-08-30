"""Choosing which students the two stages train on.

Selection runs on the corpus itself, so a rebuild lands on the same students
without any identifier being carried in this repository. Ordering by how much
free-text work a student did keeps every selected student rich enough to both
train on and hold out, and the shuffle inside the leading group avoids taking
only the very heaviest users.
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from studentsim.core.seeds import TRAINER_SEED

csv.field_size_limit(sys.maxsize)

#: Answer types where the student typed the answer. Platform-side multiple
#: choice is left out, because the distractors come from what other
#: students actually wrote.
FREE_TEXT_TYPES = {
    "Numeric",
    "Algebraic Expression",
    "Exact Match",
    "Exact Fraction",
    "Numeric Expression",
    "Ordering",
    "Drop Down",
    "Algebraic Expression, Numeric",
    "Exact Fraction, Exact Match",
}

#: The leading group the specialization students are drawn from.
POOL_SIZE = 25

SPECIALIZE_STUDENTS = 15
POOLED_STUDENTS = 200

SEED = TRAINER_SEED


def free_text_problems(problems_csv: Path) -> set[str]:
    """Problem ids the platform scores as free text.

    A problem can appear once per part, and the file gives the answer type per
    part. The type of the last part listed is the one that decides, matching
    how the reference build read the file.
    """
    answer_type: dict[str, str] = {}
    with open(problems_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            answer_type[str(row["problem_id"])] = row.get("Answer Types") or ""
    return {pid for pid, kind in answer_type.items() if kind in FREE_TEXT_TYPES}


def count_free_text(interactions_csv: Path, free_text: set[str]) -> dict[str, int]:
    """How many free-text problems each student attempted."""
    counts: dict[str, int] = defaultdict(int)
    with open(interactions_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            # Every student is entered on their first row, whatever it was, so
            # that students tied on this count keep the order the file gives.
            tally = counts[row["user_id"]]
            if str(row.get("problem_id")) in free_text:
                counts[row["user_id"]] = tally + 1
    return counts


def choose_students(counts: dict[str, int]) -> tuple[list[str], list[str]]:
    """The specialization roster and the pooled roster that contains it."""
    ranked = [student for student, _ in sorted(counts.items(), key=lambda kv: -kv[1])]

    leading = ranked[:POOL_SIZE]
    random.Random(SEED).shuffle(leading)
    specialize = leading[:SPECIALIZE_STUDENTS]

    remaining = POOLED_STUDENTS - SPECIALIZE_STUDENTS
    rest = [s for s in ranked[POOL_SIZE : POOL_SIZE + remaining] if s not in set(specialize)]
    random.Random(SEED).shuffle(rest)
    return specialize, specialize + rest[:remaining]
