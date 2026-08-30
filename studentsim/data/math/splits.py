"""Dividing each student's work into what may be trained on and what is held out.

A student's attempts are ordered in time, and the split is chronological, so a
simulator is always asked to predict later work from earlier work. Every
student then contributes the same number of records, so that none of them
dominates a population mean.
"""

from __future__ import annotations

import csv
import html
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from studentsim.data.math.answers import repair_fraction

csv.field_size_limit(sys.maxsize)

#: The share of a student's attempts, earliest first, that may be trained on,
#: and the share reserved for validation. The remainder is held out.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10


@dataclass(slots=True)
class Attempt:
    interaction_id: str
    student_id: str
    problem_id: str
    answer_text: str
    correct: bool
    """Whether what the student wrote matches the platform's answer key.

    The platform's own score is not used: it marks an attempt wrong when the
    student asked for a hint, even if the answer they typed was right.
    """
    end_time: str


def normalize_answer(text: str) -> str:
    return html.unescape(text or "").strip()


def read_attempts(
    interactions_csv: Path,
    students: set[str],
    free_text: set[str],
    keys: dict[str, str],
) -> dict[str, list[Attempt]]:
    """Each student's free-text attempts, earliest first."""
    by_student: dict[str, list[Attempt]] = defaultdict(list)
    with open(interactions_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["user_id"] not in students:
                continue
            problem = str(row.get("problem_id"))
            if problem not in free_text:
                continue
            written = normalize_answer(row.get("answer_text"))
            by_student[row["user_id"]].append(
                Attempt(
                    interaction_id=str(row.get("id")),
                    student_id=row["user_id"],
                    problem_id=problem,
                    answer_text=written,
                    correct=written.lower() == keys.get(problem, "").lower(),
                    end_time=row.get("end_time") or "",
                )
            )
    for attempts in by_student.values():
        attempts.sort(key=lambda a: a.end_time)
    return dict(by_student)


def answer_keys(problems_csv: Path, *, repair: bool = False) -> dict[str, str]:
    """The platform's answer for every free-text problem that has one.

    A problem the platform scores some other way is left out, and so is one
    with no recorded answer, because neither can say whether a student was
    right. ``repair`` applies the date fix, which the split wants and the
    profile does not.
    """
    from studentsim.data.math.rosters import FREE_TEXT_TYPES

    keys: dict[str, str] = {}
    with open(problems_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("Answer Types") or "") not in FREE_TEXT_TYPES:
                continue
            raw = html.unescape((row.get("Fill-in Answers") or "").strip()).strip()
            if not raw:
                continue
            keys[str(row["problem_id"])] = (repair_fraction(raw) if repair else None) or raw
    return keys


def split_chronologically(attempts: list[Attempt]) -> dict[str, list[Attempt]]:
    """Earliest attempts train, latest are held out."""
    n_train = int(len(attempts) * TRAIN_FRACTION)
    n_val = int(len(attempts) * VAL_FRACTION)
    return {
        "train": attempts[:n_train],
        "val": attempts[n_train : n_train + n_val],
        "test": attempts[n_train + n_val :],
    }


def equalized(
    splits: dict[str, dict[str, list[Attempt]]],
    students: list[str],
    which: str,
) -> tuple[list[Attempt], int]:
    """The same number of records from every student, taken from the tail."""
    cap = min(len(splits[student][which]) for student in students)
    out: list[Attempt] = []
    for student in students:
        out.extend(splits[student][which][-cap:])
    return out, cap
