"""Build the math training and evaluation sets from a FoundationalASSIST extract.

The students, their split, and every record are derived from the extract
itself, so two builds of the same extract give identical record sets.
A student's hashed identifier seeds their draws and is then replaced by a
positional name, so nothing in the output identifies anyone.

    python -m studentsim.data.build_math
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import zlib
from pathlib import Path

from studentsim.core.seeds import DATA_SAMPLER_SEED
from studentsim.data.math.choices import append_choices, build_choices, collect_answers
from studentsim.data.math.problems import clean_text
from studentsim.data.math.profile import compute_profile
from studentsim.data.math.prompts import build_user_message
from studentsim.data.math.rosters import (
    choose_students,
    count_free_text,
    free_text_problems,
)
from studentsim.data.math.splits import (
    answer_keys,
    equalized,
    read_attempts,
    split_chronologically,
)

csv.field_size_limit(sys.maxsize)

#: A student's first attempts are context for the ones that follow, so no
#: record is built from them.
CONTEXT_ATTEMPTS = 10

SEED = DATA_SAMPLER_SEED


def load_problems(problems_csv: Path, keys: dict[str, str]) -> tuple[dict, dict]:
    """Problem text and answer type, for the problems that have an answer key."""
    bodies: dict[str, str] = {}
    answer_types: dict[str, str] = {}
    with open(problems_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            problem = str(row["problem_id"])
            if problem not in keys:
                continue
            bodies[problem] = clean_text(row.get("Problem Body") or "")
            answer_types[problem] = row.get("Answer Types") or ""
    return bodies, answer_types


def load_skills(skills_csv: Path) -> dict[str, list[str]]:
    """The skills each problem exercises, in the order the file lists them."""
    skills: dict[str, list[str]] = collections.defaultdict(list)
    with open(skills_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            skills[str(row["problem_id"])].append(row.get("node_name") or "")
    return dict(skills)


def _write(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(records):,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m studentsim.data.math.build")
    parser.add_argument("--raw", type=Path, default=Path("data/math/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/math"))
    parser.add_argument(
        "--exclude",
        type=Path,
        help="JSON list of problem ids the answer audit rejected. "
        "Run studentsim.data.math.audit to produce it.",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Build without an audit, taking every answer key as correct. "
        "Students are then scored against keys known to contain errors.",
    )
    args = parser.parse_args(argv)

    interactions = args.raw / "Interactions.csv"
    problems_csv = args.raw / "Problems.csv"

    # Building without an audit is a choice, not a default. Some keys in the
    # source are wrong, so taking them all as correct scores students against
    # answers that are not the right ones, and it does so silently: the build
    # succeeds and the corpus looks fine.
    rejected: set[str] = set()
    if args.exclude and args.exclude.is_file():
        rejected = {str(pid) for pid in json.loads(args.exclude.read_text())}
        print(f"Excluding {len(rejected):,} problems the audit rejected")
    elif args.no_audit:
        print("Building with no audit: every answer key is taken as correct")
    else:
        missing = f"no audit result at {args.exclude}" if args.exclude else "no audit result given"
        raise SystemExit(
            f"{missing}. Some answer keys in the source are wrong, so a build "
            "that skips this step scores students against answers that are not "
            "the right ones.\n"
            "Produce one with:\n"
            "  python -m studentsim.data.math.audit "
            f"--raw {args.raw} --out data/math/audit_excluded.json\n"
            "then pass it with --exclude, or pass --no-audit to build anyway. "
            "See data/math/raw/README.md for what the audit does and what it costs."
        )

    print("Choosing students ...")
    free_text = free_text_problems(problems_csv)
    specialize, pooled = choose_students(count_free_text(interactions, free_text))
    print(f"  {len(pooled)} for pooled training, of which {len(specialize)} specialize")

    # The key as the platform recorded it is what a student was scored against,
    # so that is what decides whether they were right and what the prompt shows.
    keys = answer_keys(problems_csv)
    # A problem whose key was repaired is trusted, since the repair is exact, so
    # the audit's verdict on its corrupted form does not apply.
    fixed = answer_keys(problems_csv, repair=True)
    repaired = {p for p, key in fixed.items() if keys.get(p) != key}
    usable = {p for p in keys if p not in (rejected - repaired)}
    bodies, answer_types = load_problems(problems_csv, keys)
    skills = load_skills(args.raw / "Skills.csv")

    print("Reading attempts ...")
    # A student's whole record conditions their prompt, including work on
    # problems the audit rejected: they still did that work, and it still says
    # something about them. The rejection only keeps a problem from being one
    # a simulator is asked to answer.
    attempts = read_attempts(interactions, set(pooled), set(keys), keys)
    splits = {
        student: split_chronologically([a for a in attempts[student] if a.problem_id in usable])
        for student in pooled
    }

    pooled_train, cap_pooled = equalized(splits, pooled, "train")
    specialize_train, cap_specialize = equalized(splits, specialize, "train")
    validation, cap_val = equalized(splits, specialize, "val")
    held_out, cap_test = equalized(splits, specialize, "test")
    print(
        f"  per student: {cap_pooled} pooled, {cap_specialize} specialization, "
        f"{cap_val} validation, {cap_test} held out"
    )

    names = {student: f"student_{i:02d}" for i, student in enumerate(pooled)}
    order = {student: {a.interaction_id: i for i, a in enumerate(attempts[student])} for student in pooled}
    written_by_others = collect_answers(interactions, usable, set(pooled))

    def record(attempt) -> dict | None:
        """One attempt as a multiple-choice record, or ``None`` if it has no choices."""
        history = attempts[attempt.student_id]
        index = order[attempt.student_id][attempt.interaction_id]
        if index < CONTEXT_ATTEMPTS:
            return None
        choices = build_choices(
            student_answer=attempt.answer_text,
            problem_id=attempt.problem_id,
            written_by_others=written_by_others,
            seed=SEED ^ (zlib.crc32(attempt.interaction_id.encode()) & 0xFFFFFFFF),
        )
        if choices is None:
            return None
        lettered, letter = choices
        prompt = build_user_message(
            student_name=names[attempt.student_id],
            profile=compute_profile(history[:index], skills),
            current=attempt,
            earlier=history[:index],
            skills=skills,
            bodies=bodies,
            keys=keys,
            answer_types=answer_types,
        )
        return {
            "messages": [
                {"role": "user", "content": append_choices(prompt, lettered)},
                {"role": "assistant", "content": letter},
            ]
        }

    def wrong_attempt(attempt, split: str) -> dict | None:
        """An attempt the student got wrong, as input to the tutor-turn step.

        ``split`` travels with it because the tutor turns written from these go
        two ways: the held-out ones are what guidance responsiveness is scored
        on, and the rest are Stage-2 training records. Nothing downstream can
        tell them apart once the attempt is gone.
        """
        if attempt.correct:
            return None
        history = attempts[attempt.student_id]
        index = order[attempt.student_id][attempt.interaction_id]
        if index < CONTEXT_ATTEMPTS:
            return None
        listed = skills.get(attempt.problem_id) or ["Other"]
        earlier = [
            (bodies.get(a.problem_id, "")[:120], a.answer_text, keys.get(a.problem_id, ""))
            for a in history[:index]
            if not a.correct and (skills.get(a.problem_id) or ["Other"])[0] == listed[0]
        ]
        built = record(attempt)
        if built is None:
            return None
        return {
            "interaction_id": attempt.interaction_id,
            "student": names[attempt.student_id],
            "split": split,
            "prompt": built["messages"][0]["content"],
            "problem": bodies.get(attempt.problem_id, ""),
            "skill": listed[0],
            "student_answer": attempt.answer_text,
            "correct_answer": keys.get(attempt.problem_id, ""),
            "past_errors": earlier[-5:],
        }

    print("Writing:")
    _write([r for a in pooled_train if (r := record(a))], args.out / "stage1_pooled.jsonl")
    _write(
        [r for a in specialize_train if (r := wrong_attempt(a, "stage2"))]
        + [r for a in held_out if (r := wrong_attempt(a, "heldout"))],
        args.out / "wrong_attempts.jsonl",
    )
    for student in specialize:
        name = names[student]
        _write(
            [r for a in specialize_train if a.student_id == student and (r := record(a))],
            args.out / "stage2" / f"{name}.jsonl",
        )
        _write(
            [r for a in validation if a.student_id == student and (r := record(a))],
            args.out / "val" / f"{name}.jsonl",
        )
        _write(
            [r for a in held_out if a.student_id == student and (r := record(a))],
            args.out / "test_st" / f"{name}.jsonl",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
