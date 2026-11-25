"""Build the math training and evaluation sets from a FoundationalASSIST extract.

The students, their split, and every record are derived from the extract
itself, so two builds of the same extract give identical record sets.
A student's hashed identifier seeds their draws and is then replaced by a
positional name, so nothing in the output identifies anyone.

    python -m studentsim.data.math.build --exclude data/math/audit_excluded.json --seed 65

Every record is multiple choice: the prompt lists four options, drawn from what
other students wrote on the same problem, and the answer is a letter. A problem
with too few distinct answers keeps its free-text form.

What this step writes:

``test_st/<student>.jsonl``
    Behavioral fidelity: the held-out draw followed by the validation tail,
    one two-message record each, with the lettered options under ``candidates``.
``stage1_st.jsonl`` and ``stage2_st/<student>.jsonl``
    The single-turn part of each training stage.
``wrong_attempts.jsonl``
    One row per guidance record still to be written: every wrong attempt in the
    held-out and validation splits in each tutor style, which guidance
    responsiveness is scored on, and the wrong attempts the two training stages
    draw. :mod:`studentsim.data.math.generate` writes their tutor turns and
    completes ``stage1_pooled.jsonl``, ``stage2/`` and ``test_mt/``.

``--seed`` decides which training records each stage draws and how their options
are lettered, so each pair in :data:`~studentsim.core.seeds.CROSS_SEED_RUNS` has
its own build, given its own ``--out``. The evaluation sets do not depend on it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

from studentsim.core.seeds import DATA_SAMPLER_SEED
from studentsim.data.math.choices import (
    append_choices,
    build_choices,
    choice_turns,
    collect_answers,
    keyed_seed,
)
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

#: The tutor styles, in the order every per-style loop runs.
MODES = ("error_remediation", "socratic", "conceptual")

#: Seeds the per-student order of the multi-turn pools and the held-out draw.
SPLIT_SEED = 42

#: The held-out draw of each specialization student, by roster position: their
#: held-out split is shuffled with ``SPLIT_SEED`` plus this offset and the first
#: ``cap`` attempts are kept.
HELD_OUT_DRAW_OFFSETS = (
    4606, 584, 7933, 5929, 7732, 5702, 6929, 961, 258, 9412, 7872, 1065, 553, 1729, 3111,
)

#: Seeds the options of the evaluation records, which every training seed shares.
EVAL_SEED = DATA_SAMPLER_SEED

#: Share of a training stage that carries a tutor turn.
MULTITURN_RATIO = 0.2

#: Records per specialization student, before the guided share is capped by what
#: the student's wrong attempts allow.
STAGE2_TOTAL = 154

#: Weight of each tutor style when a stage draws its guided records.
MODE_WEIGHTS = {mode: 1.0 for mode in MODES}


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


def shuffle_per_student(attempts: list, seed: int) -> dict[str, list]:
    """Each student's attempts in a seeded order, students taken by identifier.

    One generator serves every student in turn, so a student's order depends
    on everyone sorted before them.
    """
    by_student: dict[str, list] = collections.defaultdict(list)
    for attempt in attempts:
        by_student[attempt.student_id].append(attempt)
    rng = random.Random(seed)
    ordered: dict[str, list] = {}
    for student in sorted(by_student):
        group = list(by_student[student])
        rng.shuffle(group)
        ordered[student] = group
    return ordered


def weighted_draw(candidates: list[tuple], target: int, rng: random.Random) -> list[tuple]:
    """Up to ``target`` ``(attempt, mode)`` pairs, spread over the tutor styles.

    Styles are taken in the order they first appear in ``candidates``.
    """
    if not candidates or target <= 0:
        return []
    by_mode: dict[str, list[tuple]] = collections.defaultdict(list)
    for pair in candidates:
        by_mode[pair[1]].append(pair)
    total_weight = sum(MODE_WEIGHTS.get(mode, 1.0) for mode in by_mode) or 1.0
    selected: list[tuple] = []
    for mode, pool in by_mode.items():
        pool = list(pool)
        rng.shuffle(pool)
        selected.extend(pool[: round(target * MODE_WEIGHTS.get(mode, 1.0) / total_weight)])
    rng.shuffle(selected)
    return selected[:target]


def balanced_draw(wrong_by_student: dict[str, list], target: int, seed: int) -> list[tuple]:
    """``(attempt, mode)`` pairs spread evenly over students and tutor styles.

    Each wrong attempt is given one style first, then every student contributes
    the same number per style, so a student with many wrong attempts does not
    fill the pooled stage on their own.
    """
    n_students = len(wrong_by_student) or 1
    total_weight = sum(MODE_WEIGHTS.values()) or 1.0
    per_cell = {
        mode: max(1, round(target * MODE_WEIGHTS[mode] / total_weight / n_students))
        for mode in sorted(MODES)
    }
    selected: list[tuple] = []
    for student in sorted(wrong_by_student):
        assign = random.Random(keyed_seed(seed, f"assign:{student}"))
        by_mode: dict[str, list[tuple]] = collections.defaultdict(list)
        for attempt in sorted(wrong_by_student[student], key=lambda a: a.interaction_id):
            offered = sorted(MODES)
            weights = [MODE_WEIGHTS[mode] for mode in offered]
            mode = assign.choices(offered, weights=weights, k=1)[0]
            by_mode[mode].append((attempt, mode))
        for mode in sorted(MODES):
            pool = list(by_mode.get(mode, []))
            random.Random(keyed_seed(seed, f"sample:{student}:{mode}")).shuffle(pool)
            selected.extend(pool[: per_cell[mode]])
    random.Random(seed).shuffle(selected)
    return selected


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
    parser.add_argument(
        "--seed",
        type=int,
        default=DATA_SAMPLER_SEED,
        help="Data-sampler seed of the run the training sets are for.",
    )
    args = parser.parse_args(argv)
    seed = args.seed

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

    # The key as the platform recorded it is what the prompt shows. The repaired
    # key decides whether an attempt was wrong and is the answer a guidance
    # record leads to.
    keys = answer_keys(problems_csv)
    fixed = answer_keys(problems_csv, repair=True)
    # A problem whose key was repaired is trusted, since the repair is exact, so
    # the audit's verdict on its corrupted form does not apply.
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
    cap_test = min(len(splits[student]["test"]) for student in specialize)
    print(
        f"  per student: {cap_pooled} pooled, {cap_specialize} specialization, "
        f"{cap_val} validation, {cap_test} held out"
    )

    names = {student: f"student_{i:02d}" for i, student in enumerate(pooled)}
    order = {
        student: {a.interaction_id: i for i, a in enumerate(attempts[student])}
        for student in pooled
    }
    written_by_others = collect_answers(interactions, usable, set(pooled))

    def wrong(attempt) -> bool:
        return attempt.answer_text.lower() != fixed.get(attempt.problem_id, "").lower()

    def prompt_for(attempt) -> str:
        history = attempts[attempt.student_id]
        index = order[attempt.student_id][attempt.interaction_id]
        return build_user_message(
            student_name=names[attempt.student_id],
            profile=compute_profile(history[:index], skills),
            current=attempt,
            earlier=history[:index],
            skills=skills,
            bodies=bodies,
            keys=keys,
            answer_types=answer_types,
        )

    def single_turn(attempt, letter_seed: int, *, keep_free_text: bool) -> dict | None:
        """One attempt as a multiple-choice record.

        Without enough options it stays free text when ``keep_free_text``, and
        is left out otherwise.
        """
        prompt = prompt_for(attempt)
        choices = build_choices(
            student_answer=attempt.answer_text,
            problem_id=attempt.problem_id,
            written_by_others=written_by_others,
            seed=letter_seed,
        )
        if choices is None:
            if not keep_free_text:
                return None
            return {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": attempt.answer_text},
                ]
            }
        lettered, letter = choices
        return {
            "messages": [
                {"role": "user", "content": append_choices(prompt, lettered)},
                {"role": "assistant", "content": letter},
            ],
            "candidates": lettered,
        }

    def training_single(attempt, stage: str) -> dict:
        key = f"mc:{stage}st:letter:{attempt.student_id}:{attempt.interaction_id}"
        record = single_turn(attempt, keyed_seed(seed, key), keep_free_text=True)
        return {"messages": record["messages"]}

    def guidance_row(attempt, mode: str, target: str, choices: dict | None) -> dict:
        """A wrong attempt in one tutor style, as input to the tutor-turn step."""
        history = attempts[attempt.student_id]
        index = order[attempt.student_id][attempt.interaction_id]
        listed = skills.get(attempt.problem_id) or ["Other"]
        earlier = [
            (bodies.get(a.problem_id, "")[:120], a.answer_text, keys.get(a.problem_id, ""))
            for a in history[:index]
            if not a.correct and (skills.get(a.problem_id) or ["Other"])[0] == listed[0]
        ]
        return {
            "interaction_id": attempt.interaction_id,
            "student": names[attempt.student_id],
            "target": target,
            "style": mode,
            "prompt": prompt_for(attempt),
            "problem": bodies.get(attempt.problem_id, ""),
            "skill": listed[0],
            "student_answer": attempt.answer_text,
            "correct_answer": fixed.get(attempt.problem_id, ""),
            "past_errors": earlier[-5:],
            "choices": choices,
        }

    def training_choices(attempt, stage: str) -> dict | None:
        ids = f"{attempt.student_id}:{attempt.interaction_id}"
        return choice_turns(
            student_answer=attempt.answer_text,
            correct_answer=fixed.get(attempt.problem_id, ""),
            problem_id=attempt.problem_id,
            written_by_others=written_by_others,
            seed_first=keyed_seed(seed, f"mc:{stage}mt:letter01:{ids}"),
            seed_second=keyed_seed(seed, f"mc:{stage}mt:letter23:{ids}"),
        )

    def eval_choices(attempt, mode: str) -> dict | None:
        student, interaction = attempt.student_id, attempt.interaction_id
        return choice_turns(
            student_answer=attempt.answer_text,
            correct_answer=fixed.get(attempt.problem_id, ""),
            problem_id=attempt.problem_id,
            written_by_others=written_by_others,
            seed_first=keyed_seed(EVAL_SEED, f"{student}:{interaction}:{mode}:letter"),
            seed_second=keyed_seed(EVAL_SEED, f"{student}:{interaction}:msg23:{mode}:letter"),
        )

    rows: list[dict] = []
    print("Writing:")

    # Pooled stage: every single-turn record, less as many as the guided records
    # take, so the stage keeps its size.
    pooled_wrong = collections.defaultdict(list)
    for attempt in pooled_train:
        if wrong(attempt):
            pooled_wrong[attempt.student_id].append(attempt)
    guided = balanced_draw(
        pooled_wrong, round(len(pooled_train) * MULTITURN_RATIO), seed
    )
    singles = [training_single(a, "p1") for a in pooled_train]
    keep = max(0, len(pooled_train) - len(guided))
    if keep < len(singles):
        random.Random(seed).shuffle(singles)
        singles = singles[:keep]
    _write(singles, args.out / "stage1_st.jsonl")
    rows += [guidance_row(a, mode, "stage1", training_choices(a, "p1")) for a, mode in guided]

    # Specialization stage, per student: the guided share is drawn first, then
    # the single-turn records fill the rest.
    specialize_wrong = shuffle_per_student(
        [a for a in specialize_train if wrong(a)], SPLIT_SEED
    )
    for student in specialize:
        rng = random.Random(keyed_seed(seed, student))
        own = [a for a in specialize_train if a.student_id == student]
        candidates = [(a, mode) for mode in MODES for a in specialize_wrong.get(student, [])]
        total = min(STAGE2_TOTAL, len(own) + len(candidates))
        n_guided = min(round(total * MULTITURN_RATIO), len(candidates))
        drawn = weighted_draw(candidates, n_guided, rng)
        rng.shuffle(own)
        _write(
            [training_single(a, "p2") for a in own[: total - n_guided]],
            args.out / "stage2_st" / f"{names[student]}.jsonl",
        )
        rows += [
            guidance_row(a, mode, "stage2", training_choices(a, "p2")) for a, mode in drawn
        ]

    # Evaluation. Fidelity takes the held-out draw and the validation tail;
    # responsiveness takes every wrong attempt in the whole held-out split and
    # in the validation tail, in every tutor style.
    held_out_wrong = shuffle_per_student(
        [a for s in specialize for a in splits[s]["test"] if wrong(a)], SPLIT_SEED
    )
    validation_wrong = shuffle_per_student(
        sorted((a for a in validation if wrong(a)), key=lambda a: a.row), SPLIT_SEED
    )
    for position, student in enumerate(specialize):
        draw = list(splits[student]["test"])
        random.Random(SPLIT_SEED + HELD_OUT_DRAW_OFFSETS[position]).shuffle(draw)
        fidelity = draw[:cap_test] + [a for a in validation if a.student_id == student]
        _write(
            [
                record
                for a in fidelity
                if a.answer_text
                and (
                    record := single_turn(
                        a,
                        keyed_seed(EVAL_SEED, a.interaction_id),
                        keep_free_text=False,
                    )
                )
            ],
            args.out / "test_st" / f"{names[student]}.jsonl",
        )

        rng = random.Random(keyed_seed(EVAL_SEED, student))
        pool = held_out_wrong.get(student, []) + validation_wrong.get(student, [])
        for mode in MODES:
            ordered = list(pool)
            rng.shuffle(ordered)
            rows += [guidance_row(a, mode, "test_mt", eval_choices(a, mode)) for a in ordered]

    _write(rows, args.out / "wrong_attempts.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
