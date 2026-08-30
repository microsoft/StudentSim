"""Generating the multi-turn math records.

Each record starts from an attempt the student got wrong. The student's
reasoning, the tutor's response, and the student's second attempt are written
by a language model; the problem, the wrong answer, and the correct answer come
from the data.

Every style is written in one run by default. Records go to one file per
student, under ``test_mt`` for the held-out attempts and ``stage2_mt`` for the
rest, which is the shape evaluation and Stage-2 training read.

The first turn is the student reasoning towards their own wrong answer, which
does not depend on how a tutor later responds, so it is written once and reused
by the styles that follow, and cached on disk so a rerun does not pay for it
again.

    python -m studentsim.data.math.generate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from studentsim.core.llm import Message, open_client
from studentsim.data.math.guidance import (
    STYLES,
    TURN1_THINKING_SYS,
    TURN1_THINKING_USER,
    format_past_errors,
)

#: Room for a tutor turn or a student's reasoning.
MAX_TOKENS = 600

#: Where each split's records go. ``test_mt`` is where evaluation reads the
#: multi-turn held-out set from, matching the L2 build.
SPLIT_DIRS = {"stage2": "stage2_mt", "heldout": "test_mt"}


def _ask(client, system: str, user: str) -> str:
    reply = client.complete(
        [Message("system", system), Message("user", user)],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    return reply.text.strip()


def write_turn1(client, *, problem: str, skill: str, student_answer: str) -> str:
    """The student's reasoning on the way to their wrong answer."""
    return _ask(
        client,
        TURN1_THINKING_SYS,
        TURN1_THINKING_USER.format(
            problem_body=problem, skill=skill, student_answer=student_answer
        ),
    )


def build_record(
    client,
    *,
    style: str,
    student: str,
    prompt: str,
    problem: str,
    skill: str,
    student_answer: str,
    correct_answer: str,
    turn1: str,
    past_errors: list,
) -> dict:
    """One four-message record in the given tutor style."""
    tutor_system, tutor_user, reply_system, reply_user = STYLES[style]
    fields = {
        "problem_body": problem,
        "skill": skill,
        "student_answer": student_answer,
        "correct_answer": correct_answer,
        "turn1_thinking": turn1,
        "past_errors_block": format_past_errors(past_errors),
    }
    tutor = _ask(client, tutor_system, tutor_user.format(**fields))
    reply = _ask(client, reply_system, reply_user.format(**fields, tutor_instruction=tutor))
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<think>{turn1}</think>{student_answer}"},
            {"role": "user", "content": tutor},
            {"role": "assistant", "content": f"<think>{reply}</think>{correct_answer}"},
        ],
        "style": style,
        "student": student,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m studentsim.data.math.generate")
    parser.add_argument("--style", choices=sorted(STYLES), action="append",
                        help="Repeatable; every style is written when this is not given.")
    parser.add_argument("--wrong-attempts", type=Path,
                        default=Path("data/math/wrong_attempts.jsonl"),
                        help="JSONL of wrong attempts, as the build step writes it.")
    parser.add_argument("--out", type=Path, default=Path("data/math"),
                        help="Directory to write one multiturn_<style>.jsonl into.")
    parser.add_argument("--turn1-cache", type=Path, default=Path("data/math/turn1_cache.jsonl"))
    parser.add_argument("--model", default="gpt-5.4", help="the model that writes the turns")
    args = parser.parse_args(argv)

    styles = args.style or sorted(STYLES)
    client = open_client(args.model)
    cache: dict[str, str] = {}
    if args.turn1_cache.is_file():
        for line in args.turn1_cache.read_text().splitlines():
            entry = json.loads(line)
            cache[entry["interaction_id"]] = entry["turn1_thinking"]
    print(f"Reusing {len(cache):,} first-turn traces", flush=True)

    attempts = [json.loads(line) for line in args.wrong_attempts.read_text().splitlines()]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(attempts):,} attempts x {len(styles)} styles", flush=True)

    # Records go to one file per student, in the directory their split is read
    # from: held-out records are what guidance responsiveness is scored on, and
    # the rest are Stage-2 training records. The styles share the first turn,
    # which is the student reasoning towards their own wrong answer and is
    # settled before any tutor speaks, so writing it per style would pay for
    # the same trace once per style.
    written: dict[tuple[str, str], list[dict]] = {}
    with open(args.turn1_cache, "a", encoding="utf-8") as cache_file:
        for index, attempt in enumerate(attempts, 1):
            key = attempt["interaction_id"]
            if key not in cache:
                cache[key] = write_turn1(
                    client,
                    problem=attempt["problem"],
                    skill=attempt["skill"],
                    student_answer=attempt["student_answer"],
                )
                cache_file.write(
                    json.dumps({"interaction_id": key, "turn1_thinking": cache[key]}) + "\n"
                )
                cache_file.flush()
            for style in styles:
                record = build_record(
                    client,
                    style=style,
                    student=attempt["student"],
                    prompt=attempt["prompt"],
                    problem=attempt["problem"],
                    skill=attempt["skill"],
                    student_answer=attempt["student_answer"],
                    correct_answer=attempt["correct_answer"],
                    turn1=cache[key],
                    past_errors=attempt.get("past_errors", []),
                )
                where = SPLIT_DIRS[attempt.get("split", "heldout")]
                written.setdefault((where, attempt["student"]), []).append(record)
            if index % 25 == 0:
                print(f"  {index:,}/{len(attempts):,}", flush=True)

    for (where, student), records in sorted(written.items()):
        target = args.out / where / f"{student}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    for where in sorted({w for w, _ in written}):
        n = sum(len(v) for (w, _), v in written.items() if w == where)
        students = len({s for w, s in written if w == where})
        print(f"wrote {n:,} records for {students} students under {args.out / where}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
