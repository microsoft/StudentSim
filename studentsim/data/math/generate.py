"""Generating the multi-turn math records.

Each record starts from an attempt the student got wrong. The student's
reasoning and the tutor's response are written by a language model; the
problem, the wrong answer, and the correct answer come from the data.

The build step lists every record to write in ``wrong_attempts.jsonl``, each in
one tutor style and with its options already drawn. A record with options is
multiple choice in both turns: the student's first answer is the letter of their
own wrong answer, the tutor turn ends with options built around the correct
answer, and the second answer is the letter of the correct one. A record whose
problem has too few distinct answers stays free text.

This step writes three things into ``--out``:

``test_mt/<student>.jsonl``
    The guidance responsiveness set, with the tutor style under
    ``instruction_type``. Its free-text records keep the student's reasoning
    in both answers.
``stage1_pooled.jsonl`` and ``stage2/<student>.jsonl``
    The training stages: the single-turn records the build wrote, followed by
    the guidance records. Training answers carry no reasoning.

The student's reasoning towards their wrong answer does not depend on how a
tutor later responds, so it is written once per attempt, reused by every style,
and cached on disk so a rerun does not pay for it again. The reasoning behind
the second answer is written only where a record keeps it.

    python -m studentsim.data.math.generate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from studentsim.core.llm import Message, open_client
from studentsim.data.math.choices import append_choices
from studentsim.data.math.guidance import (
    STYLES,
    TURN1_THINKING_SYS,
    TURN1_THINKING_USER,
    format_past_errors,
)

#: Room for a tutor turn or a student's reasoning.
MAX_TOKENS = 600

#: Where the build step writes the single-turn part of each training stage.
STAGE1_SINGLE = "stage1_st.jsonl"
STAGE2_SINGLE = "stage2_st"


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


def _answer(reasoning: str, answer: str) -> str:
    return f"<think>\n{reasoning}\n</think>\n\n{answer}"


def assemble(row: dict, *, turn1: str, tutor: str, reply: str | None) -> list[dict]:
    """The four messages of one record.

    A training record drops the reasoning from both answers. A record with
    options turns both answers into letters and lists the options after the
    prompt and after the tutor turn.
    """
    training = row["target"] != "test_mt"
    choices = row["choices"]
    first = row["student_answer"] if training else _answer(turn1, row["student_answer"])
    second = row["correct_answer"]
    if not training and choices is None:
        second = _answer(reply or "", row["correct_answer"])
    messages = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": first},
        {"role": "user", "content": tutor},
        {"role": "assistant", "content": second},
    ]
    if choices is None:
        return messages
    return [
        {"role": "user", "content": append_choices(row["prompt"], choices["first"])},
        {"role": "assistant", "content": choices["first_letter"]},
        {"role": "user", "content": append_choices(tutor, choices["second"])},
        {"role": "assistant", "content": choices["second_letter"]},
    ]


def write_record(client, row: dict, turn1: str) -> list[dict]:
    """Write the tutor turn for one row, and the second reasoning where it is kept."""
    tutor_system, tutor_user, reply_system, reply_user = STYLES[row["style"]]
    fields = {
        "problem_body": row["problem"],
        "skill": row["skill"],
        "student_answer": row["student_answer"],
        "correct_answer": row["correct_answer"],
        "turn1_thinking": turn1,
        "past_errors_block": format_past_errors(row.get("past_errors", [])),
    }
    tutor = _ask(client, tutor_system, tutor_user.format(**fields))
    reply = None
    if row["target"] == "test_mt" and row["choices"] is None:
        reply = _ask(client, reply_system, reply_user.format(**fields, tutor_instruction=tutor))
    return assemble(row, turn1=turn1, tutor=tutor, reply=reply)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m studentsim.data.math.generate")
    parser.add_argument("--wrong-attempts", type=Path,
                        default=Path("data/math/wrong_attempts.jsonl"),
                        help="JSONL of guidance records to write, as the build step writes it.")
    parser.add_argument("--out", type=Path, default=Path("data/math"),
                        help="The build step's output directory, which this step completes.")
    parser.add_argument("--turn1-cache", type=Path, default=Path("data/math/turn1_cache.jsonl"))
    parser.add_argument("--model", default="gpt-5.4", help="the model that writes the turns")
    args = parser.parse_args(argv)

    client = open_client(args.model)
    cache: dict[str, str] = {}
    if args.turn1_cache.is_file():
        for entry in _read_jsonl(args.turn1_cache):
            cache[entry["interaction_id"]] = entry["turn1_thinking"]
    print(f"Reusing {len(cache):,} first-turn traces", flush=True)

    rows = _read_jsonl(args.wrong_attempts)
    print(f"{len(rows):,} guidance records to write", flush=True)

    stage1: list[dict] = []
    stage2: dict[str, list[dict]] = {}
    held_out: dict[str, list[dict]] = {}
    with open(args.turn1_cache, "a", encoding="utf-8") as cache_file:
        for index, row in enumerate(rows, 1):
            key = row["interaction_id"]
            if key not in cache:
                cache[key] = write_turn1(
                    client,
                    problem=row["problem"],
                    skill=row["skill"],
                    student_answer=row["student_answer"],
                )
                cache_file.write(
                    json.dumps({"interaction_id": key, "turn1_thinking": cache[key]}) + "\n"
                )
                cache_file.flush()
            messages = write_record(client, row, cache[key])
            if row["target"] == "stage1":
                stage1.append({"messages": messages})
            elif row["target"] == "stage2":
                stage2.setdefault(row["student"], []).append({"messages": messages})
            else:
                held_out.setdefault(row["student"], []).append(
                    {"messages": messages, "instruction_type": row["style"]}
                )
            if index % 25 == 0:
                print(f"  {index:,}/{len(rows):,}", flush=True)

    _write_jsonl(_read_jsonl(args.out / STAGE1_SINGLE) + stage1, args.out / "stage1_pooled.jsonl")
    for single in sorted((args.out / STAGE2_SINGLE).glob("*.jsonl")):
        student = single.stem
        _write_jsonl(
            _read_jsonl(single) + stage2.get(student, []), args.out / "stage2" / single.name
        )
    for student, records in sorted(held_out.items()):
        _write_jsonl(records, args.out / "test_mt" / f"{student}.jsonl")
    print(
        f"wrote {len(stage1):,} pooled, {sum(map(len, stage2.values())):,} specialization and "
        f"{sum(map(len, held_out.values())):,} held-out guidance records under {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
