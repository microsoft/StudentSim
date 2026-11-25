"""``studentsim-baselines``: score a prompted or state-tracking baseline.

A baseline does not have a per-student adapter, so it cannot go through the
evaluation path the trained simulators use, which decodes from a checkpoint.
It generates from a model as it is, and this entry point runs that generation
over the same held-out records and scores what comes back.

Records are read per student. ``--single-turn`` and ``--multi-turn`` each take
a directory of ``<student_id>.jsonl`` files (the ``test_st`` and ``test_mt``
layout) or a single file, which counts as one student. Each student is scored
on their own records, and the reported figure is the mean over students.

The model sees the conversation as role-separated turns:

- chess, closed model: a system message asking for a bare UCI move, then the
  position (without the Maia2 ``Likely next moves`` line, and with the move
  request asking for a bare move), and for responsiveness the student's move
  and the tutor's turn after it. Replies are capped at 10 tokens.
- L2: a system message describing the learner for fidelity (400 tokens), and
  for responsiveness a system message asking for the corrected text only
  (64 tokens). The base model gets no system message for responsiveness.
- math: fidelity sends the multiple-choice prompt alone and reads per-letter
  log-probabilities from a 4-token reply. Responsiveness sends the three turns
  after a system message (none for the base model) and reads up to 128 tokens.

Maia2 sees the position and the two ratings only. Its fidelity counts a
position it cannot read as a miss. Its responsiveness is scored once per
distinct position, against the move most of that position's records give,
and leaves out positions it cannot read.

How each reply is read is set out in :mod:`studentsim.baselines.scoring`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from studentsim.baselines.scoring import (
    chess_move_matches,
    guidance_matcher,
    l2_essay,
    letter_from_logprobs,
    macro_mean,
    option_letters,
)
from studentsim.core.llm import Message
from studentsim.eval.protocol import protocol_for
from studentsim.eval.scoring import record_mode

BASELINES = ("maia2", "azure", "qwen3")
CHESS_MAIA_CANDIDATES_RE = re.compile(r"(?m)^[ \t]*Likely next moves: .*(?:\n|$)")
CHESS_MOVE_REQUEST = "What is your next move? Respond in UCI format."
CHESS_BARE_MOVE_REQUEST = (
    "Respond with ONLY the UCI move (4-5 lowercase characters like 'e2e4', no explanation)."
)
CHESS_SYSTEM_PROMPT = (
    "You are a chess move generator. Output ONLY the UCI move as 4-5 lowercase "
    "characters (e.g., 'e2e4', 'g1f3', 'a7a8q'). No reasoning, no explanation, "
    "no punctuation, no prefix or suffix — just the move."
)
L2_FIDELITY_SYSTEM_PROMPT = (
    "You are simulating an English-learner writing an essay. Write the essay "
    "in the learner's voice — including the same kinds of small mistakes they "
    "typically make. Output only the essay text, no headers."
)
L2_GUIDANCE_SYSTEM_PROMPT = (
    "You are simulating an English-learner. After receiving a tutor's "
    "correction instruction, output ONLY the corrected text — no explanation, "
    "no rewriting the rest of the essay."
)
MATH_GUIDANCE_SYSTEM_PROMPT = "You are simulating a student doing math practice."

FIDELITY_MAX_NEW_TOKENS = {"chess": 10, "l2": 400}
"""Reply caps for fidelity; math fidelity reads letters instead."""
GUIDANCE_MAX_NEW_TOKENS = {"chess": 10, "l2": 64, "math": 128}


def read_records(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def student_files(path: Path, only: Sequence[str] | None = None) -> dict[str, Path]:
    """Each student's record file: every ``*.jsonl`` in a directory, or one file."""
    if path.is_dir():
        files = {p.stem: p for p in sorted(path.glob("*.jsonl"))}
    else:
        files = {path.stem: path}
    if only:
        missing = sorted(set(only) - set(files))
        if missing:
            raise ValueError(f"no records for students {missing} under {path}")
        files = {sid: files[sid] for sid in only}
    return files


def chess_baseline_prompt(prompt: str) -> str:
    """The chess prompt as a model without a per-student adapter receives it.

    Two things separate it from what the trained simulator reads. The
    ``Likely next moves`` line is a Maia2-derived probability list, and Maia2
    stands as its own baseline, so a prompted model is scored on the profile
    the two share rather than on Maia2's move distribution. The move request
    then asks for a bare move.
    """
    prompt = CHESS_MAIA_CANDIDATES_RE.sub("", prompt)
    return prompt.replace(CHESS_MOVE_REQUEST, CHESS_BARE_MOVE_REQUEST)


def _turns(record: dict, needed: int) -> list[dict]:
    messages = record["messages"]
    if len(messages) < needed:
        raise ValueError(f"record has {len(messages)} messages, needs {needed}")
    return messages


def reference(record: dict, *, turn: int) -> str:
    """The recorded answer: the student's first response, or the one after guidance."""
    return str(_turns(record, 2 if turn == 0 else 4)[1 if turn == 0 else 3]["content"])


def baseline_chat(record: dict, *, domain: str, baseline: str, turn: int) -> list[Message]:
    """The turns a baseline receives for ``turn`` (0: first response, 1: after guidance)."""
    messages = _turns(record, 2 if turn == 0 else 4)
    first = str(messages[0]["content"])
    if domain == "chess":
        first = chess_baseline_prompt(first)
    chat = [Message(role="user", content=first)]
    if turn == 1:
        chat += [
            Message(role=messages[1]["role"], content=messages[1]["content"]),
            Message(role=messages[2]["role"], content=messages[2]["content"]),
        ]

    system = None
    if domain == "chess":
        system = CHESS_SYSTEM_PROMPT
    elif domain == "l2" and turn == 0:
        system = L2_FIDELITY_SYSTEM_PROMPT
    elif baseline == "azure" and turn == 1:
        system = L2_GUIDANCE_SYSTEM_PROMPT if domain == "l2" else MATH_GUIDANCE_SYSTEM_PROMPT
    if system is not None:
        chat.insert(0, Message(role="system", content=system))
    return chat


def record_elos(record: dict, default: int) -> tuple[int, int]:
    """The two ratings Maia2 is conditioned on: the record's, else ``default``."""
    elo_self = record.get("elo_self") or default
    elo_oppo = record.get("elo_oppo") or default
    return int(elo_self), int(elo_oppo)


def build_simulator(
    name: str,
    *,
    domain: str,
    model: str | None,
    elo: int,
    device: str,
    workers: int = 8,
    maia2_root: str = "./maia2_models",
):
    """Construct the baseline named on the command line."""
    if name == "maia2":
        from studentsim.baselines.maia2 import Maia2Simulator

        return Maia2Simulator(elo=elo, device=device, save_root=maia2_root)
    if name == "azure":
        from studentsim.baselines.azure_openai import AzureOpenAIClient
        from studentsim.baselines.llm_simulator import LLMClientSimulator

        if not model:
            raise ValueError("--model names the deployment to call")
        # The GPT-5 family caps top_logprobs at 5; anything else takes 20.
        top_k = 5 if "gpt-5" in model.lower() else 20
        return LLMClientSimulator(
            client=AzureOpenAIClient(deployment=model),
            domain=domain,
            top_logprobs_k=top_k,
            workers=workers,
        )
    if name == "qwen3":
        from studentsim.baselines.base_qwen3 import build_base_qwen3_simulator

        if not model:
            raise ValueError("--model names the base checkpoint")
        return build_base_qwen3_simulator(base_model=model, domain=domain, device=device)
    raise ValueError(f"unknown baseline {name!r}; expected one of {list(BASELINES)}")


def _maia2_fidelity(simulator, records: Sequence[dict], elo: int) -> dict:
    rows = []
    for record in records:
        fen = _fen(record)
        gold = reference(record, turn=0)
        try:
            move = simulator.top_move(fen, *record_elos(record, elo))
        except Exception as error:  # noqa: BLE001 - a position Maia2 cannot read is a miss
            rows.append({"prediction": None, "reference": gold, "correct": False,
                         "error": str(error)})
            continue
        rows.append({"prediction": move, "reference": gold, "correct": move == gold})
    hits = sum(row["correct"] for row in rows)
    return {"accuracy": hits / len(rows) if rows else 0.0, "n_samples": len(rows), "rows": rows}


def _maia2_responsiveness(simulator, records: Sequence[dict], elo: int) -> dict:
    """Maia2 reads no guidance, so each distinct position is scored once."""
    positions: dict[str, list[dict]] = {}
    for record in records:
        positions.setdefault(_fen(record), []).append(record)
    rows, skipped = [], 0
    for fen, group in positions.items():
        gold = group[0].get("best_move_uci") or Counter(
            reference(r, turn=1) for r in group
        ).most_common(1)[0][0]
        try:
            move = simulator.top_move(fen, *record_elos(group[0], elo))
        except Exception:  # noqa: BLE001 - a position Maia2 cannot read is left out
            skipped += 1
            continue
        rows.append({"fen": fen, "prediction": move, "reference": gold, "correct": move == gold})
    hits = sum(row["correct"] for row in rows)
    return {
        "accuracy": hits / len(rows) if rows else 0.0,
        "n_samples": len(rows),
        "n_skipped": skipped,
        "rows": rows,
    }


def _fen(record: dict) -> str:
    from studentsim.baselines.maia2 import extract_fen

    fen = extract_fen(str(record["messages"][0]["content"]))
    if fen is None:
        raise ValueError("record prompt carries no FEN")
    return fen


def score_fidelity(
    simulator,
    records: Sequence[dict],
    *,
    domain: str,
    baseline: str,
    counter=None,
    elo: int = 1500,
) -> dict:
    """How closely the baseline imitates one student's first turns."""
    if baseline == "maia2":
        return _maia2_fidelity(simulator, records, elo)
    chats = [baseline_chat(r, domain=domain, baseline=baseline, turn=0) for r in records]
    references = [reference(r, turn=0) for r in records]

    if domain == "math":
        letters = [option_letters(r) for r in records]
        logprobs = simulator.letter_logprobs_batch(chats, letters)
        rows = []
        for row, options, gold in zip(logprobs, letters, references):
            predicted = letter_from_logprobs(row, options)
            rows.append({"letter_logprobs": row, "prediction": predicted,
                         "reference": gold, "correct": predicted == gold.strip()})
        hits = sum(row["correct"] for row in rows)
        return {"accuracy": hits / len(rows) if rows else 0.0, "n_samples": len(rows),
                "rows": rows}

    predictions = simulator.generate_chats(
        chats, max_new_tokens=FIDELITY_MAX_NEW_TOKENS[domain]
    )
    if domain == "l2":
        if counter is None:
            raise ValueError("L2 fidelity needs a LanguageTool counter")
        from studentsim.domains.l2.fidelity import per_record_density_match

        scores = [
            per_record_density_match(
                predicted=counter.count(l2_essay(prediction)),
                reference=counter.count(gold),
            )
            for prediction, gold in zip(predictions, references)
        ]
        rows = [{"prediction": p, "reference": g, "score": s}
                for p, g, s in zip(predictions, references, scores)]
        return {"accuracy": sum(scores) / len(scores) if scores else 0.0,
                "n_samples": len(scores), "rows": rows}

    rows = [{"prediction": p, "reference": g, "correct": chess_move_matches(p, g)}
            for p, g in zip(predictions, references)]
    hits = sum(row["correct"] for row in rows)
    return {"accuracy": hits / len(rows) if rows else 0.0, "n_samples": len(rows), "rows": rows}


def score_responsiveness(
    simulator,
    records: Sequence[dict],
    *,
    domain: str,
    baseline: str,
    elo: int = 1500,
) -> dict:
    """The share of one student's second turns the baseline gets right after guidance.

    Per-mode counts come out alongside the total, because the held-out set
    holds an equal number of records per guidance mode and the breakdown is
    what makes modes comparable.
    """
    if baseline == "maia2":
        return _maia2_responsiveness(simulator, records, elo)
    mode_field = protocol_for(domain).mode_field
    chats = [baseline_chat(r, domain=domain, baseline=baseline, turn=1) for r in records]
    references = [reference(r, turn=1) for r in records]
    predictions = simulator.generate_chats(
        chats,
        max_new_tokens=GUIDANCE_MAX_NEW_TOKENS[domain],
        catch_errors=domain == "math",
    )
    matcher = guidance_matcher(domain, baseline)
    rows = []
    per_mode: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record, prediction, gold in zip(records, predictions, references):
        correct = matcher(prediction, gold)
        mode = record_mode(record, mode_field)
        rows.append({"mode": mode, "prediction": prediction, "reference": gold,
                     "correct": correct})
        if mode:
            per_mode[mode][0] += int(correct)
            per_mode[mode][1] += 1
    hits = sum(row["correct"] for row in rows)
    return {
        "accuracy": hits / len(rows) if rows else 0.0,
        "n_samples": len(rows),
        "per_mode_counts": {m: per_mode[m] for m in sorted(per_mode)},
        "rows": rows,
    }


def score_students(
    files: dict[str, Path],
    score: Callable[[list[dict]], dict],
    *,
    raw_dir: Path | None = None,
) -> dict:
    """Score each student on their own records and average over students."""
    per_student: dict[str, dict] = {}
    mode_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for student, path in files.items():
        result = score(read_records(path))
        rows = result.pop("rows")
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            with open(raw_dir / f"{student}.jsonl", "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for mode, (hits, total) in result.get("per_mode_counts", {}).items():
            mode_totals[mode][0] += hits
            mode_totals[mode][1] += total
        per_student[student] = result
        print(f"  {student}: {result['accuracy']:.4f}  (n={result['n_samples']})", flush=True)
    summary: dict = {
        "mean": macro_mean({s: r["accuracy"] for s, r in per_student.items()}),
        "n_students": len(per_student),
        "per_student": per_student,
    }
    if mode_totals:
        summary["per_mode"] = {m: h / t for m, (h, t) in sorted(mode_totals.items()) if t}
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studentsim-baselines",
        description="Score a baseline on the held-out sets the simulators are scored on.",
    )
    parser.add_argument("--domain", required=True, choices=("chess", "l2", "math"))
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--model", default=None, help="Deployment name or checkpoint path.")
    parser.add_argument("--single-turn", type=Path, default=None,
                        help="Fidelity records: a directory of <student_id>.jsonl, or one file.")
    parser.add_argument("--multi-turn", type=Path, default=None,
                        help="Guidance records: a directory of <student_id>.jsonl, or one file.")
    parser.add_argument("--student-id", action="append", default=None,
                        help="Score only this student; repeat for more.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Keep per-record predictions here.")
    parser.add_argument("--elo", type=int, default=1500,
                        help="Maia2 only: rating for both players when a record has none.")
    parser.add_argument("--maia2-root", default="./maia2_models",
                        help="Maia2 only: where its checkpoint is kept.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Closed models only: requests in flight at once.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.single_turn is None and args.multi_turn is None:
        parser.error("nothing to score; pass --single-turn, --multi-turn, or both")
    if args.baseline == "maia2" and args.domain != "chess":
        parser.error("maia2 plays chess and nothing else")
    if args.baseline == "qwen3" and args.domain == "chess":
        parser.error("the base model is scored on l2 and math only")

    simulator = build_simulator(
        args.baseline, domain=args.domain, model=args.model, elo=args.elo,
        device=args.device, workers=args.workers, maia2_root=args.maia2_root,
    )

    report: dict = {"domain": args.domain, "baseline": args.baseline, "model": args.model}
    if args.single_turn is not None:
        counter = None
        if args.domain == "l2" and args.baseline != "maia2":
            from studentsim.domains.l2.languagetool import LanguageToolCounter

            counter = LanguageToolCounter()
        print("fidelity", flush=True)
        report["fidelity"] = score_students(
            student_files(args.single_turn, args.student_id),
            lambda records: score_fidelity(
                simulator, records, domain=args.domain, baseline=args.baseline,
                counter=counter, elo=args.elo,
            ),
            raw_dir=args.raw_dir / "fidelity" if args.raw_dir else None,
        )
    if args.multi_turn is not None:
        print("responsiveness", flush=True)
        report["responsiveness"] = score_students(
            student_files(args.multi_turn, args.student_id),
            lambda records: score_responsiveness(
                simulator, records, domain=args.domain, baseline=args.baseline,
                elo=args.elo,
            ),
            raw_dir=args.raw_dir / "responsiveness" if args.raw_dir else None,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for metric in ("fidelity", "responsiveness"):
        if metric in report:
            print(f"{metric:<15} mean={report[metric]['mean']:.4f}  "
                  f"n_students={report[metric]['n_students']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
