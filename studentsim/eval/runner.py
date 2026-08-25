"""Running behavioral fidelity and guidance responsiveness for one student.

Fidelity decodes the student's held-out single-turn records and scores the
prediction against what the student actually did. Guidance responsiveness
decodes the multi-turn records, whose tutor turn carries the domain's turn-2
suffix, and scores the updated answer the same way.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from studentsim.eval.checkpoints import find_last_checkpoint
from studentsim.eval.fidelity import (
    LetterLogprobsFactory,
    score_error_density,
    score_letter_choice,
    score_move_match,
)
from studentsim.eval.infer import RunnerCallable, _default_runner, build_infer_command, run_infer
from studentsim.eval.normalize import normalizer_for
from studentsim.eval.protocol import EvalProtocol, protocol_for
from studentsim.eval.scoring import Score, score_results

TURN2_INDEX = 2


def _read_records(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def apply_turn2_suffix(source: str | Path, target: str | Path, suffix: str) -> None:
    """Copy multi-turn records, appending ``suffix`` to the tutor turn.

    A multi-turn record is ``[student turn 1, simulator turn 1, tutor turn]``,
    so the tutor turn is message index 2.
    """
    with open(source, encoding="utf-8") as fin, open(target, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if len(messages) > TURN2_INDEX and messages[TURN2_INDEX].get("role") == "user":
                turn = dict(messages[TURN2_INDEX])
                turn["content"] = f"{turn['content']}\n\n{suffix}"
                messages[TURN2_INDEX] = turn
            fout.write(json.dumps(record) + "\n")


@dataclass(frozen=True)
class StudentResult:
    """One student's fidelity and responsiveness scores."""

    student_id: str
    domain: str
    fidelity: Score | None = None
    responsiveness: Score | None = None


def _decode_and_score(
    *,
    dataset: Path,
    adapter_dir: Path,
    base_model: str,
    protocol: EvalProtocol,
    repetition_penalty: float,
    with_modes: bool,
    score: Callable[[Path], Score] | None = None,
    raw_dir: Path | None,
    raw_name: str,
    runner: RunnerCallable,
    swift_binary: str | None,
) -> Score:
    with tempfile.TemporaryDirectory(prefix="studentsim_infer_") as tmp:
        result_path = Path(tmp) / "result.jsonl"
        command = build_infer_command(
            base_model=base_model,
            adapter_dir=adapter_dir,
            dataset=dataset,
            result_path=result_path,
            max_new_tokens=protocol.max_new_tokens,
            max_batch_size=protocol.max_batch_size,
            repetition_penalty=repetition_penalty,
            temperature=protocol.temperature,
            swift_binary=swift_binary,
        )
        run_infer(command, runner=runner)
        if not result_path.is_file():
            raise RuntimeError(f"swift infer wrote no results for {dataset}")
        if score is not None:
            outcome = score(result_path)
        else:
            outcome = score_results(
                result_path,
                normalize=normalizer_for(protocol.domain),
                strip_think=protocol.strip_think,
                input_path=dataset if with_modes else None,
                mode_field=protocol.mode_field,
            )
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / raw_name).write_text(
                result_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
    return outcome


def evaluate_student(
    *,
    student_id: str,
    domain: str,
    base_model: str,
    adapter_root: str | Path,
    single_turn: str | Path | None = None,
    multi_turn: str | Path | None = None,
    raw_dir: str | Path | None = None,
    issue_counter: object | None = None,
    letter_logprobs_factory: LetterLogprobsFactory | None = None,
    runner: RunnerCallable = _default_runner,
    swift_binary: str | None = None,
) -> StudentResult:
    """Evaluate one student's adapter on whichever sets are supplied."""
    protocol = protocol_for(domain)
    adapter_dir = find_last_checkpoint(adapter_root)
    if adapter_dir is None:
        raise FileNotFoundError(f"no checkpoint found under {adapter_root}")

    raw_path = Path(raw_dir) if raw_dir is not None else None
    fidelity = responsiveness = None

    if single_turn is not None:
        if domain == "l2" and issue_counter is None:
            raise ValueError(
                "L2 fidelity compares error profiles, so it needs an issue_counter"
            )
        if domain == "math":
            if letter_logprobs_factory is None:
                raise ValueError(
                    "math fidelity ranks the multiple-choice letters, so it needs "
                    "letter_logprobs_factory to score against this student's adapter"
                )
            fidelity = score_letter_choice(
                list(_read_records(single_turn)),
                letter_logprobs=letter_logprobs_factory(base_model, adapter_dir),
            )
        else:
            fidelity = _decode_and_score(
                dataset=Path(single_turn),
                adapter_dir=adapter_dir,
                base_model=base_model,
                protocol=protocol,
                repetition_penalty=protocol.fidelity_repetition_penalty,
                with_modes=False,
                score=lambda path: (
                    score_error_density(path, counter=issue_counter)
                    if domain == "l2"
                    else score_move_match(path, strip_think=protocol.strip_think)
                ),
                raw_dir=raw_path,
                raw_name=f"fidelity_raw_{student_id}.jsonl",
                runner=runner,
                swift_binary=swift_binary,
            )

    if multi_turn is not None:
        with tempfile.TemporaryDirectory(prefix="studentsim_turn2_") as tmp:
            prepared = Path(tmp) / "multi_turn.jsonl"
            apply_turn2_suffix(multi_turn, prepared, protocol.turn2_suffix)
            responsiveness = _decode_and_score(
                dataset=prepared,
                adapter_dir=adapter_dir,
                base_model=base_model,
                protocol=protocol,
                repetition_penalty=protocol.guidance_repetition_penalty,
                with_modes=True,
                score=None,
                raw_dir=raw_path,
                raw_name=f"responsiveness_raw_{student_id}.jsonl",
                runner=runner,
                swift_binary=swift_binary,
            )

    return StudentResult(
        student_id=student_id,
        domain=domain,
        fidelity=fidelity,
        responsiveness=responsiveness,
    )


def evaluate_students(
    *,
    student_ids: Sequence[str],
    domain: str,
    base_model: str,
    adapter_root: str | Path,
    single_turn_dir: str | Path | None = None,
    multi_turn_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    issue_counter: object | None = None,
    letter_logprobs_factory: LetterLogprobsFactory | None = None,
    runner: RunnerCallable = _default_runner,
    swift_binary: str | None = None,
) -> list[StudentResult]:
    """Evaluate every student in ``student_ids``, one adapter each."""
    results: list[StudentResult] = []
    for student_id in student_ids:
        single = Path(single_turn_dir) / f"{student_id}.jsonl" if single_turn_dir else None
        multi = Path(multi_turn_dir) / f"{student_id}.jsonl" if multi_turn_dir else None
        results.append(
            evaluate_student(
                student_id=student_id,
                domain=domain,
                base_model=base_model,
                adapter_root=Path(adapter_root) / student_id,
                single_turn=single if single and single.is_file() else None,
                multi_turn=multi if multi and multi.is_file() else None,
                raw_dir=raw_dir,
                issue_counter=issue_counter,
                letter_logprobs_factory=letter_logprobs_factory,
                runner=runner,
                swift_binary=swift_binary,
            )
        )
    return results
