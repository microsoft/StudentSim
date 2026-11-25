"""Behavioral fidelity, which is scored differently in each response space.

Chess compares the predicted move to the recorded one. L2 compares the error
profile of the generated essay to that of the recorded essay, because two
essays by the same learner never match at the surface. Math casts the record
into a multiple-choice form and takes the letter the simulator ranks first,
which needs per-letter likelihoods rather than a decode.

Math guidance responsiveness is decoded here too, by :func:`hf_chat_decoder`,
because it runs on the same HuggingFace backend as the letter ranking.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from studentsim.domains.l2.fidelity import IssueCounter, per_record_density_match
from studentsim.domains.math.fidelity import argmax_letter
from studentsim.eval.normalize import normalize_move
from studentsim.eval.scoring import Score, require_result_fields, score_results


def _rows(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def score_move_match(result_path: str | Path, *, strip_think: bool) -> Score:
    """Chess fidelity: the fraction of positions where the move matches."""
    return score_results(result_path, normalize=normalize_move, strip_think=strip_think)


def score_error_density(result_path: str | Path, *, counter: IssueCounter) -> Score:
    """L2 fidelity: mean per-record agreement of the two error profiles."""
    rows = list(_rows(result_path))
    if not rows:
        return Score(accuracy=0.0, n_samples=0)
    require_result_fields(rows, result_path)
    scores = [
        per_record_density_match(
            predicted=counter.count(str(row["response"])),
            reference=counter.count(str(row["labels"])),
        )
        for row in rows
    ]
    return Score(accuracy=sum(scores) / len(scores), n_samples=len(scores))


def score_letter_choice(
    records: Sequence[Mapping],
    *,
    letter_logprobs: "LetterLogprobs",
    decimals: int | None = None,
) -> Score:
    """Math fidelity: the fraction of records whose top-ranked letter is right.

    Each record is a two-message exchange whose assistant turn is the letter
    the student's answer occupies in that record's multiple-choice form, and
    whose ``candidates`` maps each offered letter to its option. Only the
    offered letters are ranked, in alphabetical order, the first of a tie
    winning. A record of any other shape raises. ``decimals`` rounds the
    student's score.
    """
    hits = 0
    total = 0
    for index, record in enumerate(records):
        messages = record.get("messages", [])
        candidates = record.get("candidates")
        if len(messages) != 2 or not candidates:
            raise ValueError(
                f"fidelity record {index} is not a [user, letter] exchange with "
                f"candidates; keys present: {sorted(record)}"
            )
        letters = sorted(candidates)
        target = str(messages[-1].get("content", "")).strip()
        if target not in letters:
            raise ValueError(
                f"fidelity record {index} answers {target!r}, which is not one of "
                f"its candidate letters {letters}"
            )
        logprobs = letter_logprobs(messages[:1], letters)
        total += 1
        hits += int(argmax_letter(logprobs, candidate_letters=letters) == target)
    if total == 0:
        return Score(accuracy=0.0, n_samples=0)
    accuracy = hits / total
    if decimals is not None:
        accuracy = round(accuracy, decimals)
    return Score(accuracy=accuracy, n_samples=total)


LetterLogprobsFactory = Callable[[str, "Path | str"], "LetterLogprobs"]
"""Builds a scorer bound to one student's adapter."""


class LetterLogprobs:
    """Callable returning the log-probability of each candidate letter."""

    def __call__(
        self, prompt_messages: Sequence[Mapping], letters: Sequence[str]
    ) -> Mapping[str, float]:  # pragma: no cover - protocol
        raise NotImplementedError


def hf_letter_logprobs(base_model: str, adapter_dir: str | Path) -> LetterLogprobs:
    """Score letters by teacher-forcing them against a locally loaded adapter."""
    from studentsim.core.simulator import SimulatorSpec
    from studentsim.inference.hf_simulator import HFSimulator

    simulator = HFSimulator(
        SimulatorSpec(
            base_model=base_model,
            lora_adapter_path=str(adapter_dir),
            domain="math",
        )
    )

    def score(prompt_messages: Sequence[Mapping], letters: Sequence[str]):
        (turn,) = prompt_messages
        return simulator.logprobs(str(turn["content"]), candidates=list(letters))

    return score


ChatDecoder = Callable[[Sequence[Mapping]], str]
"""Decodes the next assistant turn of a chat."""

ChatDecoderFactory = Callable[[str, "Path | str", int], ChatDecoder]
"""Builds a decoder bound to one student's adapter and a token budget."""


def hf_chat_decoder(base_model: str, adapter_dir: str | Path, max_new_tokens: int) -> ChatDecoder:
    """Decode one record at a time with HuggingFace against a merged adapter."""
    from studentsim.core.simulator import SimulatorSpec
    from studentsim.inference.hf_simulator import HFSimulator

    simulator = HFSimulator(
        SimulatorSpec(
            base_model=base_model,
            lora_adapter_path=str(adapter_dir),
            domain="math",
        )
    )

    def decode(messages: Sequence[Mapping]) -> str:
        return simulator.continue_chat(messages, max_new_tokens=max_new_tokens)

    return decode
