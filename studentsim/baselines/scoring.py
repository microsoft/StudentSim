"""Scoring a prompted baseline.

Each domain reads a baseline's reply in its own way, and L2 guidance reads it
differently for a closed model than for the base model.

- Chess: reasoning blocks come off, the reply is lowercased, and the first
  coordinate move in it is the prediction. It counts when it equals the
  lowercased recorded move.
- L2 fidelity: reasoning blocks and surrounding whitespace come off the essay,
  which is then compared to the recorded essay by error density
  (:func:`studentsim.domains.l2.fidelity.per_record_density_match`).
- L2 guidance, closed model: reasoning blocks come off, ``(delete)`` reads as
  ``delete``, and the reply is stripped and lowercased.
- L2 guidance, base model: reasoning blocks come off, the reply is lowercased,
  and punctuation, quotes and whitespace are trimmed from both ends.
- Math guidance: reasoning blocks, spaces and ``.,;:!?`` at either end, and
  markup come off, and the reply is lowercased.
- Math fidelity: the letter the model ranks first among the record's own
  options, with an option the model did not rank floored at
  :data:`~studentsim.domains.math.fidelity.LOGPROB_FLOOR`.

Every guidance reading applies the same function to the reply and to the
recorded answer, and an empty reading never counts. Per-student figures are
averaged with equal weight per student (:func:`macro_mean`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Final

from studentsim.domains.math.fidelity import LOGPROB_FLOOR, MATH_LETTERS, argmax_letter

_THINK_TRAILING: Final = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_BLOCK: Final = re.compile(r"<think>.*?</think>", re.DOTALL)
_UCI_MOVE: Final = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")
_DELETE_MARK: Final = re.compile(r"\(delete\)", re.IGNORECASE)
_TRAILING_PUNCT: Final = re.compile(r"[.,!?;:\"'\s]+$")
_LEADING_PUNCT: Final = re.compile(r"^[.,!?;:\"'\s]+")
_MARKUP: Final = re.compile(r"<[^>]+>")
_OPTION_LINE: Final = re.compile(r"(?m)^[ \t]+([A-H])\) ")
_OPTIONS_HEADER: Final = "Multiple-choice options:"

Normalizer = Callable[[str], str]


def chess_move(text: str) -> str:
    """The first coordinate move in a reply, or ``""`` if there is none."""
    clean = _THINK_TRAILING.sub("", text or "").strip().lower()
    match = _UCI_MOVE.search(clean)
    return match.group(1) if match else ""


def chess_move_matches(prediction: str, reference: str) -> bool:
    """Whether the move read from a reply is the recorded move."""
    return chess_move(prediction) == reference.lower()


def l2_essay(text: str) -> str:
    """A generated essay without reasoning blocks or surrounding whitespace."""
    if not text:
        return ""
    return _THINK_TRAILING.sub("", text).strip()


def l2_correction_closed(text: str) -> str:
    """A closed model's L2 correction as it is compared."""
    if text is None:
        return ""
    text = _THINK_TRAILING.sub("", text).strip()
    text = _DELETE_MARK.sub("delete", text)
    return text.strip().lower()


def l2_correction_base(text: str) -> str:
    """The base model's L2 correction as it is compared."""
    if not text:
        return ""
    text = _THINK_TRAILING.sub("", text).strip().lower()
    text = _TRAILING_PUNCT.sub("", text)
    return _LEADING_PUNCT.sub("", text)


def math_answer(text: str) -> str:
    """A math answer as it is compared."""
    if not text:
        return ""
    text = _THINK_BLOCK.sub("", text).strip()
    text = text.strip(" .,;:!?")
    text = _MARKUP.sub("", text).strip()
    return text.lower()


def exact_match(prediction: str, reference: str, normalize: Normalizer) -> bool:
    """Whether two answers agree once read by ``normalize``; empty never agrees."""
    predicted = normalize(prediction)
    return predicted == normalize(reference) and predicted != ""


def guidance_matcher(domain: str, baseline: str) -> Callable[[str, str], bool]:
    """How a guided reply is compared to the recorded one."""
    if domain == "chess":
        return chess_move_matches
    if domain == "l2":
        normalize = l2_correction_base if baseline == "qwen3" else l2_correction_closed
    elif domain == "math":
        normalize = math_answer
    else:
        raise ValueError(f"unknown domain: {domain!r}")
    return lambda prediction, reference: exact_match(prediction, reference, normalize)


def guidance_accuracy(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    matcher: Callable[[str, str], bool],
) -> float:
    """The share of guided answers a baseline gets right."""
    if len(predictions) != len(references):
        raise ValueError(
            f"got {len(predictions)} predictions for {len(references)} references"
        )
    if not predictions:
        return 0.0
    return sum(matcher(p, r) for p, r in zip(predictions, references)) / len(predictions)


def option_letters(record: Mapping) -> tuple[str, ...]:
    """The letters a math record offers, in alphabetical order.

    Read from a ``candidates`` mapping when the record carries one, and
    otherwise from the lettered option lines after the prompt's
    ``Multiple-choice options:`` header.
    """
    candidates = record.get("candidates")
    if isinstance(candidates, Mapping):
        return tuple(sorted(candidates))
    prompt = str(record["messages"][0]["content"])
    options = prompt.rpartition(_OPTIONS_HEADER)[2]
    found = _OPTION_LINE.findall(options)
    return tuple(sorted(dict.fromkeys(found))) or MATH_LETTERS


def letter_from_logprobs(
    logprobs: Mapping[str, float], letters: Sequence[str] = MATH_LETTERS
) -> str:
    """The letter a model ranks first among ``letters``.

    A letter absent from ``logprobs`` sits at the floor, and a tie goes to the
    letter that comes first.
    """
    return argmax_letter(
        {letter: float(logprobs.get(letter, LOGPROB_FLOOR)) for letter in letters},
        candidate_letters=letters,
    )


def closed_letter_logprobs(content: Iterable, letters: Sequence[str]) -> dict[str, float]:
    """Per-letter log-probabilities read from a closed model's generated tokens.

    ``content`` is the SDK's per-token list, each entry carrying ``token`` and
    ``top_logprobs``. The first generated token that reads as one of
    ``letters`` once stripped and uppercased is the answer position; each of
    its alternatives that reads as a letter sets that letter to the highest
    log-probability seen for it. Letters never seen stay at the floor.
    """
    scores = {letter: LOGPROB_FLOOR for letter in letters}
    wanted = set(letters)
    for token in content:
        if token.token.strip().upper() in wanted:
            for alternative in token.top_logprobs:
                letter = alternative.token.strip().upper()
                if letter in wanted:
                    scores[letter] = max(scores[letter], alternative.logprob)
            break
    return scores


def math_fidelity_accuracy(
    per_record_logprobs: Sequence[Mapping[str, float]],
    correct_letters: Sequence[str],
    letters: Sequence[Sequence[str]] | None = None,
) -> float:
    """The share of problems where the baseline ranks the student's answer first.

    ``letters`` gives each record's own options; without it every record
    offers :data:`~studentsim.domains.math.fidelity.MATH_LETTERS`.
    """
    if len(per_record_logprobs) != len(correct_letters):
        raise ValueError(
            f"got {len(per_record_logprobs)} rows for {len(correct_letters)} answers"
        )
    if letters is None:
        letters = [MATH_LETTERS] * len(correct_letters)
    if not correct_letters:
        return 0.0
    hits = sum(
        letter_from_logprobs(row, options) == correct.strip()
        for row, correct, options in zip(per_record_logprobs, correct_letters, letters)
    )
    return hits / len(correct_letters)


def macro_mean(per_student: Mapping[str, float]) -> float:
    """The mean of per-student scores, each student weighted equally."""
    if not per_student:
        return 0.0
    return sum(per_student.values()) / len(per_student)
