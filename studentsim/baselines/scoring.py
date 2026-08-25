"""Scoring a prompted baseline.

A baseline is not scored the way a trained simulator is, and the difference is
deliberate. A trained simulator answers in the format it was trained on, so it
is held to an exact match. A prompted model has never seen that format and
wraps its answer in whatever it likes, so holding it to the same standard would
measure formatting rather than behaviour.

Two things follow. Guidance is matched leniently: reasoning blocks and markup
come off, trailing punctuation goes, and an answer that matches any one of
several comma-separated alternatives counts. Math fidelity is read from the
model's per-letter likelihoods rather than from free text, since a closed model
may not emit a bare letter but will rank one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from studentsim.domains.math.fidelity import LOGPROB_FLOOR, MATH_LETTERS, argmax_letter

_THINK_BLOCK: Final = re.compile(r"<think>.*?</think>", re.DOTALL)
_MARKUP: Final = re.compile(r"<[^>]+>")
_WHITESPACE: Final = re.compile(r"\s+")
_TRAILING: Final = " .,;:!?%"


def normalize(text: str) -> str:
    """Strip what a prompted model adds around its answer."""
    stripped = _THINK_BLOCK.sub("", text or "")
    stripped = _MARKUP.sub("", stripped).strip()
    return _WHITESPACE.sub(" ", stripped).strip(_TRAILING).lower()


def _without_spaces(text: str) -> str:
    return _WHITESPACE.sub("", normalize(text))


def matches(prediction: str, reference: str) -> bool:
    """Whether a baseline's answer counts as the recorded one.

    Three readings are tried: the normalized forms agree, they agree once
    spacing is discarded, or the prediction matches one of the alternatives a
    comma-separated reference lists.
    """
    if not prediction or not reference:
        return False
    predicted, expected = normalize(prediction), normalize(reference)
    if predicted == expected:
        return True
    if _without_spaces(prediction) == _without_spaces(reference):
        return True
    for alternative in expected.split(","):
        alternative = alternative.strip()
        if not alternative:
            continue
        if predicted == alternative or _without_spaces(prediction) == _WHITESPACE.sub(
            "", alternative
        ):
            return True
    return False


def guidance_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    """The share of guided answers a baseline gets right."""
    if len(predictions) != len(references):
        raise ValueError(
            f"got {len(predictions)} predictions for {len(references)} references"
        )
    if not predictions:
        return 0.0
    return sum(matches(p, r) for p, r in zip(predictions, references)) / len(predictions)


def letter_from_logprobs(
    logprobs: Mapping[str, float], letters: Sequence[str] = MATH_LETTERS
) -> str:
    """The letter a model ranks first among the choices.

    A closed model returns only its top few tokens, so a letter it did not
    return is treated as very unlikely rather than as missing. That floor is
    what keeps the ranking defined when the answer sits outside the returned
    set.
    """
    return argmax_letter(
        {letter: float(logprobs.get(letter, LOGPROB_FLOOR)) for letter in letters},
        candidate_letters=letters,
    )


def math_fidelity_accuracy(
    per_record_logprobs: Sequence[Mapping[str, float]],
    correct_letters: Sequence[str],
    letters: Sequence[str] = MATH_LETTERS,
) -> float:
    """The share of problems where the baseline ranks the student's answer first."""
    if len(per_record_logprobs) != len(correct_letters):
        raise ValueError(
            f"got {len(per_record_logprobs)} rows for {len(correct_letters)} answers"
        )
    if not correct_letters:
        return 0.0
    hits = sum(
        letter_from_logprobs(row, letters) == correct.strip().upper()
        for row, correct in zip(per_record_logprobs, correct_letters)
    )
    return hits / len(correct_letters)
