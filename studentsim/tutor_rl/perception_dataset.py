"""Assembling what the perception head trains on.

The head has to recognise six ways a tutor message can misdescribe the board,
and no single source of labels covers all six. Rules read the position with a
chess library and settle where a piece stands. They cannot settle whether a
piece named at no square exists at all, or whether a move described in prose is
legal: those are phrased too many ways for patterns to match. A model reading the same text finds those, and
costs money per message. So the set is assembled from three sources, each
supplying what it is good for.

Judgements from a model carry every class and are trusted fully. Rules over
post-RL text supply bulk labels for the two classes they read well, with the
rest left at zero for the judgements to teach. Rules over pre-RL text supply
negatives: messages a careful reader finds nothing wrong with, which is what
stops the head concluding that the way a trained tutor writes is itself the
error.

The split groups by position. A position that appears in training must not
appear in validation under a different message, or the head is scored on
boards it has already been fitted to.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

from studentsim.tutor_rl.multihead import ERROR_TYPES

JUDGED_WEIGHT: Final = 1.0
RULE_WEIGHT: Final = 0.5
"""Rules are weaker supervision than a reading of the text, and weigh half."""

RULE_RELIABLE: Final = ("wrong_square", "wrong_piece")
"""The classes rules read well enough to label in bulk."""

VALIDATION_BUCKETS: Final = 10
"""Positions land in one of a hundred buckets; this many go to validation."""


def labels_from_judgement(errors: Sequence[dict] | None) -> dict[str, int]:
    """Reduce a judge's list of faults to the six the head models.

    A judge names more kinds of error than the head has classes, including
    misnamed moves and wrong evaluations. Those are dropped rather than folded
    into a neighbouring class, which would teach the head a boundary the gate
    does not use.
    """
    named = {error.get("type") for error in (errors or [])}
    return {error_type: int(error_type in named) for error_type in ERROR_TYPES}


def fen_bucket(fen: str) -> int:
    """Which of a hundred buckets a position falls in, stably."""
    return int.from_bytes(hashlib.md5(fen.encode()).digest()[:4], "big") % 100


def is_validation(fen: str, buckets: int = VALIDATION_BUCKETS) -> bool:
    return fen_bucket(fen) < buckets


def _row(fen: str, wrong_move: str, text: str, labels: dict[str, int],
         source: str, weight: float, subset: str) -> dict:
    return {
        "fen": fen,
        "wrong_move": wrong_move,
        "tutor_text": text,
        "perception": [float(labels[e]) for e in ERROR_TYPES],
        "label_source": source,
        "weight": weight,
        "subset": subset,
        "split": "val" if is_validation(fen) else "train",
    }


def from_judgements(generations: dict, judgements: dict, *, name: str = "") -> Iterator[dict]:
    """Pair each generated message with the judgement of it.

    The two files line up by position, one list per side, so the pairing is by
    index and a length mismatch is an error rather than something to truncate.
    """
    for side in ("pre", "post"):
        texts = generations[f"answers_{side}"]
        verdicts = judgements[f"{side}_results"]
        if len(texts) != len(verdicts):
            raise ValueError(
                f"{name or 'judgements'} {side}: {len(verdicts)} verdicts for {len(texts)} texts"
            )
        for index, (text, verdict) in enumerate(zip(texts, verdicts)):
            if not text or not isinstance(text, str) or verdict is None:
                continue
            errors = verdict.get("errors")
            # A judgement that failed to parse says nothing, and reading it
            # as a clean message would teach the head that it is one. Such a
            # verdict carries an empty error list beside its own marker, which
            # is why the marker is what has to be read: the list alone cannot
            # tell "nothing wrong with this message" from "no answer about
            # this message", and the second is much the commoner of the two.
            if errors is None or "_parse_error" in verdict:
                continue
            yield _row(
                generations["fens"][index], generations["wrongs"][index], text,
                labels_from_judgement(errors), "judged", JUDGED_WEIGHT, f"{name}/{side}",
            )


def clean_negatives(generations: dict, *, name: str = "") -> Iterator[dict]:
    """Pre-RL messages the rules find nothing wrong with.

    Only the spotless ones are kept. A message the rules half-flag might still
    carry something they missed, and as a negative it would teach the head the
    opposite of what it should.
    """
    from studentsim.tutor_rl.perception_labels import label

    for index, text in enumerate(generations["answers_pre"]):
        if not text or not isinstance(text, str):
            continue
        fen = generations["fens"][index]
        if label(fen, text).any_error():
            continue
        yield _row(
            fen, generations["wrongs"][index], text,
            {e: 0 for e in ERROR_TYPES}, "rules", RULE_WEIGHT, f"{name}/clean",
        )


def rule_bulk(generations: dict, *, name: str = "") -> Iterator[dict]:
    """Post-RL messages labelled by rule, on the classes rules read well.

    The other four are left at zero, which here means the rules did not look
    rather than that the message is clean. The judged rows are what teach those.
    """
    from studentsim.tutor_rl.perception_labels import label

    for index, text in enumerate(generations["answers_post"]):
        if not text or not isinstance(text, str):
            continue
        fen = generations["fens"][index]
        found = label(fen, text).flags
        labels = {e: 0 for e in ERROR_TYPES}
        for error_type in RULE_RELIABLE:
            labels[error_type] = int(found[error_type])
        yield _row(
            fen, generations["wrongs"][index], text, labels,
            "rules", RULE_WEIGHT, f"{name}/bulk",
        )


def build(pairs: Sequence[tuple[Path, Path]], *, rule_bulk_rows: bool = True) -> list[dict]:
    """Assemble the labelled set from generation and judgement files."""
    rows: list[dict] = []
    for generations_path, judgements_path in pairs:
        name = Path(generations_path).stem
        generations = json.loads(Path(generations_path).read_text())
        judgements = json.loads(Path(judgements_path).read_text())
        rows.extend(from_judgements(generations, judgements, name=name))
        rows.extend(clean_negatives(generations, name=name))
        if rule_bulk_rows:
            rows.extend(rule_bulk(generations, name=name))
    return rows
