"""Turning decoded results into a score.

A result row carries the simulator's decode under ``response`` and the
student's recorded answer under ``labels``. Both go through the same
normalizer, and the score is the fraction of rows where they agree.

The per-mode breakdown reads the guidance mode from the *input* file rather
than the result file, aligning the two by row index. That alignment is why
inference runs with shuffling disabled.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from studentsim.eval.normalize import Normalizer


@dataclass(frozen=True)
class Score:
    """One student's score on one evaluation set."""

    accuracy: float
    n_samples: int
    per_mode: dict[str, "ModeScore"] = field(default_factory=dict)


@dataclass(frozen=True)
class ModeScore:
    accuracy: float
    n_samples: int


RESULT_FIELDS = ("response", "labels")


def require_result_fields(rows: list[dict], path: str | Path) -> None:
    """Fail loudly when any result row lacks the decode or the reference.

    Both fields defaulting to an empty string would make the row match
    itself, which reads as a perfect score rather than as a broken run.
    """
    for index, row in enumerate(rows):
        missing = [f for f in RESULT_FIELDS if f not in row]
        if missing:
            raise ValueError(
                f"{path} row {index} has no {' or '.join(missing)} field; scoring it "
                f"would compare empty strings. Fields present: {sorted(row)}"
            )


def _read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


MODE_FIELDS = ("instruction_type", "style")
"""Fields a record may carry its guidance mode under, the first one present wins."""


def record_mode(row: dict, mode_field: str = "instruction_type") -> str:
    """The guidance mode of one record, read from ``mode_field`` or ``style``."""
    for name in (mode_field, *MODE_FIELDS):
        if row.get(name):
            return str(row[name])
    return ""


def read_modes(input_path: str | Path, mode_field: str) -> list[str]:
    """Read the per-record guidance mode from an evaluation input file."""
    return [record_mode(row, mode_field) for row in _read_jsonl(input_path)]


def score_results(
    result_path: str | Path,
    *,
    normalize: Normalizer,
    strip_think: bool = False,
    input_path: str | Path | None = None,
    mode_field: str = "instruction_type",
    empty_is_miss: bool = False,
) -> Score:
    """Score a result file of ``response`` and ``labels`` rows.

    Passing ``input_path`` adds the per-mode breakdown, read from
    ``mode_field``; without it only the overall accuracy is returned. With
    ``empty_is_miss`` a row whose normalized decode is empty scores 0 even
    against an empty reference.
    """
    rows = list(_read_jsonl(result_path))
    if not rows:
        return Score(accuracy=0.0, n_samples=0)
    require_result_fields(rows, result_path)

    modes = read_modes(input_path, mode_field) if input_path is not None else []

    hits = 0
    mode_hits: dict[str, int] = {}
    mode_total: dict[str, int] = {}
    for index, row in enumerate(rows):
        predicted = normalize(str(row["response"]), strip_think)
        hit = int(
            predicted == normalize(str(row["labels"]), strip_think)
            and not (empty_is_miss and predicted == "")
        )
        hits += hit
        if index < len(modes) and modes[index]:
            mode = modes[index]
            mode_hits[mode] = mode_hits.get(mode, 0) + hit
            mode_total[mode] = mode_total.get(mode, 0) + 1

    per_mode = {
        mode: ModeScore(
            accuracy=round(mode_hits[mode] / mode_total[mode], 6),
            n_samples=mode_total[mode],
        )
        for mode in sorted(mode_total)
    }
    return Score(accuracy=hits / len(rows), n_samples=len(rows), per_mode=per_mode)
