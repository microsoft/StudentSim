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
    """Fail loudly when a result file lacks the decode or the reference.

    Both fields defaulting to an empty string would make every record match
    itself, which reads as a perfect score rather than as a broken run.
    """
    missing = [f for f in RESULT_FIELDS if f not in rows[0]]
    if missing:
        raise ValueError(
            f"{path} has no {' or '.join(missing)} field; scoring it would "
            f"compare empty strings. Fields present: {sorted(rows[0])}"
        )


def _read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_modes(input_path: str | Path, mode_field: str) -> list[str]:
    """Read the per-record guidance mode from an evaluation input file."""
    return [str(row.get(mode_field, "")) for row in _read_jsonl(input_path)]


def score_results(
    result_path: str | Path,
    *,
    normalize: Normalizer,
    strip_think: bool = False,
    input_path: str | Path | None = None,
    mode_field: str = "instruction_type",
) -> Score:
    """Score a ``swift infer`` result file.

    Passing ``input_path`` adds the per-mode breakdown, read from
    ``mode_field``; without it only the overall accuracy is returned.
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
        hit = int(
            normalize(str(row.get("response", "")), strip_think)
            == normalize(str(row.get("labels", "")), strip_think)
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
