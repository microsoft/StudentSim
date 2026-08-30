"""The reward callback verl calls once per rollout.

verl hands over the tutor message the actor produced and a payload naming the
position and the move the student had played. Scoring it means asking the
simulator what the student plays after reading that message, looking the
resulting position up in the precomputed Stockfish table, and multiplying by
the two gates.

The multi-head simulator is loaded once per worker and kept, because loading it
per rollout would dominate the run.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from studentsim.tutor_rl.gates import (
    ALPHA_PERCEPTION,
    ALPHA_STYLE,
    gated_reward,
    perception_weights,
)
from studentsim.tutor_rl.multihead import MultiHeadSimulator
from studentsim.tutor_rl.reward import move_quality

_SIMULATOR: MultiHeadSimulator | None = None
_WEIGHTS: tuple[float, ...] | None = None
_TABLE: dict[tuple[str, str], float] | None = None
_LOAD_LOCK = threading.Lock()

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _load(
    *,
    adapter_path: str,
    heads_checkpoint: str,
    heads_metrics: str,
    reward_table: str,
    base_model: str,
    device: str,
) -> None:
    """Bring up this worker's simulator, gate weights, and reward table."""
    global _SIMULATOR, _WEIGHTS, _TABLE
    # A worker scores its rollouts on several threads, and all of them arrive
    # here before the first has anything to show for it. Without the lock they
    # each build a simulator over the top of the others, and a half-built model
    # has weights that cannot be moved onto the GPU.
    with _LOAD_LOCK:
        if _SIMULATOR is None:
            _SIMULATOR = MultiHeadSimulator(
                base_model=base_model,
                adapter_path=adapter_path,
                heads_checkpoint=heads_checkpoint,
                device=device,
            )
        if _WEIGHTS is None:
            _WEIGHTS = perception_weights(heads_metrics)
        if _TABLE is None:
            _TABLE = read_reward_table(reward_table)


def read_reward_table(path: str | Path) -> dict[tuple[str, str], float]:
    """Read the precomputed move-quality lookup.

    One row per (position, move) the playground can reach, holding the
    centipawn evaluation after that move. Rollouts that land outside the table
    are treated as illegal.
    """
    table: dict[tuple[str, str], float] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            table[(row["fen"], row["move"])] = float(row["cp"])
    return table


def parse_ground_truth(payload: str) -> tuple[str, str, str]:
    """Unpack the ``fen|color|wrong_move`` payload verl carries through."""
    parts = payload.split("|")
    if len(parts) != 3:
        raise ValueError(f"expected ground_truth = 'fen|color|wrong_move', got {payload!r}")
    return parts[0], parts[1], parts[2]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **reward_kwargs: Any,
) -> float:
    """Score one rollout.

    ``solution_str`` is the tutor message the actor wrote; ``ground_truth``
    names the position and the move the student had played.
    """
    fen, _color, wrong_move = parse_ground_truth(ground_truth)
    _load(
        adapter_path=reward_kwargs["adapter_path"],
        heads_checkpoint=reward_kwargs["heads_checkpoint"],
        heads_metrics=reward_kwargs["heads_metrics"],
        reward_table=reward_kwargs["reward_table"],
        base_model=reward_kwargs.get("base_model", BASE_MODEL),
        device=reward_kwargs.get("device", "cuda:0"),
    )
    assert _SIMULATOR is not None and _WEIGHTS is not None and _TABLE is not None

    signals = _SIMULATOR.infer_batch([fen], [wrong_move], [solution_str])
    quality = move_quality(
        _TABLE.get((fen, signals.moves[0])),
        _TABLE.get((fen, wrong_move)),
        mate_clip_cp=int(reward_kwargs.get("mate_clip_cp", 1500)),
        scale_cp=int(reward_kwargs.get("scale_cp", 500)),
        illegal_reward=float(reward_kwargs.get("illegal_reward", -1.0)),
    )
    return gated_reward(
        quality,
        style_probs=signals.style_probs[0],
        perception_probs=signals.perception_probs[0],
        preferred_style=reward_kwargs.get("style_preferred"),
        weights=_WEIGHTS,
        alpha_style=float(reward_kwargs.get("alpha_style", ALPHA_STYLE)),
        alpha_perception=float(reward_kwargs.get("alpha_perception", ALPHA_PERCEPTION)),
    )
