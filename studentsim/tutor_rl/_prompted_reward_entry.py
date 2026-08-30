"""The reward callback for the run whose student is a prompted model.

Same shape as :mod:`studentsim.tutor_rl._reward_entry`: verl hands over the
tutor message and a payload naming the position and the move the student had
played, and gets back one scalar. What differs is where the revised move comes
from, which is the whole of the comparison between the two.

No gates. They read a trained simulator's own backbone, and a model reached
through an API exposes none, so this is the move-quality term alone.

The client and the answers it has already given are kept per worker. A rollout
group asks about one position several times, and every ask is a paid call.
"""

from __future__ import annotations

import threading
from typing import Any

from studentsim.tutor_rl._reward_entry import parse_ground_truth, read_reward_table
from studentsim.tutor_rl.prompted_student import PromptedStudent
from studentsim.tutor_rl.reward import move_quality

_STUDENT: PromptedStudent | None = None
_TABLE: dict[tuple[str, str], float] | None = None
_LOAD_LOCK = threading.Lock()


def _load(*, student_model: str, reward_table: str) -> None:
    """Open this worker's client and read its copy of the reward table."""
    global _STUDENT, _TABLE
    # Several scoring threads arrive here before the first has finished, and
    # each would otherwise open its own client and its own cache, so the same
    # question would be paid for once per thread.
    with _LOAD_LOCK:
        if _STUDENT is None:
            from studentsim.core.llm import open_client

            _STUDENT = PromptedStudent(client=open_client(student_model))
        if _TABLE is None:
            _TABLE = read_reward_table(reward_table)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **reward_kwargs: Any,
) -> float:
    """Score one rollout against a prompted student's revised move."""
    fen, _color, wrong_move = parse_ground_truth(ground_truth)
    _load(
        student_model=reward_kwargs["student_model"],
        reward_table=reward_kwargs["reward_table"],
    )
    assert _STUDENT is not None and _TABLE is not None

    post_move = _STUDENT.move(
        fen=fen, wrong_move=wrong_move, tutor_message=solution_str or ""
    )
    return move_quality(
        _TABLE.get((fen, post_move)),
        _TABLE.get((fen, wrong_move)),
        mate_clip_cp=int(reward_kwargs.get("mate_clip_cp", 1500)),
        scale_cp=int(reward_kwargs.get("scale_cp", 500)),
        illegal_reward=float(reward_kwargs.get("illegal_reward", -1.0)),
    )
