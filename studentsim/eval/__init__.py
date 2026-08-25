"""Evaluation: decode a trained simulator and score it.

One student is evaluated by decoding their held-out records with
``swift infer`` against their adapter, then comparing each decode to what the
student actually did. :func:`evaluate_student` runs both metrics for one
student, :func:`evaluate_students` runs a roster, and the aggregate functions
reduce a roster to the domain-level numbers.
"""

from studentsim.eval.aggregate import (
    Aggregate,
    CrossSeed,
    aggregate_across_seeds,
    aggregate_fidelity,
    aggregate_responsiveness,
)
from studentsim.eval.checkpoints import find_last_checkpoint
from studentsim.eval.fidelity import (
    score_error_density,
    score_letter_choice,
    score_move_match,
)
from studentsim.eval.infer import InferCommand, build_infer_command, run_infer
from studentsim.eval.normalize import normalize_answer, normalize_move, normalizer_for
from studentsim.eval.protocol import EvalProtocol, protocol_for
from studentsim.eval.runner import (
    StudentResult,
    apply_turn2_suffix,
    evaluate_student,
    evaluate_students,
)
from studentsim.eval.scoring import ModeScore, Score, score_results

__all__ = [
    "Aggregate",
    "CrossSeed",
    "EvalProtocol",
    "InferCommand",
    "ModeScore",
    "Score",
    "StudentResult",
    "aggregate_across_seeds",
    "aggregate_fidelity",
    "aggregate_responsiveness",
    "apply_turn2_suffix",
    "build_infer_command",
    "evaluate_student",
    "evaluate_students",
    "find_last_checkpoint",
    "normalize_answer",
    "normalize_move",
    "normalizer_for",
    "protocol_for",
    "run_infer",
    "score_error_density",
    "score_letter_choice",
    "score_move_match",
    "score_results",
]
