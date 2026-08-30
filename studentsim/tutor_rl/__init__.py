"""Training a chess tutor against a student simulator.

This subpackage is chess-only and depends on
:mod:`studentsim.domains.chess`. The tutor policy, its supervised starting
point and every training hyperparameter are held fixed, so what a run is
comparing is the reward and nothing else.

:class:`GatedReward` is one: a trained student simulator's move-quality
improvement, scaled by a style gate and a perception gate read off that same
simulator's backbone. :class:`PromptedStudentReward` is the other: the same
move-quality term, with the revised move from a model prompted to play the
student. It carries no gates, because a model reached through an API exposes
no backbone for a head to read. :class:`StudentSimReward` is the move-quality
term on its own, which the gates multiply.
"""

from studentsim.tutor_rl.gates import (
    gated_reward,
    perception_gate,
    perception_weights,
    style_gate,
)
from studentsim.tutor_rl.multihead import (
    ERROR_TYPES,
    STYLE_LABELS,
    MultiHeadSimulator,
    parse_uci,
)
from studentsim.tutor_rl.prompted_student import PromptedStudent
from studentsim.tutor_rl.reward import (
    GatedReward,
    PromptedStudentReward,
    RewardModel,
    StudentSimReward,
)
from studentsim.tutor_rl.rl_config import (
    GpuLayout,
    RewardConfig,
    RLConfig,
)
from studentsim.tutor_rl.stockfish_cache import (
    InMemoryStockfishLookup,
    LayeredStockfishLookup,
    LiveStockfishLookup,
    SqliteStockfishLookup,
    StockfishLookup,
)
from studentsim.tutor_rl.verl_runner import VerlCommand, build_verl_command

__all__ = [
    "ERROR_TYPES",
    "STYLE_LABELS",
    "MultiHeadSimulator",
    "gated_reward",
    "parse_uci",
    "perception_gate",
    "perception_weights",
    "style_gate",
    "GatedReward",
    "GpuLayout",
    "PromptedStudent",
    "PromptedStudentReward",
    "InMemoryStockfishLookup",
    "LayeredStockfishLookup",
    "LiveStockfishLookup",
    "RLConfig",
    "RewardConfig",
    "RewardModel",
    "SqliteStockfishLookup",
    "StockfishLookup",
    "StudentSimReward",
    "VerlCommand",
    "build_verl_command",
]
