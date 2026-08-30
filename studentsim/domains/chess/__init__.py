"""Chess domain: per-player simulators of move choice and post-guidance updates.

Source corpus: Lichess May 2025 standard-time-control export, public CC0.
Response space: single UCI move per record (e.g., ``"d8d4"``).
Guidance modes: error remediation, comparative, strategic, socratic.
"""

from studentsim.domains.chess.modes import (
    CHESS_MODE_NAMES,
    CHESS_MODES,
    COMPARATIVE,
    ERROR_REMEDIATION,
    SOCRATIC,
    STRATEGIC,
)
from studentsim.domains.chess.profile import (
    ChessProfileFields,
    build_chess_profile,
    render_chess_profile,
)
from studentsim.domains.chess.prompts import (
    build_chess_multi_turn_messages,
    build_chess_single_turn_user_text,
)

__all__ = [
    "CHESS_MODES",
    "CHESS_MODE_NAMES",
    "COMPARATIVE",
    "ChessProfileFields",
    "ERROR_REMEDIATION",
    "SOCRATIC",
    "STRATEGIC",
    "build_chess_multi_turn_messages",
    "build_chess_profile",
    "build_chess_single_turn_user_text",
    "render_chess_profile",
]
