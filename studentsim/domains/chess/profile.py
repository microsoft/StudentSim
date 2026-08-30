"""Chess player profile schema and rendering.

The :class:`StudentProfile.features` dict for a chess player carries the typed
fields defined in :class:`ChessProfileFields`. :func:`build_chess_profile`
constructs a :class:`StudentProfile` from those fields (validating ranges along
the way), and :func:`render_chess_profile` renders the textual ``"Player
context:"`` block exactly as it appears in every chess training prompt.

The render format is load-bearing for reproducibility: the released chess
checkpoints were trained on this exact string format, so changing it would
silently degrade evaluation quality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from studentsim.core.records import StudentProfile
from studentsim.domains.chess.modes import CHESS_DOMAIN_NAME


@dataclass(frozen=True, slots=True)
class ChessProfileFields:
    """Typed schema for the ``features`` dict of a chess :class:`StudentProfile`.

    All fields are required so that downstream renderers never have to special-case
    missing data. Use :func:`build_chess_profile` to construct from a dict.

    Field semantics
    ---------------
    ``win_rate``, ``best_move_accuracy``, ``offensive_move_rate``,
    ``defensive_move_rate`` are probabilities in [0, 1]. ``avg_cpl`` is
    centipawn-loss (non-negative). ``piece_type_probs`` maps a piece name
    ("bishop", "knight", "queen", ...) to its usage probability over the
    player's recorded games; values sum to 1.0 across present pieces.
    ``last_3_game_results`` is a length-3 list of ``"win" | "loss" | "draw"``
    strings ordered oldest-to-newest. ``opening_move_predictions`` maps a UCI
    string to its probability among the player's first-move tendencies.
    ``candidate_moves`` is the Maia top-k (uci, probability) list rendered as
    the ``Likely next moves:`` line; an empty sequence omits the line, which
    is how the prompted-baseline prompts are built.
    """

    win_rate: float
    best_move_accuracy: float
    avg_cpl: float
    style: str
    offensive_move_rate: float
    defensive_move_rate: float
    piece_type_probs: Mapping[str, float] = field(default_factory=dict)
    last_3_game_results: tuple[str, ...] = ()
    player_consecutive_best_moves: int = 0
    opponent_consecutive_best_moves: int = 0
    pieces_gained_last_n: int = 0
    pieces_lost_last_n: int = 0
    opening_move_predictions: Mapping[str, float] = field(default_factory=dict)
    current_move_idx: int = 0
    candidate_moves: Sequence[tuple[str, float]] = ()

    def __post_init__(self) -> None:
        _check_unit_interval("win_rate", self.win_rate)
        _check_unit_interval("best_move_accuracy", self.best_move_accuracy)
        _check_unit_interval("offensive_move_rate", self.offensive_move_rate)
        _check_unit_interval("defensive_move_rate", self.defensive_move_rate)
        if self.avg_cpl < 0:
            raise ValueError(f"avg_cpl must be non-negative, got {self.avg_cpl}")
        for piece, p in self.piece_type_probs.items():
            _check_unit_interval(f"piece_type_probs[{piece}]", p)
        for mv, p in self.opening_move_predictions.items():
            _check_unit_interval(f"opening_move_predictions[{mv}]", p)
        for mv, p in self.candidate_moves:
            _check_unit_interval(f"candidate_moves[{mv}]", p)
        for r in self.last_3_game_results:
            if r not in {"win", "loss", "draw"}:
                raise ValueError(
                    f"last_3_game_results entry must be win/loss/draw, got {r!r}"
                )


def _check_unit_interval(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def build_chess_profile(student_id: str, fields: ChessProfileFields) -> StudentProfile:
    """Construct a :class:`StudentProfile` for a chess player from typed fields.

    Tuples in the fields are normalized to lists so the features dict
    round-trips cleanly through JSON (tuples are not a JSON type).
    """
    features = _normalize_for_json(asdict(fields))
    return StudentProfile(
        student_id=student_id,
        domain=CHESS_DOMAIN_NAME,
        features=features,
    )


def _normalize_for_json(value):
    """Recursively replace tuples with lists so the value is JSON-roundtrippable."""
    if isinstance(value, tuple):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, list):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_for_json(v) for k, v in value.items()}
    return value


def render_chess_profile(profile: StudentProfile) -> str:
    """Render the ``Player context:`` block.

    The output begins with ``"Player context:"`` on its own line, followed by
    indented feature lines (two-space indent). Returns ``""`` only when called
    with a profile whose features dict is empty (no feature lines to emit), which
    in production never happens because :func:`build_chess_profile` always emits
    all required fields.
    """
    if profile.domain != CHESS_DOMAIN_NAME:
        raise ValueError(
            f"render_chess_profile called with non-chess profile (domain={profile.domain!r})"
        )

    f = profile.features
    lines: list[str] = []

    # Basic stats.
    if "win_rate" in f:
        lines.append(f"  Recent win rate: {f['win_rate'] * 100:.1f}%")
    if "best_move_accuracy" in f:
        lines.append(f"  Best move accuracy: {f['best_move_accuracy'] * 100:.2f}%")
    if "avg_cpl" in f:
        lines.append(f"  Avg centipawn loss: {f['avg_cpl']:.2f}")

    # Style stats.
    if "style" in f:
        lines.append(f"  Playing style: {f['style']}")
    if "offensive_move_rate" in f and "defensive_move_rate" in f:
        lines.append(
            f"  Offensive move rate: {f['offensive_move_rate'] * 100:.2f}%, "
            f"defensive: {f['defensive_move_rate'] * 100:.2f}%"
        )

    # Piece usage.
    piece_probs = f.get("piece_type_probs") or {}
    if piece_probs:
        piece_str = ", ".join(
            f"{k}: {v * 100:.2f}%" for k, v in sorted(piece_probs.items())
        )
        lines.append(f"  Piece usage: {piece_str}")

    # History.
    last_results = f.get("last_3_game_results") or ()
    if last_results:
        lines.append(f"  Last 3 game results: {', '.join(last_results)}")
    if "player_consecutive_best_moves" in f and "opponent_consecutive_best_moves" in f:
        lines.append(
            f"  Consecutive best moves by player: {f['player_consecutive_best_moves']}, "
            f"by opponent: {f['opponent_consecutive_best_moves']}"
        )
    if "pieces_gained_last_n" in f and "pieces_lost_last_n" in f:
        lines.append(
            f"  Recent captures: gained {f['pieces_gained_last_n']}, "
            f"lost {f['pieces_lost_last_n']}"
        )

    # Opening repertoire (only shown for early-game positions).
    if int(f.get("current_move_idx", 100)) <= 10:
        opening_preds = f.get("opening_move_predictions") or {}
        if len(opening_preds) == 1:
            mv = next(iter(opening_preds))
            lines.append(f"  Next move tendency: {mv}")
        elif len(opening_preds) >= 2:
            sorted_moves = sorted(opening_preds.items(), key=lambda x: -x[1])[:5]
            pred_str = ", ".join(f"{mv} ({prob * 100:.1f}%)" for mv, prob in sorted_moves)
            lines.append(f"  Opening move tendencies: {pred_str}")

    # Predicted next moves, when the profile carries any. The line is omitted
    # rather than left empty, so a profile without them reads the same either
    # way.
    candidates = f.get("candidate_moves") or ()
    if candidates:
        parts = [f"{mv} ({prob * 100:.1f}%)" for mv, prob in candidates]
        lines.append(f"  Likely next moves: {', '.join(parts)}")

    if not lines:
        return ""
    return "Player context:\n" + "\n".join(lines)
