"""Prompt rendering for chess single-turn and multi-turn records.

The exact string format here must remain consistent with the templates the
released chess checkpoints were trained on; rewording would silently degrade
evaluation. Multi-turn records carry four messages.
"""

from __future__ import annotations

from collections.abc import Mapping

from studentsim.core.records import (
    MultiTurnRecord,
    SingleTurnRecord,
    StudentProfile,
)
from studentsim.domains.chess.modes import CHESS_DOMAIN_NAME
from studentsim.domains.chess.profile import render_chess_profile


def build_chess_single_turn_user_text(
    *,
    profile: StudentProfile,
    fen: str,
    player_color: str,
) -> str:
    """Render the chess single-turn user prompt.

    ``player_color`` is one of ``"white" | "black"``; it is title-cased in
    the rendered text.
    """
    if profile.domain != CHESS_DOMAIN_NAME:
        raise ValueError(
            f"build_chess_single_turn_user_text: profile.domain must be 'chess', "
            f"got {profile.domain!r}"
        )
    if player_color.lower() not in {"white", "black"}:
        raise ValueError(f"player_color must be 'white' or 'black', got {player_color!r}")

    context_block = render_chess_profile(profile)
    color_cap = player_color.capitalize()

    if context_block:
        return (
            "You are playing chess as a specific player.\n\n"
            f"{context_block}\n\n"
            "The current position in FEN notation is:\n"
            f"{fen}\n\n"
            f"You are playing as {color_cap}. "
            "What is your next move? Respond in UCI format."
        )
    return (
        "You are playing chess. The current position in FEN notation is:\n"
        f"{fen}\n\n"
        f"You are playing as {color_cap}. "
        "What is your next move? Respond in UCI format."
    )


def build_chess_multi_turn_messages(
    *,
    profile: StudentProfile,
    fen: str,
    player_color: str,
    student_wrong_move: str,
    tutor_message: str,
) -> list[dict[str, str]]:
    """Render the chess multi-turn chat (first three messages, the prompt seen at eval).

    Returns a list of three messages: ``user`` (the same single-turn prompt),
    ``assistant`` (the student's wrong move), and ``user`` (the tutor's natural-
    language guidance). The model is expected to generate the fourth message
    (the canonical correction). Caller appends the reference ``assistant``
    message for training and uses this list verbatim at inference for the
    guidance-responsiveness eval.
    """
    user_text = build_chess_single_turn_user_text(
        profile=profile, fen=fen, player_color=player_color
    )
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": student_wrong_move},
        {"role": "user", "content": tutor_message},
    ]


def _read_fen_and_color(record: SingleTurnRecord | MultiTurnRecord) -> tuple[str, str]:
    """Extract FEN + player color from a record's ``meta`` dict.

    Single- and multi-turn records carry the same two keys, so one reader
    serves both.
    """
    meta: Mapping[str, object] = record.meta
    fen = meta.get("fen")
    color = meta.get("player_color")
    if not isinstance(fen, str):
        raise ValueError(f"record.meta missing string 'fen': {meta!r}")
    if not isinstance(color, str):
        raise ValueError(f"record.meta missing string 'player_color': {meta!r}")
    return fen, color


def render_chess_single_turn(record: SingleTurnRecord) -> str:
    """Render the user-message text for a chess single-turn record.

    A thin adapter over :func:`build_chess_single_turn_user_text`.
    """
    if record.domain != CHESS_DOMAIN_NAME:
        raise ValueError(f"non-chess record: domain={record.domain!r}")
    fen, color = _read_fen_and_color(record)
    return build_chess_single_turn_user_text(
        profile=record.profile, fen=fen, player_color=color
    )


def render_chess_multi_turn(record: MultiTurnRecord) -> str:
    """Render the multi-turn prompt as a flattened text block.

    Concatenates the three-message chat into a single string suitable for the
    ``Domain.render_multi_turn_prompt`` contract (which returns ``str``). The
    flattened form uses ``ROLE: content`` delimiters; callers needing the raw
    message list (training data prep, judge eval) should use
    :func:`build_chess_multi_turn_messages` directly.
    """
    if record.domain != CHESS_DOMAIN_NAME:
        raise ValueError(f"non-chess record: domain={record.domain!r}")
    fen, color = _read_fen_and_color(record)
    msgs = build_chess_multi_turn_messages(
        profile=record.profile,
        fen=fen,
        player_color=color,
        student_wrong_move=record.wrong_response,
        tutor_message=record.tutor_guidance,
    )
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)
