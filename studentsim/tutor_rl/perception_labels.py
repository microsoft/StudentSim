"""Labelling what a tutor message gets wrong about the board.

The perception head learns to spot six ways guidance can misdescribe a
position. Its training labels come from checking each concrete claim the text
makes against the position with a chess library, which makes them cheap and
deterministic.

The rules are written for precision rather than coverage. A false positive
teaches the head that clean guidance is dirty, which is the expensive mistake;
a claim phrased in a way no rule matches is simply not labelled, and the head
sees it as clean. What the rules miss is not spread evenly. They
recover under 15% of two classes, hallucinated pieces and illegal moves, and
model judgements cover those, and :mod:`studentsim.tutor_rl.perception_dataset` mixes them in.
A head trained on rules alone therefore learns a differently shaped gate: it is
carried by illegal moves, which the rules do catch once a line is walked, and it
barely registers hallucinated pieces, which carry about a fifth of the weight in
the recorded gate. Adding further rules does not close that; the supervision
does.

Two of the six errors are what the corpus filter already drops rows for, and
this module reads them the same way it does: a move is illegal when walking the
line cannot reach it, and a piece claim is checked in the same three phrasings.
What is added here is which error a claim makes, since the filter only needs to
know that one was made.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final

from studentsim.tutor_rl.multihead import ERROR_TYPES

PIECE_TYPES: Final = ("king", "queen", "rook", "bishop", "knight", "pawn")

_PIECES = "|".join(PIECE_TYPES)
_OWNERS = (
    r"your|yours|the opponent'?s|opponent'?s|their|white'?s|black'?s|white|black|its|the"
)
_OWNER = rf"((?:{_OWNERS})\s+)?"
_SQUARE = r"([a-h][1-8])"

_PIECE_ON: Final = re.compile(
    rf"\b{_OWNER}({_PIECES})\s+(?:is\s+)?on\s+{_SQUARE}\b", re.IGNORECASE
)
_PIECE_AT: Final = re.compile(
    rf"\b{_OWNER}({_PIECES})\s+(?:is\s+)?at\s+{_SQUARE}\b", re.IGNORECASE
)
_SQUARE_PIECE: Final = re.compile(rf"\b{_SQUARE}\s+{_OWNER}({_PIECES})\b", re.IGNORECASE)
_CAPTURE: Final = re.compile(
    rf"\b(?:captur\w+|takes?)\s+(?:the\s+)?({_PIECES})\s+(?:on\s+)?{_SQUARE}\b", re.IGNORECASE
)


@dataclass
class PerceptionLabels:
    """Which of the six errors a message makes, and what showed it."""

    flags: dict[str, bool] = field(default_factory=lambda: {e: False for e in ERROR_TYPES})
    evidence: list[str] = field(default_factory=list)

    def vector(self) -> list[float]:
        """The label the head is trained against."""
        return [1.0 if self.flags[error] else 0.0 for error in ERROR_TYPES]

    def any_error(self) -> bool:
        return any(self.flags.values())


def _colour_of(owner: str, student):
    """Which side a claim assigns a piece to, if it assigns one.

    Guidance is addressed to the student, so "your rook" names a side just as
    plainly as "white's rook" does. The student is whoever is to move in the
    position, since it is their move the guidance is about.
    """
    import chess

    owner = owner.strip().lower()
    if owner.startswith("white"):
        return chess.WHITE
    if owner.startswith("black"):
        return chess.BLACK
    if owner.startswith("your"):
        return student
    if owner.startswith(("opponent", "the opponent", "their")):
        return not student
    return None


def _piece_exists(board, piece_type: int, colour) -> bool:
    return bool(board.pieces(piece_type, colour)) if colour is not None else bool(
        board.pieces(piece_type, True) or board.pieces(piece_type, False)
    )


def _piece_claims(text: str) -> Iterator[tuple[str, str, str, str]]:
    """Every (phrase, owner, piece, square) the text asserts."""
    for pattern in (_PIECE_ON, _PIECE_AT):
        for match in pattern.finditer(text):
            yield match.group(0), match.group(1) or "", match.group(2), match.group(3)
    for match in _SQUARE_PIECE.finditer(text):
        yield match.group(0), match.group(2) or "", match.group(3), match.group(1)


def label(fen: str, text: str) -> PerceptionLabels:
    """Check every claim the text makes against the position."""
    import chess

    from studentsim.tutor_rl.sft_corpus import count_illegal_moves

    board = chess.Board(fen)
    result = PerceptionLabels()
    piece_index = {name: getattr(chess, name.upper()) for name in PIECE_TYPES}

    for phrase, owner, piece_name, square in _piece_claims(text):
        actual = board.piece_at(chess.parse_square(square.lower()))
        claimed_type = piece_index[piece_name.lower()]
        claimed_colour = _colour_of(owner, board.turn)
        if actual is None:
            # Nothing stands there. If the piece is nowhere on the board at all
            # the message invented it; otherwise it put a real piece elsewhere.
            key = (
                "hallucinated_piece"
                if not _piece_exists(board, claimed_type, claimed_colour)
                else "wrong_square"
            )
            result.flags[key] = True
            result.evidence.append(phrase)
        elif actual.piece_type != claimed_type:
            result.flags["wrong_piece"] = True
            result.evidence.append(phrase)
        elif claimed_colour is not None and actual.color != claimed_colour:
            result.flags["wrong_color"] = True
            result.evidence.append(phrase)

    for match in _CAPTURE.finditer(text):
        piece_name, square = match.group(1).lower(), match.group(2)
        actual = board.piece_at(chess.parse_square(square.lower()))
        if actual is None or actual.piece_type != piece_index[piece_name]:
            result.flags["wrong_capture"] = True
            result.evidence.append(match.group(0))

    if count_illegal_moves(text, fen):
        result.flags["illegal_move"] = True

    return result


