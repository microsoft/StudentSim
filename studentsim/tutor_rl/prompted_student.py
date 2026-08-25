"""A closed model prompted to play the student, used as the reward's simulator.

This is the comparison the tutor RL result is measured against: the same
move-quality reward, over the same positions, with the revised move produced by
a frontier model asked to behave like a student instead of by a trained
simulator. Everything else about the run is held fixed, so what the comparison
isolates is where the move comes from.

The model is asked for the move as a JSON object naming the two squares rather
than as coordinate notation, because a model told to answer in UCI answers in
algebraic anyway; two squares are something it will spell reliably, and joining
them gives the notation the reward table is keyed by.

A prompted student has no per-student parameters to be faithful with, so its
answer depends on the position and the tutor's message alone. That makes the
answers cacheable, which matters because each one is a paid call and rollouts
repeat positions.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Final

from studentsim.core.llm import LLMClient, Message

SYSTEM: Final = (
    "You are simulating an intermediate-level chess student. "
    "You receive a board position, your original incorrect move, and your tutor's advice. "
    "Decide what move to play next as this student would, following the tutor's guidance. "
    "If the tutor names a specific move or direction, follow it. "
    "If the advice is vague, make a plausible choice.\n\n"
    'Reply with ONLY a JSON object in this exact format: {"from": "XX", "to": "YY"}\n'
    "where XX and YY are chess squares (one letter a-h plus one digit 1-8).\n"
    'Example: {"from": "e2", "to": "e4"} means move the piece on e2 to e4.\n'
    'Example: {"from": "g1", "to": "f3"} means move the piece on g1 to f3.\n'
    "No other text. Just the JSON."
)

USER_TEMPLATE: Final = """\
Board (FEN): {fen}
Your original incorrect move: {wrong_move}
Tutor's advice:
---
{text}
---
What move do you play? Reply ONLY with: {{"from": "XX", "to": "YY"}}"""

MAX_TOKENS: Final = 64
CACHE_TEXT_CHARS: Final = 600
"""How much of the tutor message identifies a call. Long messages agree on
their opening, and the student's answer follows the advice it opens with."""

_SQUARE = re.compile(r"[a-h][1-8]")
_UCI = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


def parse_move(reply: str | None) -> str:
    """Read the move out of the reply, in the three shapes it comes back in.

    The asked-for JSON first, then coordinate notation written out, then any
    two squares in the text. Coordinate notation is tried before loose squares
    because a promotion carries a fifth character that reading two squares
    would drop, and the shortened move is not in the table, so a legal
    promotion would score as one that never happened.

    An unreadable reply gives the empty string, which the caller scores the
    same way. Guessing a move instead would have it scored as though the
    student had played it.
    """
    if not reply:
        return ""
    text = reply.strip()
    try:
        parsed = json.loads(text)
        source = str(parsed.get("from", "")).lower().strip()
        target = str(parsed.get("to", "")).lower().strip()
        if _SQUARE.fullmatch(source) and _SQUARE.fullmatch(target):
            return source + target
    except (json.JSONDecodeError, AttributeError):
        pass
    lowered = text.lower()
    found = _UCI.search(lowered)
    if found:
        return found.group(1)
    squares = _SQUARE.findall(lowered)
    return squares[0] + squares[1] if len(squares) >= 2 else ""


def cache_key(fen: str, wrong_move: str, tutor_message: str) -> str:
    raw = f"{fen}|{wrong_move}|{tutor_message[:CACHE_TEXT_CHARS]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


@dataclass
class PromptedStudent:
    """Ask a model what the student plays after reading the tutor.

    Returns the move in coordinate notation, or the empty string when the
    reply could not be read as a move. A failed call is the same answer: the
    reward treats it as a move that is not on the board, which is the same
    thing a student writing nonsense would earn.
    """

    client: LLMClient
    max_tokens: int = MAX_TOKENS
    temperature: float = 0.0
    cache: dict[str, str] = field(default_factory=dict)
    calls: int = 0
    """How many times the model was actually asked, cache misses only."""

    def move(self, *, fen: str, wrong_move: str, tutor_message: str) -> str:
        key = cache_key(fen, wrong_move, tutor_message)
        if key in self.cache:
            return self.cache[key]
        messages = [
            Message(role="system", content=SYSTEM),
            Message(
                role="user",
                content=USER_TEMPLATE.format(
                    fen=fen, wrong_move=wrong_move, text=tutor_message
                ),
            ),
        ]
        self.calls += 1
        try:
            reply = self.client.complete(
                messages, max_tokens=self.max_tokens, temperature=self.temperature
            ).text
        except Exception:  # noqa: BLE001 - a failed call scores, it does not stop the run
            reply = None
        move = parse_move(reply)
        self.cache[key] = move
        return move
