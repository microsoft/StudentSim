"""Turning generated guidance into the corpus the tutor is trained on.

A tutor that says false things about the board teaches the student the wrong
thing, and the model will happily learn to say them if the corpus does. Four
things in this build are what keep that out.

The board is rendered as an image with the side to move at the bottom, so the
tutor sees the position the way the student does. The same position is written
out square by square in the text, so a claim about a piece has something to be
checked against. Answers that name moves the position does not allow are
dropped, and so are answers whose claims about which piece stands where
contradict the position. A share of the examples keeps its mode tag and the
rest have it stripped, so the tutor works with or without one.

Checking a move for legality is subtler than it looks. Guidance walks lines:
after "e2e4 e7e5" the next move is legal in the line, not at the position the
student is looking at. A move counts as real if it is legal either at the
position or as a continuation of the line so far, and only a move that is
neither is a hallucination.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from studentsim.core.seeds import DATA_SAMPLER_SEED

PIECE_NAMES: Final = ("king", "queen", "rook", "bishop", "knight", "pawn")

MODES: Final = ("error_remediation", "socratic", "strategic", "comparative")
"""The guidance modes, in the order that seeds each one's tag dropout."""

_UCI = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
_PIECE_ALTERNATION = "|".join(PIECE_NAMES)
_PIECE_ON = re.compile(rf"\b({_PIECE_ALTERNATION})\s+on\s+([a-h][1-8])\b", re.IGNORECASE)
_PIECE_AT = re.compile(rf"\b({_PIECE_ALTERNATION})\s+at\s+([a-h][1-8])\b", re.IGNORECASE)
_SQUARE_PIECE = re.compile(rf"\b([a-h][1-8])\s+({_PIECE_ALTERNATION})\b", re.IGNORECASE)

# A guidance answer that quotes its source is describing its own prompt. The
# giveaway is usually a number: an engine evaluation reaches the text as two
# decimal places, which no chess explanation otherwise produces.
_LEAK_BASIC = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (r"\bengine\b", r"[+-]?\b\d+\.\d{2}\b")
)
_LEAK_STRICT = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (r"\bcentipawn(s)?\b", r"\bstockfish\b", r"\bcomputer\b", r"\bevaluator\b")
)
MIN_STRICT_LENGTH: Final = 100

DEFAULT_MODE_TAG_DROPOUT: Final = 0.25
DEFAULT_MAX_ILLEGAL: Final = 1
DEFAULT_VAL_FRACTION: Final = 0.05
DEFAULT_TEST_FRACTION: Final = 0.05
BOARD_SIZE: Final = 512
ARROW_COLOR: Final = "#cc0000"
ARROW_OPACITY: Final = 0.5


def render_board(fen: str, wrong_move: str | None = None, *, size: int = BOARD_SIZE) -> bytes:
    """Draw the position as a PNG, seen from the side that is to move.

    Orienting by side to move is what spares the tutor from mentally flipping
    the board when Black is to move. The student's move is drawn as a
    translucent arrow, translucent so the pieces under it stay readable.
    """
    import cairosvg
    import chess
    import chess.svg

    board = chess.Board(fen)
    arrows = []
    if wrong_move:
        try:
            arrows = [
                chess.svg.Arrow(
                    chess.parse_square(wrong_move[:2].lower()),
                    chess.parse_square(wrong_move[2:4].lower()),
                    color=ARROW_COLOR,
                )
            ]
        except ValueError:
            arrows = []
    svg = chess.svg.board(board, arrows=arrows, size=size, orientation=board.turn)
    # cairosvg drops the alpha channel of an eight-digit hex colour, so the
    # opacity is set through the class python-chess stamps on its arrows.
    style = f"<style>.arrow {{ opacity: {ARROW_OPACITY}; }}</style>"
    svg = svg.replace("</svg>", f"{style}</svg>", 1)
    return cairosvg.svg2png(bytestring=svg.encode("utf-8"))


def count_illegal_moves(answer: str, fen: str) -> int:
    """How many moves the answer names that the position cannot reach.

    Moves are walked in order. One that continues the line so far is fine, and
    so is one that starts a fresh line from the position under discussion. A
    move that fits neither reading did not exist on any board.
    """
    import chess

    root = chess.Board(fen)
    root_legal = {move.uci().lower() for move in root.legal_moves}
    line = chess.Board(fen)
    illegal = 0
    for match in _UCI.finditer(answer):
        uci = match.group(1).lower()
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            illegal += 1
            continue
        if move in line.legal_moves:
            line.push(move)
            continue
        if uci in root_legal:
            line = chess.Board(fen)
            line.push_uci(uci)
            continue
        illegal += 1
    return illegal


def contradicts_position(answer: str, fen: str) -> bool:
    """Whether the answer puts a piece somewhere it is not.

    Three phrasings are checked: a piece "on" a square, a piece "at" a square,
    and a square followed by a piece. An empty square counts as a
    contradiction, since the answer asserted something stands there.
    """
    import chess

    board = chess.Board(fen)

    def wrong(piece_name: str, square: str) -> bool:
        actual = board.piece_at(chess.parse_square(square.lower()))
        if actual is None:
            return True
        return chess.PIECE_NAMES[actual.piece_type] != piece_name.lower()

    for pattern in (_PIECE_ON, _PIECE_AT):
        for match in pattern.finditer(answer):
            if wrong(match.group(1), match.group(2)):
                return True
    for match in _SQUARE_PIECE.finditer(answer):
        if wrong(match.group(2), match.group(1)):
            return True
    return False


def leaks_instructions(answer: str, level: str = "basic") -> bool:
    """Whether the answer tells the student where its information came from.

    A tutor that cites an engine is describing its own prompt, which is not
    teaching and does not survive contact with a student. ``strict`` adds the
    words for the machinery itself and refuses an answer too short to be
    teaching anything.
    """
    if level == "none":
        return False
    if not answer:
        return True
    if any(pattern.search(answer) for pattern in _LEAK_BASIC):
        return True
    if level == "strict":
        if any(pattern.search(answer) for pattern in _LEAK_STRICT):
            return True
        if len(answer) < MIN_STRICT_LENGTH:
            return True
    return False


def student_colour(fen: str) -> str:
    """Which colour the student is playing, which is whoever is to move."""
    import chess

    return "black" if chess.Board(fen).turn == chess.BLACK else "white"


def pieces_by_square(fen: str) -> str:
    """Write the position out square by square.

    This is what a claim about the board can be checked against, and it gives
    the text channel the same information the image carries.
    """
    import chess

    board = chess.Board(fen)
    lines = []
    for color, label in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        placed = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.color == color:
                placed.append(f"{chess.PIECE_NAMES[piece.piece_type]} {chess.square_name(square)}")
        lines.append(f"{label}: " + ", ".join(placed))
    return "\n".join(lines)


@dataclass
class CorpusSpec:
    """What to drop and how to present what is kept."""

    max_illegal_moves: int = DEFAULT_MAX_ILLEGAL
    filter_piece_state: bool = True
    filter_instruction_leak: bool = True
    mode_tag_dropout: float = DEFAULT_MODE_TAG_DROPOUT
    include_color_hint: bool = True
    seed: int = DATA_SAMPLER_SEED
    mode_ratios: Mapping[str, float] | None = None
    max_rows: int | None = None
    val_fraction: float = DEFAULT_VAL_FRACTION
    test_fraction: float = DEFAULT_TEST_FRACTION

    def __post_init__(self) -> None:
        if not 0.0 <= self.mode_tag_dropout <= 1.0:
            raise ValueError(f"mode_tag_dropout must be in [0, 1], got {self.mode_tag_dropout}")


@dataclass(frozen=True)
class Rejection:
    """Why one generated answer did not make it into the corpus."""

    position_id: str
    mode: str
    reason: str


def accepts(answer: str, fen: str, spec: CorpusSpec) -> str | None:
    """The reason to drop this answer, or ``None`` to keep it."""
    illegal = count_illegal_moves(answer, fen)
    if illegal > spec.max_illegal_moves:
        return f"names {illegal} moves the position cannot reach"
    if spec.filter_piece_state and contradicts_position(answer, fen):
        return "describes a piece that is not there"
    if spec.filter_instruction_leak and leaks_instructions(answer):
        return "mentions where its information came from"
    return None


def build_example(
    row: dict,
    spec: CorpusSpec,
    rng: random.Random,
    *,
    image_path: str | None,
    system_prompts: Mapping[str, str] | None = None,
    generic_system_prompt: str | None = None,
) -> dict:
    """One training example: the position, and the guidance to reproduce.

    The system prompt leads the user turn rather than sitting in its own
    message. Which one it is follows the tag: a row that kept its mode tag gets
    that mode's prompt, and a row whose tag was stripped gets the generic one,
    since without the tag there is nothing to say which style is wanted.
    """
    keep_tag = rng.random() >= spec.mode_tag_dropout
    # Sections are separated by a blank line and their own lines by one, which
    # is the shape the tutor was trained to read.
    sections = []
    if keep_tag:
        sections.append(f"MODE: {row['mode']}")
    sections.append(f"POSITION (FEN):\n{row['fen']}")
    sections.append(f"PIECES BY SQUARE:\n{pieces_by_square(row['fen'])}")
    student = ["STUDENT'S MOVE:"]
    if spec.include_color_hint:
        # The side to move is the student, so their colour is written out
        # rather than left to be read off the sixth field of the position.
        student.append(f"  Color: {student_colour(row['fen'])}")
    student.append(f"  Move: {row['wrong_move']}")
    sections.append("\n".join(student))
    sections.append("Explain to the student how to improve on their move.")
    user_text = "\n\n".join(sections)
    if system_prompts or generic_system_prompt:
        if keep_tag and system_prompts and row["mode"] in system_prompts:
            preamble = system_prompts[row["mode"]]
        elif generic_system_prompt:
            preamble = generic_system_prompt
        else:
            preamble = ""
        if preamble:
            user_text = preamble.strip() + "\n\n---\n\n" + user_text

    example = {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": row["instruction_uci"]},
        ],
        # The mode a row was generated under, kept whether or not its tag was
        # shown. Conflating the two loses every untagged row when the corpus is
        # balanced by mode, which is the quarter that teaches the tutor to work
        # without being told a style.
        "mode": row["mode"],
        "mode_tag_shown": keep_tag,
        "fen": row["fen"],
    }
    if image_path is not None:
        # The board goes in its own field and is referred to from the text,
        # which is the shape the training command reads.
        example["images"] = [image_path]
        example["messages"][0]["content"] = f"<image>\n{user_text}"
    return example


def build_corpus(
    rows: Sequence[dict], spec: CorpusSpec | None = None, *, image_path_for=None
) -> tuple[list[dict], list[Rejection]]:
    """Filter generated guidance and render what survives.

    Returns the examples and the rejections, because how much was dropped and
    why is worth reading before training on the rest.
    """
    spec = spec or CorpusSpec()
    # One generator per mode, seeded from its position in MODES. A single
    # generator walked over the rows in file order would drop a different set
    # of tags, since which rows it lands on depends on how many rows of other
    # modes preceded them.
    generators = {mode: dropout_rng(spec.seed, mode) for mode in MODES}
    examples: list[dict] = []
    rejections: list[Rejection] = []
    for row in rows:
        reason = accepts(row["instruction_uci"], row["fen"], spec)
        if reason is not None:
            rejections.append(
                Rejection(
                    position_id=row.get("position_id", ""), mode=row.get("mode", ""), reason=reason
                )
            )
            continue
        image_path = image_path_for(row) if image_path_for else None
        rng = generators.setdefault(row.get("mode", ""), dropout_rng(spec.seed, row.get("mode", "")))
        examples.append(build_example(row, spec, rng, image_path=image_path))
    return examples, rejections


def dropout_rng(seed: int, mode: str) -> random.Random:
    """The generator that decides which of one mode's rows keep their tag."""
    index = MODES.index(mode) if mode in MODES else len(MODES)
    return random.Random(seed * 31 + index)


def collection_caps(spec: CorpusSpec) -> dict[str, int] | None:
    """How many rows to collect per mode before the balanced draw.

    Twice the target plus a little, which leaves the draw something to choose
    from without holding the whole corpus in memory. Collecting everything and
    balancing afterwards needs tens of gigabytes on a corpus this size.
    """
    if not spec.mode_ratios or spec.max_rows is None:
        return None
    return {m: int(spec.max_rows * r * 2) + 100 for m, r in spec.mode_ratios.items()}


def allocate_counts(
    ratios: Mapping[str, float],
    available: Mapping[str, int],
    max_rows: int | None,
) -> dict[str, int]:
    """How many rows each mode contributes.

    The corpus is balanced across guidance modes, and the modes do not survive
    filtering in equal numbers, so the total is held to what the scarcest mode
    can supply at its share. No mode is ever upsampled to meet its quota.
    """
    bottleneck = min(available[m] / ratios[m] for m in ratios if ratios[m] > 0)
    target_total = bottleneck if max_rows is None else min(float(max_rows), bottleneck)
    return {m: int(target_total * ratios[m]) for m in ratios}


def balance_modes(
    rows: Sequence[dict], spec: CorpusSpec, *, mode_of=lambda r: r.get("mode")
) -> list[dict]:
    """Draw the balanced sample the tutor trains on.

    Filtering leaves the modes uneven, and a tutor trained on what survives
    would follow whichever mode came through most often. The draw is seeded, so
    the same corpus and settings give the same rows.
    """
    ratios = spec.mode_ratios
    if not ratios:
        return list(rows)
    pools: dict[str, list[dict]] = {mode: [] for mode in ratios}
    for row in rows:
        mode = mode_of(row)
        if mode in pools:
            pools[mode].append(row)
    targets = allocate_counts(ratios, {m: len(p) for m, p in pools.items()}, spec.max_rows)
    rng = random.Random(spec.seed)
    drawn: list[dict] = []
    for mode, count in targets.items():
        pool = pools[mode]
        count = min(count, len(pool))
        rng.shuffle(pool)
        drawn.extend(pool[:count])
    return drawn


def split_train_val_test(
    rows: Sequence[dict], spec: CorpusSpec
) -> tuple[list[dict], list[dict], list[dict]]:
    """Divide the corpus, deterministically given the seed.

    Each held-out part gets at least fifty rows once there are a hundred to
    divide, so a small build still leaves something to evaluate on, and never
    more than a fifth, so a small build still leaves something to train on.
    """
    rng = random.Random(spec.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    n = len(rows)
    floor = 50 if n >= 100 else 1
    n_val = min(max(floor, int(n * spec.val_fraction)), n // 5)
    n_test = min(max(floor, int(n * spec.test_fraction)), n // 5)
    test_ids = set(order[:n_test])
    val_ids = set(order[n_test : n_test + n_val])
    train, val, test = [], [], []
    for index, row in enumerate(rows):
        if index in test_ids:
            test.append(row)
        elif index in val_ids:
            val.append(row)
        else:
            train.append(row)
    return train, val, test
