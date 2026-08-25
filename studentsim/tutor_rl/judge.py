"""Asking a strong model which claims in a tutor message are false.

The perception head learns six ways a message can misdescribe the board, and
rules only read two of them well. The rest — a piece that is not on the board
at all, a move asserted to be legal, a capture of something that is not there —
are phrased too many ways for patterns to catch, so they are labelled by a model
reading the message against the position.

What makes that reading trustworthy is that the model is not asked to play
chess. Every fact it needs is handed to it: which piece stands on which square,
which moves are legal, and what the engine thinks each is worth. Its job is to
check the message against that table, and the prompt says so in as many words,
because a judge that reasons about the position instead of consulting it will
disagree with the position.

The verdicts are written back per message, and a message whose verdict failed to
parse is marked as such rather than recorded as clean. That distinction matters
downstream: :mod:`studentsim.tutor_rl.perception_dataset` drops those rows, and
reading them as clean would teach the head that a message nobody judged has
nothing wrong with it.
"""

from __future__ import annotations

import base64
import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Final

MOVES_PER_ROW: Final = 12
"""Legal moves are wrapped at this many per line to keep the list readable."""

BOARD_SIZE: Final = 480
ARROW_COLOR: Final = "#cc0000"

SYSTEM: Final = """You are a chess expert auditing a chess tutor's instructional message for FACTUAL errors. \
The tutor wrote a message to a student about a chess position the student played from. \
Your only job is to identify *factual* errors in the tutor's text — claims that contradict the actual chess position or the rules of chess.

Do NOT flag:
- Style choices, tone, didactic structure
- Strategic-judgement opinions ("this move is slightly weaker than that move" — opinion, not fact)
- Vague or general statements ("you should think about king safety")
- The tutor's choice of recommended move (even if you disagree, that's a judgement)

DO flag (count each occurrence as one error):
- Wrong piece on a square ("your bishop on e5" when e5 has a knight, or is empty, or has the opponent's piece)
- Wrong square for a piece ("the rook on h8" when no rook is on h8)
- Wrong color attribution ("White's queen" when it's Black's queen, or vice versa)
- Illegal move asserted as legal ("you can play e2e5" when e2-e5 is illegal in this position)
- Continuation move that's illegal in the resulting position
- Wrong claim about whose turn it is to move
- Wrong claim about a capture target ("capturing the pawn on d4" when d4 has no pawn or has a piece other than a pawn)
- Wrong identification of the student's actual move (e.g., calling it a knight move when it was a king move)
- Misstating what piece type the student moved
- Hallucinated piece that isn't on the board
- Mathematically/logically wrong claim about the position (e.g., "you are a pawn ahead" when material is equal)
- Wrong claim about checks/captures already on the board

A single instance can have multiple errors; count each one separately."""

USER_TEMPLATE: Final = """POSITION (FEN): {fen}
WHOSE TURN: {turn}
STUDENT'S MOVE (the mistake — see red arrow on the board image): {wrong}

PIECE-ON-SQUARE MAP (from the FEN):
{piece_map}

LEGAL MOVES IN THIS POSITION (UCI):
{legal_moves}

ENGINE EVALUATION OF EACH LEGAL MOVE (Stockfish depth=15, in centipawns; higher = better for the side to move):
{engine_eval}
(The student played `{wrong}` — engine eval = {wrong_cp}. Best move = `{best_move}` at eval = {best_cp}.)

The data above ARE GROUND TRUTH. When verifying tutor claims:
- A move is illegal IFF its UCI does NOT appear in the legal-moves list.
- A piece is on a square IFF the piece-on-square map says so. Anything not in the map → square is empty.
- The student's move (`{wrong}`) moved whatever piece the FROM-square map says. The board image shows this with a red arrow.
- Move strength claims should be checked against the engine eval list above (don't flag opinion-level disagreement, e.g. "this move is slightly weaker"; DO flag claims about absolute eval values that don't match).

TUTOR'S MESSAGE TO STUDENT:
\"\"\"
{tutor_text}
\"\"\"

Enumerate every factual error in the tutor's message. **Be conservative: only flag a claim if it actually contradicts the ground truth above.** Strategic-judgement opinions, didactic style, and vague advice are NOT errors. Output strict JSON only — no preamble, no markdown fences:
{{
  "errors": [
    {{"type": "<short category, e.g. 'wrong_piece' / 'illegal_move' / 'wrong_color' / 'wrong_square' / 'wrong_capture' / 'wrong_turn' / 'hallucinated_piece' / 'misnamed_move' / 'wrong_material' / 'wrong_eval'>",
     "description": "<what's wrong, in one sentence>",
     "quote": "<exact tutor text quote, ≤120 chars>"}},
    ...
  ],
  "count": <integer total errors>
}}

If no factual errors, return {{"errors": [], "count": 0}}."""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(raw: str | None) -> dict:
    """Read a judge's reply, or record why it could not be read.

    A model told to answer in bare JSON sometimes fences it, prefaces it, or
    stops mid-object. Each of those failures returns a verdict carrying
    ``_parse_error`` and a count of -1, so a caller can tell "nothing wrong with
    this message" from "no usable answer about this message". The two are not
    distinguishable by the error list alone, which is empty either way.
    """
    if raw is None:
        return {"_parse_error": "no_response", "errors": [], "count": -1}
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    match = _JSON_BLOCK.search(text)
    if not match:
        return {"_parse_error": "no_json_block", "errors": [], "count": -1, "raw": text[:300]}
    block = match.group(0)
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError as error:
        return {
            "_parse_error": f"json_decode:{error}", "errors": [], "count": -1,
            "raw": block[:300],
        }
    if not isinstance(parsed, dict):
        return {"_parse_error": "not_dict", "errors": [], "count": -1, "raw": block[:300]}
    errors = parsed.get("errors", [])
    if not isinstance(errors, list):
        errors = []
    count = parsed.get("count")
    if not isinstance(count, int):
        count = len(errors)
    return {"errors": errors, "count": int(count)}


def side_to_move(fen: str) -> str:
    parts = fen.split()
    if len(parts) >= 2 and parts[1] in ("w", "b"):
        return "white" if parts[1] == "w" else "black"
    return "?"


def piece_map(fen: str) -> str:
    """Every occupied square and what stands on it, one per line."""
    import chess

    try:
        board = chess.Board(fen)
    except ValueError as error:
        return f"(invalid FEN: {error})"
    lines = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            colour = "white" if piece.color == chess.WHITE else "black"
            lines.append(
                f"  {chess.square_name(square)} = {colour} {chess.piece_name(piece.piece_type)}"
            )
    return "\n".join(lines) if lines else "  (empty board)"


def legal_moves(fen: str) -> str:
    import chess

    try:
        board = chess.Board(fen)
    except ValueError as error:
        return f"(invalid FEN: {error})"
    moves = sorted(move.uci() for move in board.legal_moves)
    rows = [
        ", ".join(moves[start : start + MOVES_PER_ROW])
        for start in range(0, len(moves), MOVES_PER_ROW)
    ]
    return "\n".join("  " + row for row in rows) if rows else (
        "  (no legal moves — terminal position)"
    )


def move_evaluations(fen: str, lookup) -> dict[str, int]:
    """What the engine thinks each legal move is worth, in centipawns.

    Missing moves are left out rather than defaulted: the prompt says the table
    is ground truth, and a fabricated evaluation in it would be read as one.
    """
    import chess

    try:
        board = chess.Board(fen)
    except ValueError:
        return {}
    evaluations = {}
    for move in board.legal_moves:
        uci = move.uci()
        centipawns = lookup.get(fen=fen, move_uci=uci)
        if centipawns is not None:
            evaluations[uci] = int(centipawns)
    return evaluations


def ranked(evaluations: dict[str, int]) -> list[tuple[str, int]]:
    """Moves best first, ties broken by name.

    Two moves worth the same have no order of their own, so one is named. The
    alternative is to let them fall out in whatever order the evaluations were
    read in, which is a property of where the numbers came from rather than of
    the position, and would make the same board produce two different prompts
    depending on how it was scored.
    """
    return sorted(evaluations.items(), key=lambda item: (-item[1], item[0]))


def evaluation_block(evaluations: dict[str, int]) -> str:
    if not evaluations:
        return "  (no engine eval cached for this FEN)"
    lines = []
    for move, centipawns in ranked(evaluations):
        pawns = centipawns / 100.0
        sign = "+" if pawns >= 0 else ""
        lines.append(f"  {move}: {sign}{pawns:.2f}cp")
    return "\n".join(lines)


def board_png(fen: str, wrong_move: str, *, size: int = BOARD_SIZE) -> bytes | None:
    """The position from White's side, with the student's move drawn in red.

    Always White's side, unlike the board the tutor is shown: the judge is
    reading a text against a table, and a picture that turns around with the
    side to move is one more thing for it to get wrong. Whose turn it is is
    written out in the prompt.
    """
    import cairosvg
    import chess
    import chess.svg

    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(wrong_move)
    except ValueError:
        return None
    svg = chess.svg.board(
        board,
        arrows=[chess.svg.Arrow(move.from_square, move.to_square, color=ARROW_COLOR)],
        size=size,
        orientation=chess.WHITE,
    )
    return cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=size)


def build_prompt(fen: str, wrong_move: str, tutor_text: str, evaluations: dict[str, int],
                 *, with_image: bool = True) -> list[dict]:
    """The two messages the judge is sent, ground truth and all."""
    best_move, best_cp = "?", "?"
    if evaluations:
        # The head of the same ranking the prompt lists, so the move named as
        # best is the move at the top of the table beside it.
        best_move, best_centipawns = ranked(evaluations)[0]
        best_cp = f"{best_centipawns / 100.0:+.2f}cp"
    played = evaluations.get(wrong_move.lower())
    text = USER_TEMPLATE.format(
        fen=fen,
        turn=side_to_move(fen),
        wrong=wrong_move,
        piece_map=piece_map(fen),
        legal_moves=legal_moves(fen),
        engine_eval=evaluation_block(evaluations),
        wrong_cp=f"{played / 100.0:+.2f}cp" if played is not None else "?",
        best_move=best_move,
        best_cp=best_cp,
        tutor_text=tutor_text,
    )
    content: list[dict] = [{"type": "text", "text": text}]
    if with_image:
        png = board_png(fen, wrong_move)
        if png is not None:
            encoded = base64.b64encode(png).decode()
            content.append(
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]


def judge_messages(
    items: Sequence[dict],
    send: Callable[[list[list[dict]]], list[str | None]],
    lookup,
    *,
    max_retries: int = 3,
    prior: Sequence[dict | None] | None = None,
    on_progress: Callable[[str], None] | None = None,
    label: str = "",
) -> list[dict]:
    """Judge each message, retrying the ones that came back unreadable.

    Only the unreadable are retried, and only they cost anything: a failure here
    is almost always a request that timed out under rate limiting, not a model
    that cannot answer. ``prior`` carries verdicts from an earlier run so a
    resumed pass re-asks about those and leaves the rest alone.
    """
    verdicts: list[dict | None] = list(prior) if prior is not None else [None] * len(items)
    if len(verdicts) != len(items):
        raise ValueError(f"{len(verdicts)} prior verdicts for {len(items)} messages")
    pending = [
        index for index, verdict in enumerate(verdicts)
        if verdict is None or "_parse_error" in verdict
    ]
    for attempt in range(max_retries + 1):
        if not pending:
            break
        if on_progress:
            on_progress(f"[{label}] attempt {attempt}: judging {len(pending)} messages")
        prompts = [
            build_prompt(
                items[index]["fen"], items[index]["wrong_move"], items[index]["text"],
                move_evaluations(items[index]["fen"], lookup),
            )
            for index in pending
        ]
        replies = send(prompts)
        if len(replies) != len(prompts):
            raise ValueError(f"{len(replies)} replies for {len(prompts)} prompts")
        still_pending = []
        for reply, index in zip(replies, pending, strict=True):
            verdicts[index] = parse_verdict(reply)
            if "_parse_error" in verdicts[index]:
                still_pending.append(index)
        if on_progress:
            readable = sum(1 for v in verdicts if v is not None and "_parse_error" not in v)
            on_progress(f"[{label}] attempt {attempt}: {readable}/{len(items)} readable")
        pending = still_pending
    return [
        verdict if verdict is not None
        else {"_parse_error": "exhausted_retries", "errors": [], "count": -1}
        for verdict in verdicts
    ]


def summarise(verdicts: Sequence[dict]) -> dict:
    """How the pass went, by the counts that say whether it is usable.

    Field names are kept stable so two passes can be compared directly. Rates are
    over the readable
    verdicts rather than over everything, because a message nobody managed to
    judge is not evidence that it was clean.
    """
    readable = [verdict for verdict in verdicts if "_parse_error" not in verdict]
    clean = sum(1 for verdict in verdicts if verdict.get("count", 0) == 0)
    total_errors = sum(verdict.get("count", 0) for verdict in readable)
    histogram: Counter[str] = Counter()
    for verdict in readable:
        for error in verdict.get("errors", []):
            histogram[error.get("type", "?")] += 1
    denominator = max(1, len(readable))
    return {
        "n_total": len(verdicts),
        "n_parse_error": len(verdicts) - len(readable),
        "n_valid": len(readable),
        "n_clean": clean,
        "n_with_errors": len(readable) - clean,
        "errors_per_instance_mean": total_errors / denominator,
        "total_errors": total_errors,
        "frac_clean": clean / denominator,
        "frac_with_errors": (len(readable) - clean) / denominator,
        "type_hist": dict(histogram.most_common()),
    }
