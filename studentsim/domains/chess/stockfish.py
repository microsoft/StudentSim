"""Stockfish engine wrapper.

Thin wrapper around python-chess's ``chess.engine.SimpleEngine.popen_uci``
that exposes a single method, :meth:`evaluate_move_cp`. Training does not call
it: the reward and the playground both read a precomputed cache, and this is
what fills that cache and what answers a lookup it misses.

Heavy imports (``chess``, the Stockfish subprocess) are deferred to first use.
The binary location is resolved through :func:`studentsim.core.paths.stockfish_bin`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from studentsim.core.paths import stockfish_bin

if TYPE_CHECKING:
    import chess.engine  # noqa: F401

MATE_CP = 10_000
"""What mate is worth in centipawns: large, finite, and clipped by the reward."""


class StockfishEngine:
    """Wrap a single Stockfish process.

    Use as a context manager so the subprocess is torn down deterministically::

        with StockfishEngine(depth=15) as eng:
            cp = eng.evaluate_move_cp(fen="...", move_uci="e2e4")
    """

    def __init__(
        self,
        *,
        depth: int = 15,
        binary: str | None = None,
    ) -> None:
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self._depth = depth
        self._binary = binary or stockfish_bin()
        self._engine = None  # type: ignore[assignment]

    def __enter__(self) -> StockfishEngine:
        self._open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _open(self) -> None:
        import chess.engine

        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self._binary)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def evaluate_move_cp(self, *, fen: str, move_uci: str) -> int:
        """What playing ``move_uci`` from ``fen`` is worth, in centipawns.

        Scored from the perspective of the player to move at ``fen``: positive
        means good for them. Mate is :data:`MATE_CP`, large but finite, and
        callers clip it.

        The engine searches from ``fen`` restricted to this move, rather than
        being handed the position the move leads to. The two are not the same
        question: asking about the position afterwards asks whose turn it is
        there, and a move that ends the game leaves no position to ask about.

        Raises :class:`ValueError` if ``move_uci`` is illegal at ``fen``.
        """
        import chess
        import chess.engine

        board = chess.Board(fen)
        try:
            move = chess.Move.from_uci(move_uci)
        except (chess.InvalidMoveError, ValueError) as e:
            raise ValueError(f"invalid UCI move {move_uci!r}: {e}") from e
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {move_uci!r} on FEN {fen!r}")

        # A move that ends the game is scored without an engine call. There is
        # nothing left to search, and an engine asked about a finished position
        # answers about the side that has just been mated.
        after = board.copy(stack=False)
        after.push(move)
        if after.is_checkmate():
            return MATE_CP
        if after.is_stalemate() or after.is_insufficient_material():
            return 0

        self._open()
        info = self._engine.analyse(  # type: ignore[union-attr]
            board, chess.engine.Limit(depth=self._depth), root_moves=[move]
        )
        return _cp_from_score(info["score"].pov(board.turn))


def _cp_from_score(score) -> int:
    """Read one side's score as centipawns.

    Every branch here is a case the engine really produces, and none of them
    falls back on a number: a score that cannot be read is not a drawn
    position, and reporting it as one would put a plausible evaluation into the
    reward table.
    """
    if score.is_mate():
        mate_in = score.mate()
        if mate_in is None:
            raise ValueError(f"mate score with no distance to mate: {score!r}")
        # Mate in zero is mate already delivered, which is winning, not losing.
        return MATE_CP if mate_in >= 0 else -MATE_CP
    centipawns = score.score()
    if centipawns is None:
        raise ValueError(f"score is neither a mate nor a number: {score!r}")
    return int(centipawns)


def is_legal_move(*, fen: str, move_uci: str) -> bool:
    """Pure-Python move-legality check (no engine subprocess).

    Used by the reward function's illegal-move detection so the costly engine
    eval is avoided when the model emits an illegal move.
    """
    import chess

    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
    except (chess.InvalidMoveError, ValueError):
        return False
    return move in board.legal_moves
