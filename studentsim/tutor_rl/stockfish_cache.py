"""Stockfish-evaluation cache.

The training loop calls the reward function once per rollout, and each reward call
looks up two ``(fen, move) -> centipawn`` values. Running Stockfish live at
each call is too slow (an engine evaluation costs tens to hundreds of
milliseconds), so the cache is filled offline by
``studentsim-precompute-stockfish`` and served from disk during training.

This module defines:

- :class:`StockfishLookup` Protocol with one method ``get(fen, move)``.
- :class:`InMemoryStockfishLookup`: dict-backed, and what a caller gets when
  no cache file is named.
- :class:`SqliteStockfishLookup`: production SQLite-backed.
- :class:`LiveStockfishLookup`: on-demand subprocess-backed (cache misses).
- :class:`LayeredStockfishLookup`: tries cache first, falls back to a live
  source (or raises).

The SQLite schema is a single table keyed on ``(fen, move)``; the value is the
centipawn integer from the perspective of the player who is to move at ``fen``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class StockfishLookup(Protocol):
    """Per-(FEN, move) centipawn lookup."""

    def get(self, *, fen: str, move_uci: str) -> int | None:
        """Return the cached centipawn evaluation, or ``None`` if absent."""
        ...


# --- In-memory lookup ---


class InMemoryStockfishLookup:
    """Dict-backed lookup, and what a caller gets when no cache file is named."""

    def __init__(self, data: Mapping[tuple[str, str], int] | None = None) -> None:
        self._data: dict[tuple[str, str], int] = dict(data or {})

    def get(self, *, fen: str, move_uci: str) -> int | None:
        return self._data.get((fen, move_uci))

    def set(self, *, fen: str, move_uci: str, cp: int) -> None:
        """Add one evaluation, replacing any already held for that move."""
        self._data[(fen, move_uci)] = cp


# --- SQLite lookup ---


_SCHEMA = """
CREATE TABLE IF NOT EXISTS stockfish_cache (
    fen TEXT NOT NULL,
    move TEXT NOT NULL,
    cp INTEGER NOT NULL,
    PRIMARY KEY (fen, move)
);
"""


class SqliteStockfishLookup:
    """SQLite-backed lookup. Read-only access path for rollouts.

    Use :meth:`bulk_insert` from precompute scripts to populate. Concurrent
    Trainer workers can share one read-only handle.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, *, fen: str, move_uci: str) -> int | None:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT cp FROM stockfish_cache WHERE fen = ? AND move = ?",
                (fen, move_uci),
            )
            row = cur.fetchone()
            return int(row[0]) if row is not None else None

    def bulk_insert(self, rows: Iterator[tuple[str, str, int]]) -> int:
        """Insert (fen, move, cp) rows; on conflict, keep the existing value.

        Returns the number of rows actually inserted (excluding conflicts).
        """
        with self._connect() as conn:
            cur = conn.executemany(
                "INSERT OR IGNORE INTO stockfish_cache(fen, move, cp) VALUES (?, ?, ?)",
                rows,
            )
            return cur.rowcount


# --- Live (subprocess) lookup ---


class LiveStockfishLookup:
    """Evaluate (fen, move) on demand via a :class:`StockfishEngine` subprocess.

    Significantly slower than the SQLite cache (~50-200ms per eval at depth 15).
    Used as a fallback by :class:`LayeredStockfishLookup` for cache misses, or
    directly when no cache has been filled.
    """

    def __init__(
        self,
        *,
        depth: int = 15,
        binary: str | None = None,
    ) -> None:
        # Lazy import so importing studentsim.tutor_rl does not require Stockfish.
        from studentsim.domains.chess.stockfish import StockfishEngine

        self._engine = StockfishEngine(depth=depth, binary=binary)
        self._engine._open()  # type: ignore[attr-defined]

    def get(self, *, fen: str, move_uci: str) -> int | None:
        try:
            return self._engine.evaluate_move_cp(fen=fen, move_uci=move_uci)
        except ValueError:
            # Illegal move; the caller will apply the illegal-move penalty.
            return None

    def close(self) -> None:
        self._engine.close()


# --- Layered lookup ---


class LayeredStockfishLookup:
    """Try ``primary`` first; fall back to ``fallback`` on miss.

    The intended setup is ``primary = SqliteStockfishLookup`` and
    ``fallback = LiveStockfishLookup``; this gives a fast hot path (cache hit)
    and a correctness guarantee for cache misses.
    """

    def __init__(
        self,
        *,
        primary: StockfishLookup,
        fallback: StockfishLookup | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    def get(self, *, fen: str, move_uci: str) -> int | None:
        cp = self._primary.get(fen=fen, move_uci=move_uci)
        if cp is not None:
            return cp
        if self._fallback is not None:
            return self._fallback.get(fen=fen, move_uci=move_uci)
        return None
