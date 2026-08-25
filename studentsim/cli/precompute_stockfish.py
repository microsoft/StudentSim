"""``studentsim-precompute-stockfish`` — fill the cache the tutor RL reward reads.

Every legal move at every position is evaluated, because that is what the
playground demands of the cache: a position is usable only when the cache
scores everywhere the student could go, since a student who plays an uncached
move would otherwise be scored as if the move were illegal.

Usage::

    studentsim-precompute-stockfish \\
        --fens fens.txt \\
        --cache "$STUDENTSIM_DATA_DIR/tutor_rl/stockfish.db" \\
        --depth 15 \\
        --workers 8

``fens.txt`` is a newline-separated list of positions. This is a long job and
does not need a GPU; Stockfish runs on CPU. It is resumable: positions already
scored in the cache are skipped, so it can be interrupted and rerun.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from studentsim.tutor_rl.stockfish_cache import SqliteStockfishLookup


def read_fens(path: Path) -> list[str]:
    """The positions to score, in file order, without repeats."""
    seen: dict[str, None] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            seen.setdefault(stripped, None)
    return list(seen)


def legal_moves(fen: str) -> list[str]:
    import chess

    return sorted(move.uci() for move in chess.Board(fen).legal_moves)


def _evaluate_one(task: tuple[str, str, int, str]) -> tuple[str, str, int] | Exception:
    """Score one (position, move) pair in a worker process.

    A failure comes back rather than being swallowed. A position missing even
    one move makes that position unusable, so a silent drop here would show up
    much later as an unexplained shortfall in playground size.
    """
    from studentsim.domains.chess.stockfish import StockfishEngine

    fen, move, depth, binary = task
    try:
        with StockfishEngine(depth=depth, binary=binary) as engine:
            return (fen, move, engine.evaluate_move_cp(fen=fen, move_uci=move))
    except (ValueError, RuntimeError) as error:
        return error


def _tasks(fens: Sequence[str], cache: SqliteStockfishLookup, depth: int,
           binary: str) -> Iterator[tuple[str, str, int, str]]:
    for fen in fens:
        for move in legal_moves(fen):
            if cache.get(fen=fen, move_uci=move) is None:
                yield (fen, move, depth, binary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studentsim-precompute-stockfish",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fens", type=Path, required=True,
                        help="Newline-separated positions to score.")
    parser.add_argument("--cache", type=Path, required=True, help="Output SQLite path.")
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stockfish-bin", default=None,
                        help="Defaults to STUDENTSIM_STOCKFISH_BIN, then PATH.")
    args = parser.parse_args(argv)

    from studentsim.core.paths import stockfish_bin

    binary = args.stockfish_bin or stockfish_bin()
    fens = read_fens(args.fens)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache = SqliteStockfishLookup(args.cache)

    tasks = list(_tasks(fens, cache, args.depth, binary))
    total_moves = sum(len(legal_moves(fen)) for fen in fens)
    print(f"{len(fens):,} positions, {total_moves:,} legal moves, "
          f"{len(tasks):,} left to score", flush=True)

    pending: list[tuple[str, str, int]] = []
    failures: list[Exception] = []
    done = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_evaluate_one, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, Exception):
                failures.append(result)
            else:
                pending.append(result)
            done += 1
            if len(pending) >= 1000:
                cache.bulk_insert(iter(pending))
                pending = []
            if done % 5000 == 0:
                rate = done / max(time.time() - started, 1e-6)
                print(f"  {done:,}/{len(tasks):,}  ({rate:.0f}/s)", flush=True)

    if pending:
        cache.bulk_insert(iter(pending))
    print(f"wrote {args.cache}", flush=True)
    if failures:
        print(f"{len(failures):,} evaluations failed; the positions they belong to "
              f"will not be usable. First: {failures[0]}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
