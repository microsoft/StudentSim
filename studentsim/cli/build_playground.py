"""``studentsim-build-playground`` — the positions the tutor practises on.

Episodes and the reward table are written in one pass, from the same positions,
so the prompt the tutor sees and the reward it is scored by cannot disagree
about what the position is.

A position is usable only when the cache scores every legal move at it. Without
that, a student who plays an uncached move is scored as if the move were
illegal, which is a penalty the tutor cannot influence. Positions that fall
short are counted and dropped, and the count is printed, because a cache built
against a top-k list rather than the full move set loses most of them.

Input is one JSON object per line, each with a position, the colour the student
had and the move they played. Candidate scores come from the object if it
carries them, and otherwise from the Stockfish cache that
``studentsim-precompute-stockfish`` fills.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from studentsim.tutor_rl.playground import (
    PlaygroundSpec,
    Position,
    build_playground,
    build_reward_table,
    covers_every_legal_move,
    write_jsonl,
)


def legal_moves(fen: str) -> list[str]:
    import chess

    return sorted(move.uci() for move in chess.Board(fen).legal_moves)


def read_blunders(path: Path, lookup=None) -> tuple[list[Position], int]:
    """Read the wrong-move records, scoring each position's legal moves.

    Returns the usable positions and how many were dropped for a cache that
    does not reach every legal move.
    """
    positions: list[Position] = []
    short = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            fen = row["fen"]
            if "candidates" in row:
                candidates = {move: int(cp) for move, cp in row["candidates"].items()}
            elif lookup is not None:
                candidates = {}
                for move in legal_moves(fen):
                    centipawns = lookup.get(fen=fen, move_uci=move)
                    if centipawns is not None:
                        candidates[move] = int(centipawns)
            else:
                raise ValueError(
                    f"{path}: a record carries no candidates and no --stockfish-cache "
                    "was given to score them"
                )
            position = Position(
                fen=fen,
                player_color=row.get("player_color", "white"),
                wrong_move=row["wrong_move"],
                candidates=candidates,
            )
            if not covers_every_legal_move(position, legal_moves(fen)):
                short += 1
                continue
            positions.append(position)
    return positions, short


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studentsim-build-playground",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--blunders", required=True, type=Path,
                        help="JSONL of {fen, player_color, wrong_move[, candidates]}.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Directory for train/val episodes and the reward table.")
    parser.add_argument("--stockfish-cache", type=Path,
                        help="SQLite cache to score legal moves from.")
    parser.add_argument("--cp-gap", type=int, default=PlaygroundSpec.cp_gap,
                        help="How much worse the played move must be to be worth an episode.")
    parser.add_argument("--top-n-candidates", type=int,
                        default=PlaygroundSpec.top_n_candidates)
    parser.add_argument("--mode-tag-fraction", type=float,
                        default=PlaygroundSpec.mode_tag_fraction,
                        help="Share of episodes whose prompt names a mode.")
    parser.add_argument("--validation-fraction", type=float,
                        default=PlaygroundSpec.validation_fraction)
    parser.add_argument("--seed", type=int, default=PlaygroundSpec.seed)
    parser.add_argument("--limit", type=int, default=0,
                        help="Use only this many positions; 0 for all.")
    parser.add_argument("--mate-clip-cp", type=int, default=1500)
    parser.add_argument("--scale-cp", type=int, default=500)
    parser.add_argument("--illegal-reward", type=float, default=-1.0)
    args = parser.parse_args(argv)

    lookup = None
    if args.stockfish_cache:
        from studentsim.tutor_rl.stockfish_cache import SqliteStockfishLookup

        lookup = SqliteStockfishLookup(args.stockfish_cache)

    positions, short = read_blunders(args.blunders, lookup)
    print(f"{len(positions):,} positions usable, {short:,} dropped for a cache that "
          f"does not reach every legal move", flush=True)
    if not positions:
        parser.error("no usable positions; is the Stockfish cache filled for these?")

    spec = PlaygroundSpec(
        cp_gap=args.cp_gap,
        top_n_candidates=args.top_n_candidates,
        mode_tag_fraction=args.mode_tag_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        limit=args.limit or None,
    )
    train, val = build_playground(positions, spec)
    table = build_reward_table(
        positions,
        mate_clip_cp=args.mate_clip_cp,
        scale_cp=args.scale_cp,
        illegal_reward=args.illegal_reward,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl([row.__dict__ for row in train], args.out / "train.jsonl")
    write_jsonl([row.__dict__ for row in val], args.out / "val.jsonl")
    write_jsonl(table, args.out / "reward_table.jsonl")
    print(f"  train: {len(train):,} episodes", flush=True)
    print(f"  val:   {len(val):,} episodes", flush=True)
    print(f"  reward table: {len(table):,} rows", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
