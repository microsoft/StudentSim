"""``studentsim-build-corpus`` — turn generated guidance into the tutor's SFT set.

Four stages, in this order, because each depends on the one before it:

1. **Filter.** A row is dropped for naming a move the position cannot reach,
   for putting a piece where there is none, or for citing where its information
   came from.
2. **Balance.** The modes do not survive in equal numbers, so each contributes
   its share of the draw, held to what the scarcest one can supply.
3. **Cap.** ``--max-rows`` bounds the total, applied inside the balance so the
   shares stay even.
4. **Split.** Train, validation and test, deterministic given the seed.

What was dropped and why is printed, so a build that loses most of one mode is
visible before training rather than after.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from studentsim.tutor_rl.sft_corpus import (
    MODES,
    CorpusSpec,
    balance_modes,
    build_corpus,
    split_train_val_test,
)

BOARD_DIRNAME = "boards"


def read_guidance(directory: Path, modes: list[str]) -> list[dict]:
    """Read every mode's file, tagging each row with the mode it came from.

    The mode is taken from the file rather than trusted from the row: it is
    what the balancing stage groups by, and a row that lost its tag would be
    dropped from the draw without saying so.
    """
    rows = []
    for mode in modes:
        path = directory / f"{mode}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append({**json.loads(line), "mode": mode})
    return rows


def render_boards(rows: list[dict], directory: Path) -> dict[str, str]:
    """Draw each position once, and return where each row's picture went."""
    from studentsim.tutor_rl.sft_corpus import render_board

    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for row in rows:
        key = f"{row['fen']}|{row.get('wrong_move', '')}"
        if key in paths:
            continue
        target = directory / f"board_{len(paths):06d}.png"
        target.write_bytes(render_board(row["fen"], row.get("wrong_move")))
        paths[key] = str(target)
    return paths


def write_split(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studentsim-build-corpus",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--guidance", required=True, type=Path,
                        help="Directory of <mode>.jsonl written by generate-guidance.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Directory to write train/val/test .jsonl into.")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--mode-ratios", default="",
                        help="Shares as mode=ratio pairs; the modes contribute "
                             "equally when this is not given.")
    parser.add_argument("--seed", type=int, default=CorpusSpec.seed)
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Cap on the balanced draw; 0 for no cap.")
    parser.add_argument("--mode-tag-dropout", type=float,
                        default=CorpusSpec.mode_tag_dropout,
                        help="Share of rows whose prompt does not name the mode.")
    parser.add_argument("--max-illegal-moves", type=int, default=CorpusSpec.max_illegal_moves)
    parser.add_argument("--val-fraction", type=float, default=CorpusSpec.val_fraction)
    parser.add_argument("--test-fraction", type=float, default=CorpusSpec.test_fraction)
    parser.add_argument("--no-piece-filter", action="store_true",
                        help="Keep rows that put a piece where there is none.")
    parser.add_argument("--no-leak-filter", action="store_true",
                        help="Keep rows that cite where their information came from.")
    parser.add_argument("--with-boards", action="store_true",
                        help="Render a board per position and carry it in the rows.")
    args = parser.parse_args(argv)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        parser.error(f"unknown mode(s) {unknown}; choose from {list(MODES)}")

    # An empty ratio map means the library does not balance at all, and the
    # cap rides on the balance, so leaving it unset would quietly give back
    # everything that survived filtering under a heading saying otherwise.
    if args.mode_ratios:
        pairs = [pair.split("=") for pair in args.mode_ratios.split(",") if pair]
        ratios = {name.strip(): float(share) for name, share in pairs}
    else:
        ratios = {mode: 1.0 / len(modes) for mode in modes}

    spec = CorpusSpec(
        mode_ratios=ratios,
        max_illegal_moves=args.max_illegal_moves,
        filter_piece_state=not args.no_piece_filter,
        filter_instruction_leak=not args.no_leak_filter,
        mode_tag_dropout=args.mode_tag_dropout,
        seed=args.seed,
        max_rows=args.max_rows or None,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    rows = read_guidance(args.guidance, modes)
    if not rows:
        parser.error(f"no guidance under {args.guidance}")
    print(f"read {len(rows):,} rows: {dict(Counter(r['mode'] for r in rows))}", flush=True)

    image_paths = render_boards(rows, args.out / BOARD_DIRNAME) if args.with_boards else {}

    def image_path_for(row: dict) -> str | None:
        return image_paths.get(f"{row['fen']}|{row.get('wrong_move', '')}")

    examples, rejections = build_corpus(rows, spec, image_path_for=image_path_for)
    print(f"kept {len(examples):,}, dropped {len(rejections):,}", flush=True)
    for reason, count in Counter(r.reason for r in rejections).most_common():
        print(f"  {count:,}  {reason}", flush=True)

    drawn = balance_modes(examples, spec)
    print(f"balanced to {len(drawn):,}: "
          f"{dict(Counter(row['mode'] for row in drawn))}", flush=True)

    train, val, test = split_train_val_test(drawn, spec)
    for name, part in (("train", train), ("val", val), ("test", test)):
        write_split(part, args.out / f"{name}.jsonl")
        print(f"  {name}: {len(part):,}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
