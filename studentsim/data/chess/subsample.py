"""Draw the per-student specialization subset from a player's records.

Stage 2 trains on a stratified 1,000 records per player: 800 single-turn
positions plus 50 two-turn records for each of the four guidance modes. The
released per-player files are already this draw, so this is for anyone
working from a pool of their own. Keeping the guidance modes at equal counts
is what makes the per-mode responsiveness breakdown comparable, and it is why
the draw is stratified rather than uniform.

The draw is a function of the run seed and the student's position in the
roster, so a given seed reproduces the same 1,000 records without shipping
them. Two details must not be tidied away: the modes are drawn in sorted
order, and the whole selection is shuffled afterwards. Both decide the order
the generator is consumed in, and changing either gives a different thousand
records for the same seed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from studentsim.core.seeds import DATA_SAMPLER_SEED

SINGLE_TURN: Final = 800
PER_MODE: Final = 50
MODE_FIELD: Final = "instruction_type"
DEFAULT_SEED: Final = DATA_SAMPLER_SEED


def subset_rng(seed: int, tag: str) -> random.Random:
    """The generator for one student's draw.

    Mixing the tag into the seed keeps one student's draw independent of the
    others while staying reproducible from the run seed alone.
    """
    return random.Random(seed ^ zlib.crc32(tag.encode()))


def draw(
    records: Sequence[dict],
    *,
    seed: int = DEFAULT_SEED,
    tag: str,
    single_turn: int = SINGLE_TURN,
    per_mode: int = PER_MODE,
) -> list[dict]:
    """Take the stratified subset from one player's records."""
    single = [r for r in records if len(r["messages"]) == 2]
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if len(record["messages"]) >= 4:
            by_mode[record.get(MODE_FIELD, "?")].append(record)

    rng = subset_rng(seed, tag)
    picked = rng.sample(single, single_turn)
    for mode in sorted(by_mode):
        picked += rng.sample(by_mode[mode], per_mode)
    rng.shuffle(picked)
    return picked


def read_records(path: str | Path) -> list[dict]:
    """Read one player's file.

    Iterated line by line rather than split at once: a record's own text can
    carry characters that a bulk split treats as line breaks, which would cut a
    record in half.
    """
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_records(records: Sequence[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m studentsim.data.chess.subsample",
        description="Draw each player's stratified Stage-2 subset from a pool of your own.",
    )
    parser.add_argument("--players", type=Path, required=True,
                        help="Directory holding one file per player.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--single-turn", type=int, default=SINGLE_TURN)
    parser.add_argument("--per-mode", type=int, default=PER_MODE)
    args = parser.parse_args(argv)

    sources = sorted(args.players.glob(args.pattern))
    if not sources:
        print(f"no files matching {args.pattern} under {args.players}", file=sys.stderr)
        return 1
    for index, source in enumerate(sources):
        tag = f"p{index:02d}"
        picked = draw(
            read_records(source), seed=args.seed, tag=tag,
            single_turn=args.single_turn, per_mode=args.per_mode,
        )
        write_records(picked, args.out / f"{tag}.jsonl")
        print(f"  {tag}: {len(picked)} records from {source.name}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
