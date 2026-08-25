"""``studentsim-judge-guidance`` — label tutor messages by their false claims.

Reads a file of generated guidance, asks a strong model which claims in each
message contradict the position, and writes the verdicts beside it. The output
is what :mod:`studentsim.tutor_rl.perception_dataset` reads as its judged
source, and the two files are paired by index, so the verdict list comes out the
same length as the message list even where a message could not be judged.

The engine evaluations quoted to the judge come from the Stockfish cache. A
position not in it is judged without them, which the prompt says outright rather
than passing off a gap as a table; supply ``--live-stockfish`` to fill the gaps
at the cost of an engine call per legal move.

Calls cost money. ``--limit`` bounds how many messages are judged, and a re-run
against an existing output re-asks only about the messages whose verdict could
not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studentsim.core.llm import Message
from studentsim.tutor_rl.judge import judge_messages, summarise

SIDES = ("pre", "post")
MAX_TOKENS = 4096


def _sender(client, concurrency: int):
    """Send a batch of prompts, keeping the replies in the order asked."""

    def send(prompts: list[list[dict]]) -> list[str | None]:
        def one(prompt: list[dict]) -> str | None:
            messages = [Message(role=turn["role"], content=turn["content"]) for turn in prompt]
            try:
                return client.complete(messages, max_tokens=MAX_TOKENS, temperature=0.0).text
            except Exception as error:
                # Reported and turned into an unreadable verdict rather than
                # raised: the caller re-asks about those, and one call that
                # timed out under rate limiting should not end the pass.
                print(f"  call failed: {type(error).__name__}: {error}", flush=True)
                return None

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(one, prompts))

    return send


def _lookup(cache: Path | None, live: bool):
    from studentsim.tutor_rl.stockfish_cache import (
        InMemoryStockfishLookup,
        LayeredStockfishLookup,
        LiveStockfishLookup,
        SqliteStockfishLookup,
    )

    primary = SqliteStockfishLookup(cache) if cache else InMemoryStockfishLookup()
    if not live:
        return primary
    return LayeredStockfishLookup(primary=primary, fallback=LiveStockfishLookup())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studentsim-judge-guidance",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--generations", required=True, type=Path,
                        help="JSON with fens, wrongs, answers_pre and answers_post.")
    parser.add_argument("--out", required=True, type=Path, help="Where the verdicts go.")
    parser.add_argument("--model", default="gpt-5.4", help="Deployment name of the judge.")
    parser.add_argument("--stockfish-cache", type=Path,
                        help="SQLite cache of evaluations to quote to the judge.")
    parser.add_argument("--live-stockfish", action="store_true",
                        help="Evaluate positions the cache misses, an engine call per move.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Judge only this many messages per side; 0 for all.")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Re-asks per side about messages whose verdict would not parse.")
    parser.add_argument("--sides", default="pre,post",
                        help="Which message lists to judge, comma separated.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any verdicts already in --out.")
    args = parser.parse_args(argv)

    sides = [side.strip() for side in args.sides.split(",") if side.strip()]
    unknown = [side for side in sides if side not in SIDES]
    if unknown:
        parser.error(f"unknown side(s) {unknown}; choose from {list(SIDES)}")

    generations = json.loads(args.generations.read_text())
    total = len(generations["fens"])
    count = min(args.limit, total) if args.limit else total
    print(f"{args.generations.name}: {total} messages, judging {count} per side", flush=True)

    previous: dict = {}
    if args.out.exists() and not args.no_resume:
        previous = json.loads(args.out.read_text())
        print(f"resuming from {args.out}", flush=True)

    from studentsim.core.llm import open_client

    send = _sender(open_client(args.model), args.concurrency)
    lookup = _lookup(args.stockfish_cache, args.live_stockfish)

    results: dict = {}
    summary: dict = {
        "generations": str(args.generations),
        "config": generations.get("config", {}),
        "n_judged": count,
        "judge_model": args.model,
    }
    for side in sides:
        items = [
            {"fen": generations["fens"][index],
             "wrong_move": generations["wrongs"][index],
             "text": generations[f"answers_{side}"][index]}
            for index in range(count)
        ]
        prior = previous.get(f"{side}_results")
        if prior is not None and len(prior) != count:
            print(f"  {side}: {len(prior)} prior verdicts for {count} messages, starting over",
                  flush=True)
            prior = None
        verdicts = judge_messages(
            items, send, lookup,
            max_retries=args.max_retries, prior=prior, label=side,
            on_progress=lambda line: print(f"  {line}", flush=True),
        )
        results[f"{side}_results"] = verdicts
        report = summarise(verdicts)
        summary[side] = report
        print(f"  {side}: {report['n_valid']}/{report['n_total']} judged, "
              f"{report['frac_clean']:.1%} clean, "
              f"{report['errors_per_instance_mean']:.2f} errors each", flush=True)

    if "pre" in summary and "post" in summary:
        summary["delta"] = {
            "frac_clean": summary["post"]["frac_clean"] - summary["pre"]["frac_clean"],
            "errors_per_instance": (summary["post"]["errors_per_instance_mean"]
                                    - summary["pre"]["errors_per_instance_mean"]),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, **results}, indent=2,
                                   ensure_ascii=False))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
