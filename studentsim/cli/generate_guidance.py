"""``studentsim-generate-guidance`` — write tutor guidance for a set of positions.

Every position is explained once in each of the four modes, by a model that is
shown the board as a picture as well as in text, so that what it writes can
afterwards be checked against the position it was looking at.

The input is one JSON object per line, carrying the position and whatever the
templates ask for by name. Which fields those are is a property of the
templates, not of this command, so it reports what a template wants and what
the file supplies rather than assuming they agree.

Output is one file per mode. A position whose answer could not be read is
counted and skipped: one unusable generation should not cost the rest of a run
that is paying per call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from studentsim.core.llm import Message
from studentsim.tutor_rl.guidance_generation import (
    GuidanceRequest,
    generate,
    load_templates,
    write_guidance,
)
from studentsim.tutor_rl.sft_corpus import MODES

MAX_TOKENS = 1024


def read_requests(path: Path, limit: int = 0) -> list[GuidanceRequest]:
    """Read the positions to explain, in file order."""
    requests = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            requests.append(
                GuidanceRequest(
                    position_id=str(row.get("position_id", len(requests))),
                    fen=row["fen"],
                    fields=row.get("fields", row),
                    wrong_move=row.get("wrong_move", row.get("WRONG_MOVE", "")),
                    board_image_path=row.get("board_image_path"),
                )
            )
            if limit and len(requests) >= limit:
                break
    return requests


def missing_fields(templates: dict, requests: list[GuidanceRequest]) -> dict[str, set[str]]:
    """What each template asks for that the first request does not supply.

    Checked before any call is made, because the alternative is discovering it
    once per position at the price of a generation each.
    """
    if not requests:
        return {}
    supplied = set(requests[0].fields)
    return {
        mode: template.placeholders() - supplied
        for mode, template in templates.items()
        if template.placeholders() - supplied
    }


def _image_url_for(request: GuidanceRequest) -> str | None:
    """The board as a data URL, drawn if the request did not bring a picture."""
    import base64

    from studentsim.tutor_rl.sft_corpus import render_board

    if request.board_image_path:
        raw = Path(request.board_image_path).read_bytes()
    else:
        raw = render_board(request.fen, request.wrong_move or None)
    return f"data:image/png;base64,{base64.b64encode(raw).decode()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studentsim-generate-guidance",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--positions", required=True, type=Path,
                        help="JSONL of positions, one object per line.")
    parser.add_argument("--templates", required=True, type=Path,
                        help="Directory holding <mode>_system.txt and <mode>_user.txt.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Directory to write one <mode>.jsonl into.")
    parser.add_argument("--model", default="gpt-5.4", help="Deployment name of the writer.")
    parser.add_argument("--modes", default=",".join(MODES),
                        help="Which modes to write, comma separated.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Explain only this many positions; 0 for all.")
    parser.add_argument("--no-board-image", action="store_true",
                        help="Send the position as text only.")
    args = parser.parse_args(argv)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        parser.error(f"unknown mode(s) {unknown}; choose from {list(MODES)}")

    templates = load_templates(args.templates, modes)
    requests = read_requests(args.positions, args.limit)
    if not requests:
        parser.error(f"no positions in {args.positions}")

    gaps = missing_fields(templates, requests)
    if gaps:
        parser.error(
            "the positions do not supply every field the templates ask for: "
            + "; ".join(f"{mode} wants {sorted(names)}" for mode, names in gaps.items())
        )

    print(f"{len(requests):,} positions x {len(modes)} modes = "
          f"{len(requests) * len(modes):,} generations", flush=True)

    from studentsim.core.llm import open_client

    client = open_client(args.model)

    def complete(messages: list[dict]) -> str:
        turns = [Message(role=turn["role"], content=turn["content"]) for turn in messages]
        return client.complete(turns, max_tokens=MAX_TOKENS, temperature=0.0).text

    rows = list(generate(
        requests, templates, complete=complete, modes=modes,
        image_url_for=None if args.no_board_image else _image_url_for,
    ))
    unreadable = [row for row in rows if "error" in row]
    counts = write_guidance(rows, args.out)

    for mode in modes:
        print(f"  {mode}: {counts.get(mode, 0):,}", flush=True)
    if unreadable:
        print(f"{len(unreadable):,} answers could not be read; first: "
              f"{unreadable[0]['error']}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
