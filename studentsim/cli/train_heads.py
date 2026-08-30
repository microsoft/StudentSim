"""``studentsim-train-heads``: fit the two heads the tutor RL reward reads.

The heads sit on the frozen student simulator, so the run is three steps: label
each tutor message against its position, push those through the backbone once
to pool a feature per example, then fit both linear heads on the pooled
features.

Which messages get labelled matters more than it looks. Generated guidance is
filtered before it becomes SFT data, and the filter drops rows for naming
impossible moves and misplacing pieces — which is to say it drops precisely
what the perception head exists to recognise. Labelling the filtered corpus
leaves several error types with almost no positive examples, so the
labels are taken from guidance as generated, ahead of the filter.

The per-class F1 the run reports is what weights the perception gate, so the
metrics file it writes is an input to the reward, not just a record of how
training went.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from studentsim.tutor_rl.heads import LabelledExample, calibrate, read_examples, train_heads
from studentsim.tutor_rl.perception_labels import label

# The move sits on its own line inside the section, under the colour.
_STUDENT_MOVE = re.compile(r"^\s*Move:\s*([a-h][1-8][a-h][1-8][qrbn]?)",
                           re.IGNORECASE | re.MULTILINE)


def read_corpus(path: Path, *, limit: int | None = None) -> Iterator[dict]:
    """Read the corpus rows that carry everything a labelled example needs.

    A quarter of the corpus has its mode tag stripped so the tutor learns to
    write without one. Those rows cannot train the style head, so they are
    skipped here; the perception head is trained on the same set to keep one
    feature pass covering both.
    """
    kept = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # A row whose tag was dropped still records the mode it was written
            # in, and that is the style label; what it lacks is the tag in the
            # prompt, which the head never sees.
            if not row.get("mode"):
                continue
            messages = row.get("messages", [])
            if len(messages) < 2:
                continue
            move = _STUDENT_MOVE.search(messages[0].get("content", ""))
            if not move:
                continue
            yield {
                "fen": row["fen"],
                "wrong_move": move.group(1).lower(),
                "instruction_uci": messages[1].get("content", ""),
                "mode": row["mode"],
            }
            kept += 1
            if limit is not None and kept >= limit:
                return


def build_examples(rows: Sequence[dict]) -> list[LabelledExample]:
    """Label each message against its position."""
    from studentsim.tutor_rl.multihead import STYLE_LABELS

    examples = []
    for row in rows:
        labels = label(row["fen"], row["instruction_uci"])
        style = [1.0 if row["mode"] == name else 0.0 for name in STYLE_LABELS]
        if not any(style):
            continue
        examples.append(
            LabelledExample(
                fen=row["fen"],
                wrong_move=row["wrong_move"],
                tutor_text=row["instruction_uci"],
                perception=labels.vector(),
                style=style,
            )
        )
    return examples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studentsim-train-heads",
        description="Label a tutor corpus and fit the perception and style heads.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--labels",
        type=Path,
        help="Labelled set written by the corpus builder, taken before its filters.",
    )
    source.add_argument(
        "--judged",
        type=Path,
        nargs=2,
        action="append",
        metavar=("GENERATIONS", "JUDGEMENTS"),
        help="A generations file and the judgement of it, repeatable. Assembles "
        "the set from judgements, rule-labelled bulk and clean negatives.",
    )
    source.add_argument(
        "--corpus",
        type=Path,
        help="Tutor SFT corpus jsonl, labelled here. Already filtered, so most "
        "error types will be scarce.",
    )
    parser.add_argument("--simulator", type=Path, required=True, help="Student simulator.")
    parser.add_argument("--adapter", type=Path, default=None, help="Stage-1 LoRA adapter.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the heads.")
    parser.add_argument("--limit", type=int, default=20000, help="Examples to label.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Write the labelled set and stop, without touching a GPU.",
    )
    args = parser.parse_args(argv)

    if args.judged:
        from studentsim.tutor_rl.perception_dataset import build

        rows = build([tuple(pair) for pair in args.judged])
        examples = [
            LabelledExample(
                fen=row["fen"], wrong_move=row["wrong_move"], tutor_text=row["tutor_text"],
                perception=row["perception"], style=[0.0] * 4, weight=row["weight"],
            )
            for row in rows
        ][: args.limit]
        sources = {row["label_source"] for row in rows}
        print(f"assembled {len(rows):,} rows from {sorted(sources)}", flush=True)
        # These rows carry no style, and a style target of nothing contributes
        # nothing to the loss, so this trains the perception head alone. The
        # style head needs the corpus, where the mode a row was written in is
        # the label.
        print("style labels absent; training the perception head only", flush=True)
    elif args.labels:
        examples = read_examples(args.labels)[: args.limit]
    else:
        examples = build_examples(list(read_corpus(args.corpus, limit=args.limit)))
    if not examples:
        print("no labelled examples; is the corpus tagged with a mode?", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    counts = [
        sum(1 for e in examples if e.perception[i] > 0.5) for i in range(len(examples[0].perception))
    ]
    print(f"labelled {len(examples):,} examples; positives per error type: {counts}", flush=True)
    if args.labels_only:
        with (args.out / "labels.jsonl").open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(
                    json.dumps(
                        {
                            "fen": example.fen,
                            "wrong_move": example.wrong_move,
                            "tutor_text": example.tutor_text,
                            "perception": list(example.perception),
                            "style": list(example.style),
                        }
                    )
                    + "\n"
                )
        return 0

    if args.adapter is None:
        print("--adapter is needed to pool features from the simulator", file=sys.stderr)
        return 1

    import torch

    from studentsim.tutor_rl.heads import pooled_features
    from studentsim.tutor_rl.multihead import MultiHeadSimulator

    simulator = MultiHeadSimulator(
        base_model=str(args.simulator), adapter_path=args.adapter, device=args.device
    )
    features = pooled_features(
        simulator, examples, batch_size=args.batch_size, progress_every=20
    )
    state, report = train_heads(features, examples, epochs=args.epochs, seed=args.seed)
    state = calibrate(state, features, examples)

    torch.save(state, args.out / "heads.pt")
    (args.out / "metrics.json").write_text(
        json.dumps(report.metrics_payload(), indent=2), encoding="utf-8"
    )
    print(json.dumps(report.metrics_payload(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
