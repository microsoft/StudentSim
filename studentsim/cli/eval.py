"""``studentsim-eval``: score a trained simulator on one domain.

Each Stage-2 student has an adapter under ``--adapter-root/<student_id>`` and
held-out records under ``--single-turn-dir`` and ``--multi-turn-dir``, named
``<student_id>.jsonl``. Whichever of the two directories is given decides
which metrics run; giving neither scores both from the domain's own layout.

Every path defaults to where the rest of the package puts things, so a run that
followed the pipeline needs only the domain::

    studentsim-eval --domain l2 --out runs/l2.json

which resolves to the backbone in :data:`~studentsim.core.simulator.STUDENT_BASE_MODEL`,
adapters under ``checkpoints/l2/stage2``, and records under ``data/l2/test_st``
and ``data/l2/test_mt``. All three domains use those directory names. Anything
placed elsewhere is named explicitly::

    studentsim-eval --domain l2 \\
        --adapter-root /scratch/adapters \\
        --single-turn-dir /scratch/records/test_st \\
        --out runs/l2.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from studentsim.core.paths import domain_data_dir, stage_ckpt_dir
from studentsim.core.simulator import STUDENT_BASE_MODEL
from studentsim.eval import (
    aggregate_fidelity,
    aggregate_responsiveness,
    evaluate_students,
)
from studentsim.eval.fidelity import hf_letter_logprobs

SINGLE_TURN_DIR = "test_st"
MULTI_TURN_DIR = "test_mt"
"""What each domain's builder names its held-out directories."""


def _student_ids(args: argparse.Namespace) -> list[str]:
    if args.student_id:
        return list(args.student_id)
    if args.roster:
        roster = json.loads(Path(args.roster).read_text(encoding="utf-8"))
        return [str(s) for s in roster]
    return sorted(p.name for p in Path(args.adapter_root).iterdir() if p.is_dir())


def _resolve_paths(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fill in the paths the caller left out, from the domain's own layout."""
    if args.adapter_root is None:
        args.adapter_root = stage_ckpt_dir(args.domain, 2)
        if not args.adapter_root.is_dir():
            parser.error(
                f"no Stage-2 adapters at {args.adapter_root}; "
                "train them first or pass --adapter-root"
            )

    # Naming one directory is how a caller asks for one metric, so the defaults
    # fill in only when neither was named. Filling a missing one in the other
    # case would score a metric that was not asked for.
    if args.single_turn_dir is not None or args.multi_turn_dir is not None:
        return
    records = domain_data_dir(args.domain)
    single, multi = records / SINGLE_TURN_DIR, records / MULTI_TURN_DIR
    args.single_turn_dir = single if single.is_dir() else None
    args.multi_turn_dir = multi if multi.is_dir() else None
    if args.single_turn_dir is None and args.multi_turn_dir is None:
        parser.error(
            f"no held-out records under {records}; expected {SINGLE_TURN_DIR}/ or "
            f"{MULTI_TURN_DIR}/. Build them, or pass --single-turn-dir and "
            "--multi-turn-dir"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studentsim-eval",
        description="Compute behavioral fidelity and guidance responsiveness.",
    )
    parser.add_argument("--domain", required=True, choices=("chess", "l2", "math"))
    parser.add_argument("--base-model", default=STUDENT_BASE_MODEL)
    parser.add_argument("--adapter-root", type=Path, default=None,
                        help="Default: the domain's Stage-2 checkpoint directory.")
    parser.add_argument("--single-turn-dir", type=Path, default=None)
    parser.add_argument("--multi-turn-dir", type=Path, default=None)
    parser.add_argument("--roster", type=Path, default=None, help="JSON list of student ids.")
    parser.add_argument("--student-id", action="append", default=None)
    parser.add_argument("--raw-dir", type=Path, default=None, help="Keep raw decodes here.")
    parser.add_argument("--out", type=Path, default=None, help="Write results JSON here.")
    parser.add_argument("--swift-binary", default=None)
    args = parser.parse_args(argv)

    _resolve_paths(parser, args)

    students = _student_ids(args)
    if not students:
        parser.error(f"no students found under {args.adapter_root}")

    # Fidelity is scored differently per domain: L2 compares error profiles and
    # math ranks the multiple-choice letters, so each needs its own scorer.
    issue_counter = None
    letter_logprobs_factory = None
    if args.single_turn_dir is not None:
        if args.domain == "l2":
            from studentsim.domains.l2.languagetool import LanguageToolCounter

            issue_counter = LanguageToolCounter()
        elif args.domain == "math":
            letter_logprobs_factory = hf_letter_logprobs

    results = evaluate_students(
        student_ids=students,
        domain=args.domain,
        base_model=args.base_model,
        adapter_root=args.adapter_root,
        single_turn_dir=args.single_turn_dir,
        multi_turn_dir=args.multi_turn_dir,
        raw_dir=args.raw_dir,
        issue_counter=issue_counter,
        letter_logprobs_factory=letter_logprobs_factory,
        swift_binary=args.swift_binary,
    )

    payload: dict[str, object] = {"domain": args.domain, "n_students": len(results)}
    if args.single_turn_dir is not None:
        fidelity = aggregate_fidelity(results, domain=args.domain)
        payload["fidelity"] = dataclasses.asdict(fidelity)
        print(f"fidelity        macro={fidelity.mean:.4f}  n_students={fidelity.n_students}")
        if fidelity.mean_micro != fidelity.mean:
            print(f"                micro={fidelity.mean_micro:.4f}")
    if args.multi_turn_dir is not None:
        responsiveness = aggregate_responsiveness(results, domain=args.domain)
        payload["responsiveness"] = dataclasses.asdict(responsiveness)
        print(
            f"responsiveness  macro={responsiveness.mean:.4f}  "
            f"n_students={responsiveness.n_students}"
        )
        if responsiveness.mean_micro != responsiveness.mean:
            print(f"                micro={responsiveness.mean_micro:.4f}")
        for mode, score in responsiveness.per_mode.items():
            print(f"  {mode}: {score:.4f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
