"""``studentsim-train``: launch Stage 1 or Stage 2 SFT for one domain.

Usage
-----
Stage 1 (pooled training)::

    studentsim-train --config configs/training/stage1_chess.yaml

Stage 2 over a roster of students::

    studentsim-train --config configs/training/stage2_chess.yaml \\
        --roster data/chess/rosters/stage2.json

Stage 2 for one specific student::

    studentsim-train --config configs/training/stage2_chess.yaml --student-id alice

The ``data_path`` / ``output_dir`` fields in Stage-2 YAMLs may contain a
``{student_id}`` placeholder which is substituted per-student.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from studentsim.training import (
    PerStudentDriver,
    Stage1Trainer,
    Stage2Trainer,
    TrainingConfig,
)


def _substitute_student(text: str, student_id: str) -> str:
    return text.replace("{student_id}", student_id)


def _per_student_config(base: TrainingConfig, student_id: str) -> TrainingConfig:
    """Substitute {student_id} placeholders in path-like fields."""
    return dataclasses.replace(
        base,
        data_path=_substitute_student(base.data_path, student_id),
        output_dir=_substitute_student(base.output_dir, student_id),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studentsim-train",
        description="Stage 1 or Stage 2 SFT.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to a training YAML.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--student-id",
        default=None,
        help="Stage 2 single-student mode.",
    )
    group.add_argument(
        "--roster",
        type=Path,
        default=None,
        help="JSON list of student ids; runs Stage 2 for each in order.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="With --roster, abort on the first per-student failure.",
    )
    parser.add_argument(
        "--trainer-seed",
        type=int,
        default=None,
        help=(
            "Override TrainingConfig.trainer_seed. Used by "
            "studentsim-reproduce table_std_seed to launch S=3 seeded "
            "training runs from one base YAML."
        ),
    )
    parser.add_argument(
        "--data-sampler-seed",
        type=int,
        default=None,
        help="Override TrainingConfig.data_sampler_seed (paired with --trainer-seed).",
    )
    args = parser.parse_args(argv)

    cfg = TrainingConfig.from_yaml(args.config)
    if args.trainer_seed is not None or args.data_sampler_seed is not None:
        cfg = dataclasses.replace(
            cfg,
            trainer_seed=(
                args.trainer_seed if args.trainer_seed is not None else cfg.trainer_seed
            ),
            data_sampler_seed=(
                args.data_sampler_seed
                if args.data_sampler_seed is not None
                else cfg.data_sampler_seed
            ),
        )

    if cfg.stage == 1:
        if args.student_id or args.roster:
            parser.error("Stage-1 config; --student-id and --roster are Stage-2 only.")
        trainer = Stage1Trainer(config=cfg)
        return trainer.run()

    # Stage 2.
    if args.student_id:
        per_student = _per_student_config(cfg, args.student_id)
        return Stage2Trainer(config=per_student, student_id=args.student_id).run()

    if args.roster is None:
        parser.error("Stage-2 config; pass --student-id or --roster.")

    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    driver = PerStudentDriver(
        roster=[str(s) for s in roster],
        config_builder=lambda sid: _per_student_config(cfg, sid),
    )
    driver.run_all(fail_fast=args.fail_fast)
    if driver.failed:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
