"""``studentsim-tutor-rl``: launch the chess tutor RL run through verl.

Loads a YAML under ``configs/tutor_rl/`` and
shells out to ``verl.trainer.main_ppo`` with the constructed command line.

The verl loop is a heavy, multi-GPU job; this CLI assembles its command and
hands over, and runs no training itself.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from studentsim.tutor_rl import VerlCommand, build_verl_command
from studentsim.tutor_rl.rl_config import TutorRlConfig
from studentsim.tutor_rl.verl_runner import VerlRunner


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studentsim-tutor-rl",
        description="Run the chess tutor RL training from a YAML config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="A YAML under configs/tutor_rl/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the verl command without executing it.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "A verl override to append, repeatable. Use this for settings that "
            "belong to the machine rather than the experiment, such as "
            "actor_rollout_ref.model.override_config.attn_implementation."
        ),
    )
    args = parser.parse_args(argv)

    cfg = TutorRlConfig.from_yaml(args.config)
    if args.dry_run:
        command: VerlCommand = build_verl_command(
            training=cfg.training, reward=cfg.reward, extra_overrides=args.overrides
        )
        print(command.shell())
        return 0
    runner = VerlRunner(training=cfg.training, reward=cfg.reward, extra_overrides=tuple(args.overrides))
    return runner.run()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
