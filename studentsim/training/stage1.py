"""Stage 1: pooled-training trainer.

Stage 1 trains one domain-specific LoRA on records pooled across the
Stage-1 student set, mixing single-turn and multi-turn records at the
domain's rho. The trainer assembles the ms-swift command via
:func:`build_ms_swift_command` and invokes it through an injectable runner
(default: :class:`subprocess.run`).

Tests inject a fake runner; production calls the real subprocess.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from studentsim.core.paths import swift_bin
from studentsim.training.config import TrainingConfig
from studentsim.training.ms_swift import MsSwiftCommand, build_ms_swift_command
from studentsim.training.world import check_world_size

RunnerCallable = Callable[[Sequence[str]], int]
"""A runner takes an argv list and returns the subprocess exit code."""


def _default_runner(argv: Sequence[str]) -> int:
    return subprocess.run(list(argv), check=False).returncode


@dataclass
class Stage1Trainer:
    """Wraps a Stage-1 training run.

    Parameters
    ----------
    config
        :class:`TrainingConfig` with ``stage=1``.
    swift_binary
        Path to the ``swift`` CLI; defaults to
        :func:`studentsim.core.paths.swift_bin`.
    runner
        Callable that takes the argv list and returns an exit code. Defaults
        to ``subprocess.run(...).returncode``; tests inject a fake.
    """

    config: TrainingConfig
    swift_binary: str | None = None
    runner: RunnerCallable = _default_runner

    def __post_init__(self) -> None:
        if self.config.stage != 1:
            raise ValueError(
                f"Stage1Trainer requires config.stage == 1, got {self.config.stage}"
            )

    def build_command(self) -> MsSwiftCommand:
        """Materialize the ms-swift command without executing it."""
        binary = self.swift_binary or swift_bin()
        return build_ms_swift_command(self.config, swift_binary=binary)

    def run(self) -> int:
        """Execute the Stage-1 training. Returns the subprocess exit code.

        Raises ``RuntimeError`` if the exit code is non-zero.
        """
        check_world_size(
            self.config.world_size,
            effective_batch=self.config.effective_batch,
            per_device=self.config.per_device_batch,
        )
        command = self.build_command()
        code = self.runner(command.argv)
        if code != 0:
            raise RuntimeError(
                f"Stage-1 training failed (exit code {code}); command was: {command.shell()}"
            )
        return code
