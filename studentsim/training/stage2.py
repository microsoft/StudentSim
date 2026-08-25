"""Stage 2: per-student specialization trainer.

Stage 2 loads the Stage-1 adapter and continues training the same LoRA on one
student's records with a smaller learning rate and shorter warmup. The
:class:`Stage2Trainer` invariant is that ``config.initial_adapter`` is set;
:class:`PerStudentDriver` (in :mod:`studentsim.training.per_student_driver`)
iterates this trainer across the Stage-2 student set.
"""

from __future__ import annotations

from dataclasses import dataclass

from studentsim.core.paths import swift_bin
from studentsim.training.config import TrainingConfig
from studentsim.training.ms_swift import MsSwiftCommand, build_ms_swift_command
from studentsim.training.stage1 import RunnerCallable, _default_runner
from studentsim.training.world import check_world_size


@dataclass
class Stage2Trainer:
    """Per-student Stage-2 trainer. One instance trains one student.

    Multiple students can be trained sequentially with the same Stage-1
    adapter by constructing one Stage2Trainer per student via
    :class:`PerStudentDriver`, which is the recommended entry point.
    """

    config: TrainingConfig
    student_id: str
    swift_binary: str | None = None
    runner: RunnerCallable = _default_runner

    def __post_init__(self) -> None:
        if self.config.stage != 2:
            raise ValueError(
                f"Stage2Trainer requires config.stage == 2, got {self.config.stage}"
            )
        if self.config.initial_adapter is None:
            raise ValueError(
                "Stage2Trainer requires config.initial_adapter (Stage-1 adapter path)"
            )
        if not self.student_id:
            raise ValueError("student_id must be non-empty")

    def build_command(self) -> MsSwiftCommand:
        binary = self.swift_binary or swift_bin()
        return build_ms_swift_command(self.config, swift_binary=binary)

    def run(self) -> int:
        check_world_size(
            self.config.world_size,
            effective_batch=self.config.effective_batch,
            per_device=self.config.per_device_batch,
        )
        command = self.build_command()
        code = self.runner(command.argv)
        if code != 0:
            raise RuntimeError(
                f"Stage-2 training for {self.student_id!r} failed (exit code {code}); "
                f"command was: {command.shell()}"
            )
        return code
