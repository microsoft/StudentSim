"""Stage 1 + Stage 2 training, ms-swift CLI wrapper, per-student driver.

The training stack is one source of truth; the three domains differ only in
:class:`TrainingConfig` values, not in code paths.
"""

from studentsim.training.config import LoRAConfig, OptimizerConfig, TrainingConfig
from studentsim.training.ms_swift import MsSwiftCommand, build_ms_swift_command
from studentsim.training.per_student_driver import PerStudentDriver
from studentsim.training.stage1 import Stage1Trainer
from studentsim.training.stage2 import Stage2Trainer

__all__ = [
    "LoRAConfig",
    "MsSwiftCommand",
    "OptimizerConfig",
    "PerStudentDriver",
    "Stage1Trainer",
    "Stage2Trainer",
    "TrainingConfig",
    "build_ms_swift_command",
]
