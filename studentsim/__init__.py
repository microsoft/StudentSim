"""StudentSim: per-student AI student simulators for adaptive AI tutoring research.

Top-level re-exports of the public surface.
"""

from studentsim.core.decoding import DecodingConfig
from studentsim.core.metric import FidelityMetric, GuidanceMetric
from studentsim.core.records import (
    GuidanceMode,
    MultiTurnRecord,
    SingleTurnRecord,
    StudentProfile,
)
from studentsim.core.simulator import Simulator, SimulatorSpec

__version__ = "0.0.1"

__all__ = [
    "DecodingConfig",
    "FidelityMetric",
    "GuidanceMetric",
    "GuidanceMode",
    "MultiTurnRecord",
    "SingleTurnRecord",
    "Simulator",
    "SimulatorSpec",
    "StudentProfile",
    "__version__",
]
