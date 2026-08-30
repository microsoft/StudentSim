"""Cross-domain core abstractions.

This subpackage is pure Python: it must not import torch, transformers, peft, verl,
or any other heavy runtime. Concrete backends live in studentsim.inference,
studentsim.training, studentsim.tutor_rl, etc. The split keeps the type surface
importable in lightweight contexts (CI lint, doc generation, baseline-only users).
"""

from studentsim.core.decoding import DecodingConfig
from studentsim.core.llm import LLMClient, Message
from studentsim.core.metric import FidelityMetric, GuidanceMetric
from studentsim.core.records import (
    GuidanceMode,
    MultiTurnRecord,
    SingleTurnRecord,
    StudentProfile,
)
from studentsim.core.seeds import DATA_SAMPLER_SEED, TRAINER_SEED, seed_everything
from studentsim.core.simulator import Simulator, SimulatorSpec

__all__ = [
    "DATA_SAMPLER_SEED",
    "DecodingConfig",
    "FidelityMetric",
    "GuidanceMetric",
    "GuidanceMode",
    "LLMClient",
    "Message",
    "MultiTurnRecord",
    "SingleTurnRecord",
    "Simulator",
    "SimulatorSpec",
    "StudentProfile",
    "TRAINER_SEED",
    "seed_everything",
]
