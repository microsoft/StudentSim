"""Simulator protocol: base + LoRA wrapper used everywhere the trained student
simulator needs to be invoked.

The backend used here is
:class:`studentsim.inference.hf_simulator.HFSimulator`, which loads a base
model and a LoRA adapter locally. Any other implementation of this Protocol
serves as well.

It honors the ``enable_thinking=False`` invariant from
:mod:`studentsim.core.decoding`. Callers depend on the :class:`Simulator`
Protocol declared here, not on the concrete classes; this keeps the dependency
direction (core does not import inference) and makes tests easy (a fake
Simulator is just a class with three methods).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from studentsim.core.decoding import DecodingConfig

STUDENT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
"""The backbone every student simulator is trained and evaluated on.

The same string appears in each domain's Stage-1 and Stage-2 config. It is
named here so evaluation can default to it without a caller having to repeat
the identifier, and so a change of backbone is one edit rather than a search.
The tutor policy is a different model and does not read this.
"""


@dataclass(frozen=True, slots=True)
class SimulatorSpec:
    """How to construct a Simulator: base model + (optional) LoRA adapter + domain.

    The ``domain`` field is here, not on the backend class, so that a Simulator
    constructed by user code knows which domain name
    its decoding defaults come from. The Domain itself is looked up from the
    registry; this dataclass holds only its name.
    """

    base_model: str
    lora_adapter_path: Optional[str]
    domain: str

    def __post_init__(self) -> None:
        if not self.base_model:
            raise ValueError("base_model must be non-empty")
        if not self.domain:
            raise ValueError("domain must be non-empty")


@runtime_checkable
class Simulator(Protocol):
    """Per-student or per-domain simulator.

    Concrete backends construct themselves from a :class:`SimulatorSpec`; users
    of this Protocol just need the three call methods.
    """

    spec: SimulatorSpec

    def generate(self, prompt: str, *, decoding: DecodingConfig) -> str:
        """Generate one completion for one prompt."""
        ...

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        decoding: DecodingConfig,
    ) -> Sequence[str]:
        """Generate completions for a batch of prompts. Output is positionally aligned."""
        ...

    def logprobs(
        self,
        prompt: str,
        *,
        candidates: Sequence[str],
    ) -> Mapping[str, float]:
        """Return the natural-log probability of each candidate as the next token
        (or completion) after ``prompt``.

        Used by :class:`studentsim.domains.math.fidelity.MathFidelity` for the
        four-way multiple-choice metric. Backends that cannot return logprobs
        on arbitrary candidates may raise ``NotImplementedError``.
        """
        ...
