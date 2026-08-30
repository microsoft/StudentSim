"""Base Qwen3-4B-Instruct (no LoRA) baseline.

Included as a baseline for L2 and math (left out of chess, where an
untrained model rarely emits a legal move in coordinate notation, so a score
would measure format compliance rather than behaviour).

:class:`HFSimulator` with ``lora_adapter_path=None``, exposed as
a one-liner constructor for clarity.
"""

from __future__ import annotations

from studentsim.core.simulator import Simulator, SimulatorSpec

_DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def build_base_qwen3_simulator(
    *,
    domain: str,
    base_model: str = _DEFAULT_BASE_MODEL,
    device: str = "cuda",
) -> Simulator:
    """Construct a base-Qwen3-only simulator for ``domain``.

    No LoRA adapter is loaded; the model is used as-released. Used by the
    L2 and math eval paths to score the untrained baseline.
    """
    from studentsim.inference.hf_simulator import HFSimulator  # lazy import

    spec = SimulatorSpec(
        base_model=base_model,
        lora_adapter_path=None,
        domain=domain,
    )
    return HFSimulator(spec, device=device)
