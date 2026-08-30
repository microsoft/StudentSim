"""Closed-source LLM baselines and the Maia2 chess KT baseline.

Three categories of baseline surface here:

1. **Prompted closed models**, reached through Azure OpenAI, which implement
   the :class:`studentsim.core.llm.LLMClient` Protocol.
2. **The untrained base model**, scored for L2 and math. It is left out of
   chess, where an untrained model does not reliably emit a legal move in
   coordinate notation.
3. **Maia2**: chess-only KT model (no guidance pathway).

The :class:`LLMClientSimulator` wraps any :class:`LLMClient` as a
:class:`Simulator`, so the same fidelity / guidance runners work for trained
simulators and closed-source baselines.
"""

from studentsim.baselines.azure_openai import AzureOpenAIClient
from studentsim.baselines.base_qwen3 import build_base_qwen3_simulator
from studentsim.baselines.llm_simulator import LLMClientSimulator
from studentsim.baselines.maia2 import Maia2Simulator

__all__ = [
    "AzureOpenAIClient",
    "LLMClientSimulator",
    "Maia2Simulator",
    "build_base_qwen3_simulator",
]
