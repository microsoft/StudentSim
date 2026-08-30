"""Wrap any :class:`LLMClient` as a :class:`Simulator`.

Allows the closed-source baselines (GPT-4o, GPT-5.4) to be
scored by the same runners in :mod:`studentsim.eval` as the trained
simulators.

Math fidelity needs ``Simulator.logprobs``; the wrapper implements this via the
LLM client's ``top_logprobs`` parameter on the first generated token.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from studentsim.core.decoding import DecodingConfig
from studentsim.core.llm import LLMClient, Message
from studentsim.core.simulator import Simulator, SimulatorSpec
from studentsim.domains.math.fidelity import LOGPROB_FLOOR

_DEFAULT_TOP_LOGPROBS = 20


@dataclass
class LLMClientSimulator(Simulator):
    """Adapt a :class:`LLMClient` to the :class:`Simulator` Protocol.

    Parameters
    ----------
    client
        Any :class:`LLMClient` (Azure OpenAI).
    domain
        Domain name; recorded on :attr:`spec` so downstream code can identify
        which domain this simulator is configured for.
    system_prompt
        Optional system message prepended to every chat. Defaults to ``None``.
    top_logprobs_k
        How many top logprobs to request from the client when ``logprobs(...)``
        is called. ``20`` is the OpenAI default cap; GPT-5 family caps at ``5``,
        so the caller is expected to pass ``5`` for GPT-5.4. Letters not in
        the returned top-k fall through to ``LOGPROB_FLOOR = -10``.
    """

    client: LLMClient
    domain: str
    system_prompt: str | None = None
    top_logprobs_k: int = _DEFAULT_TOP_LOGPROBS

    def __post_init__(self) -> None:
        self.spec = SimulatorSpec(
            base_model=self.client.name,
            lora_adapter_path=None,
            domain=self.domain,
        )

    def _build_messages(self, prompt: str) -> list[Message]:
        msgs: list[Message] = []
        if self.system_prompt:
            msgs.append(Message(role="system", content=self.system_prompt))
        msgs.append(Message(role="user", content=prompt))
        return msgs

    def generate(self, prompt: str, *, decoding: DecodingConfig) -> str:
        response = self.client.complete(
            self._build_messages(prompt),
            max_tokens=decoding.max_new_tokens,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
        )
        return response.text

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        decoding: DecodingConfig,
    ) -> list[str]:
        # No batched chat API; serial calls.
        # Production users wrap with concurrency at the orchestration layer.
        return [self.generate(p, decoding=decoding) for p in prompts]

    def logprobs(
        self,
        prompt: str,
        *,
        candidates: Sequence[str],
    ) -> Mapping[str, float]:
        response = self.client.complete(
            self._build_messages(prompt),
            max_tokens=1,
            temperature=0.0,
            top_logprobs=self.top_logprobs_k,
        )
        top = response.top_logprobs or {}
        return {c: float(top.get(c, LOGPROB_FLOOR)) for c in candidates}
