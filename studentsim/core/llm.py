"""The interface every step that calls a language model goes through.

It is small on purpose: chat completion plus an optional read-out of the top
log-probabilities, both of which providers support in much the same shape.
Concrete implementations live in :mod:`studentsim.baselines`, and
:func:`open_client` is where one is chosen, so a different provider can be
dropped in without touching the steps that use it.

``Message.role`` is constrained to ``"system" | "user" | "assistant"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn. ``content`` is text, or the provider's list of content parts.

    Almost every step here sends text. The chess judge sends a picture of the
    board alongside its question, and providers take that as a list of parts
    rather than a string, so the field accepts either and implementations pass
    it through untouched.
    """

    role: Role
    content: str | Sequence[Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A single completion plus optional top-k logprobs over the first generated token.

    ``top_logprobs`` is keyed by candidate string (e.g., ``"A"``, ``"B"``, ``"C"``,
    ``"D"`` for math fidelity); the value is the natural-log probability returned
    by the provider. Tokens not in the returned top-k get a floor handled by the
    caller (:data:`studentsim.domains.math.fidelity.LOGPROB_FLOOR` for math).
    """

    text: str
    top_logprobs: Mapping[str, float] | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Stateless chat-completion client.

    Implementations should be safe to call from multiple threads or async tasks
    (most provider SDKs already are). They must NOT carry mutable conversation
    state; the messages list is the entire input.
    """

    name: str
    """Short identifier for logging, e.g., ``"azure_openai/gpt-5.4"``."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_logprobs: int | None = None,
    ) -> LLMResponse:
        """Run one chat completion.

        Parameters
        ----------
        messages
            The full conversation; the first turn may be ``system``.
        max_tokens
            Token budget for the assistant's reply.
        temperature, top_p
            Sampling parameters; ``temperature=0.0`` means greedy.
        top_logprobs
            If set, request that many top-k logprobs on the first generated token.
            Used by :class:`studentsim.domains.math.fidelity.MathFidelity` for the
            four-way multiple-choice metric.
        """
        ...


def open_client(model: str) -> LLMClient:
    """A client for the named model.

    The steps that call a model ask for one through here, so swapping in
    another provider is a matter of returning a different implementation of
    :class:`LLMClient` rather than editing those steps.
    """
    from studentsim.baselines import AzureOpenAIClient

    return AzureOpenAIClient(deployment=model)
