"""Wrap a closed-model client as a baseline that answers whole chats.

A baseline receives the conversation as role-separated turns, so the calls
here take message lists. Every request goes through the ``chat_completion``
method of :class:`~studentsim.baselines.azure_openai.AzureOpenAIClient`,
greedy, and its reply is used as returned, including the error string that
call gives when the service refuses a request.

Math fidelity needs per-letter log-probabilities; they are read from the
top alternatives of the first generated token that is itself one of the
record's letters (see :func:`studentsim.baselines.scoring.closed_letter_logprobs`).

The wrapper also satisfies the :class:`Simulator` Protocol, where a prompt is
sent as one user turn.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Final, TypeVar

from studentsim.baselines.scoring import closed_letter_logprobs
from studentsim.core.decoding import DecodingConfig
from studentsim.core.llm import Message
from studentsim.core.simulator import Simulator, SimulatorSpec
from studentsim.domains.math.fidelity import LOGPROB_FLOOR

_DEFAULT_TOP_LOGPROBS = 20
LETTER_MAX_TOKENS: Final = 4
"""Generated-token cap on the request math fidelity reads letters from."""

_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclass
class LLMClientSimulator(Simulator):
    """A closed model, called once per chat.

    Parameters
    ----------
    client
        An :class:`~studentsim.baselines.azure_openai.AzureOpenAIClient`, or
        anything with the same ``chat_completion`` and ``name``.
    domain
        Domain name; recorded on :attr:`spec`.
    top_logprobs_k
        How many alternatives to request per generated token when reading
        letters. ``20`` is the OpenAI cap; the GPT-5 family caps at ``5``.
    workers
        Requests in flight at once. Results keep the input order.
    """

    client: Any
    domain: str
    top_logprobs_k: int = _DEFAULT_TOP_LOGPROBS
    workers: int = 8

    def __post_init__(self) -> None:
        self.spec = SimulatorSpec(
            base_model=self.client.name,
            lora_adapter_path=None,
            domain=self.domain,
        )

    def _map(self, function: Callable[[_T], _R], items: Sequence[_T]) -> list[_R]:
        if self.workers <= 1 or len(items) <= 1:
            return [function(item) for item in items]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(function, items))

    def chat(
        self,
        messages: Sequence[Message],
        *,
        max_new_tokens: int,
        catch_errors: bool = False,
    ) -> str:
        """The model's reply to one chat.

        With ``catch_errors``, an exception the client raises reads as an
        empty reply; otherwise it propagates.
        """
        try:
            out = self.client.chat_completion(
                list(messages), max_tokens=max_new_tokens, temperature=0.0
            )
        except Exception:
            if not catch_errors:
                raise
            return ""
        return out if isinstance(out, str) else ""

    def generate_chats(
        self,
        chats: Sequence[Sequence[Message]],
        *,
        max_new_tokens: int,
        catch_errors: bool = False,
    ) -> list[str]:
        return self._map(
            lambda chat: self.chat(
                chat, max_new_tokens=max_new_tokens, catch_errors=catch_errors
            ),
            chats,
        )

    def letter_logprobs(
        self, messages: Sequence[Message], letters: Sequence[str]
    ) -> dict[str, float]:
        """Per-letter log-probabilities, every letter at the floor when the
        request fails or returns no token log-probabilities."""
        floor = {letter: LOGPROB_FLOOR for letter in letters}
        try:
            out = self.client.chat_completion(
                list(messages),
                max_tokens=LETTER_MAX_TOKENS,
                temperature=0.0,
                top_logprobs=self.top_logprobs_k,
                return_raw=True,
            )
        except Exception:
            return floor
        if isinstance(out, str) or not out or not getattr(out, "choices", None):
            return floor
        logprobs = out.choices[0].logprobs
        if not logprobs or not logprobs.content:
            return floor
        return closed_letter_logprobs(logprobs.content, letters)

    def letter_logprobs_batch(
        self,
        chats: Sequence[Sequence[Message]],
        letters: Sequence[Sequence[str]],
    ) -> list[dict[str, float]]:
        return self._map(
            lambda pair: self.letter_logprobs(pair[0], pair[1]),
            list(zip(chats, letters)),
        )

    # --- Simulator Protocol surface -----------------------------------------

    def generate(self, prompt: str, *, decoding: DecodingConfig) -> str:
        return self.chat(
            [Message(role="user", content=prompt)], max_new_tokens=decoding.max_new_tokens
        )

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        decoding: DecodingConfig,
    ) -> list[str]:
        return self.generate_chats(
            [[Message(role="user", content=p)] for p in prompts],
            max_new_tokens=decoding.max_new_tokens,
        )

    def logprobs(
        self,
        prompt: str,
        *,
        candidates: Sequence[str],
    ) -> Mapping[str, float]:
        return self.letter_logprobs([Message(role="user", content=prompt)], candidates)
