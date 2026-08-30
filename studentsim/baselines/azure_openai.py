"""Azure OpenAI :class:`LLMClient` implementation.

Used for the closed-model baselines, and for the model that judges chess
tutor messages against the position.

Credentials are env-var-driven; no endpoint or deployment name is hardcoded
here. See :mod:`studentsim.core.paths` for the env-var schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from studentsim.core.llm import LLMClient, LLMResponse, Message
from studentsim.core.paths import require_env

if TYPE_CHECKING:
    pass

_DEFAULT_API_VERSION = "2024-08-01-preview"


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI chat-completion client.

    Parameters
    ----------
    deployment
        Azure deployment name (the user's deployment, not the model id, e.g.
        ``"gpt-4o"`` or ``"gpt-5.4"`` as deployed).
    endpoint
        Azure endpoint. Defaults to ``AZURE_OPENAI_ENDPOINT`` env var.
    api_key
        API key. Defaults to ``AZURE_OPENAI_API_KEY`` env var.
    api_version
        Azure API version. Defaults to a known-good version; override to match
        the deployment if needed.
    _inner
        Optional injected OpenAI client; tests inject a fake. Production
        callers leave this as ``None`` to construct the real client.
    """

    def __init__(
        self,
        *,
        deployment: str,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = _DEFAULT_API_VERSION,
        _inner: Any | None = None,
    ) -> None:
        if not deployment:
            raise ValueError("deployment must be non-empty")
        self.name = f"azure_openai/{deployment}"
        self._deployment = deployment
        if _inner is not None:
            self._client = _inner
        else:
            from openai import AzureOpenAI  # lazy import

            self._client = AzureOpenAI(
                api_key=api_key or require_env("AZURE_OPENAI_API_KEY"),
                azure_endpoint=endpoint or require_env("AZURE_OPENAI_ENDPOINT"),
                api_version=api_version,
            )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_logprobs: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._deployment,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_logprobs is not None:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        top_lp_map: dict[str, float] | None = None
        if top_logprobs is not None and choice.logprobs and choice.logprobs.content:
            # First generated token's top-k logprobs.
            first_token = choice.logprobs.content[0]
            top_lp_map = {tok.token: float(tok.logprob) for tok in first_token.top_logprobs}
        return LLMResponse(text=text, top_logprobs=top_lp_map)
