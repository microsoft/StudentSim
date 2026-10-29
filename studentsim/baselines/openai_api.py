"""OpenAI API :class:`LLMClient` implementation.

The non-Azure counterpart of :mod:`studentsim.baselines.azure_openai`, for
the same closed-model baselines and judges. It talks to the OpenAI API
directly, or to any OpenAI-compatible endpoint (a local server, a gateway)
when a base URL is given.

Credentials are env-var-driven; no key or endpoint is hardcoded here. See
:mod:`studentsim.core.paths` for the env-var schema.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any

from studentsim.core.llm import LLMClient, LLMResponse, Message
from studentsim.core.paths import require_env

_COMPLETION_CAP = re.compile(r"(?i)gpt-?5")
"""Model names whose generated-token cap is ``max_completion_tokens``.

Mirrors the rule in :mod:`studentsim.baselines.azure_openai`; the two must
stay in step, because the GPT-5 line rejects the older cap name on both
services.
"""


class OpenAIClient(LLMClient):
    """OpenAI chat-completion client.

    Parameters
    ----------
    model
        Model id (e.g. ``"gpt-4o"`` or ``"gpt-5.4"``). Unlike an Azure
        deployment name, this is the provider's model id itself.
    api_key
        API key. Defaults to ``OPENAI_API_KEY`` env var.
    base_url
        Base URL of an OpenAI-compatible endpoint. Defaults to
        ``OPENAI_BASE_URL`` env var, or the OpenAI API when neither is set.
    _inner
        Optional injected OpenAI client; tests inject a fake. Production
        callers leave this as ``None`` to construct the real client.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        _inner: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        self.name = f"openai/{model}"
        self._model = model
        if _inner is not None:
            self._client = _inner
        else:
            from openai import OpenAI  # lazy import

            self._client = OpenAI(
                api_key=api_key or require_env("OPENAI_API_KEY"),
                base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
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
        # Models on the GPT-5 line take the generated-token cap under a
        # different name and reject the older one outright.
        cap = "max_completion_tokens" if _COMPLETION_CAP.search(self._model) else "max_tokens"
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            cap: max_tokens,
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
