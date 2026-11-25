"""Azure OpenAI :class:`LLMClient` implementation.

Used for the closed-model baselines, and for the model that judges chess
tutor messages against the position.

Credentials are env-var-driven; no endpoint or deployment name is hardcoded
here. See :mod:`studentsim.core.paths` for the env-var schema.

Two entry points share one connection. :meth:`AzureOpenAIClient.complete` is
the :class:`LLMClient` call the rest of the package uses: one request, errors
raised. :meth:`AzureOpenAIClient.chat_completion` is the call the baselines are
scored through. It retries throttled and timed-out requests, and answers a
request the service refuses, or one that never succeeds, with
:data:`ERROR_RESPONSE` in place of a reply. That string is scored like any
other reply.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Final

from studentsim.core.llm import LLMClient, LLMResponse, Message
from studentsim.core.paths import require_env

_DEFAULT_API_VERSION = "2025-01-01-preview"
_COMPLETION_CAP = re.compile(r"(?i)gpt-?5")
"""Deployment names whose generated-token cap is ``max_completion_tokens``."""

ERROR_RESPONSE: Final = "[Error] Bad request, skipped."
"""What :meth:`AzureOpenAIClient.chat_completion` returns in place of a reply
the service refused or never gave."""

MAX_RETRIES: Final = 15
"""Attempts :meth:`AzureOpenAIClient.chat_completion` makes before giving up."""

_CONTENT_FILTER_MARKERS: Final = ("content_filter", "content management policy")


def _request_timeout() -> Any:
    import httpx  # lazy import; a dependency of the openai SDK

    return httpx.Timeout(connect=60.0, read=180.0, write=60.0, pool=None)


def _sdk_errors() -> tuple[type[BaseException], type[BaseException], type[BaseException]]:
    """The rate-limit, bad-request and timeout exception types of the SDK.

    Without the SDK installed (a test with an injected client), each is a type
    nothing raises, so every failure takes the generic path.
    """

    class _Never(Exception):
        pass

    try:
        import openai

        rate_limit, bad_request = openai.RateLimitError, openai.BadRequestError
    except ImportError:
        rate_limit = bad_request = _Never
    try:
        import httpx

        timeout = httpx.TimeoutException
    except ImportError:
        timeout = _Never
    return rate_limit, bad_request, timeout


def _is_content_filter(error: BaseException) -> bool:
    message = str(error)
    if any(marker in message.lower() for marker in _CONTENT_FILTER_MARKERS):
        return True
    if "ResponsibleAIPolicyViolation" in message:
        return True
    response = getattr(error, "response", None)
    return (
        response is not None
        and hasattr(response, "json")
        and response.json().get("error", {}).get("code") == "content_filter"
    )


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
        Azure API version, ``2025-01-01-preview`` by default.
    _inner
        Optional injected OpenAI client; tests inject a fake. Production
        callers leave this as ``None`` to construct the real client.
    _sleep
        What waits between retries; tests inject a no-op.
    """

    def __init__(
        self,
        *,
        deployment: str,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = _DEFAULT_API_VERSION,
        _inner: Any | None = None,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not deployment:
            raise ValueError("deployment must be non-empty")
        self.name = f"azure_openai/{deployment}"
        self._deployment = deployment
        self._sleep = _sleep
        if _inner is not None:
            self._client = _inner
            self._timeout = None
        else:
            from openai import AzureOpenAI  # lazy import

            self._client = AzureOpenAI(
                api_key=api_key or require_env("AZURE_OPENAI_API_KEY"),
                azure_endpoint=endpoint or require_env("AZURE_OPENAI_ENDPOINT"),
                api_version=api_version,
            )
            self._timeout = _request_timeout()

    def _cap(self) -> str:
        # Deployments on the GPT-5 line take the generated-token cap under a
        # different name and reject the older one outright.
        return "max_completion_tokens" if _COMPLETION_CAP.search(self._deployment) else "max_tokens"

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
            self._cap(): max_tokens,
            "temperature": temperature,
        }
        # The service default is 1.0, so it is sent only when it differs.
        if top_p != 1.0:
            kwargs["top_p"] = top_p
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

    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_logprobs: int | None = None,
        return_raw: bool = False,
        max_retries: int = MAX_RETRIES,
    ) -> Any:
        """One chat completion, retried, as the baselines are scored.

        Returns the reply text with surrounding whitespace stripped, or with
        ``return_raw`` the SDK's completion object. A throttled request waits
        with exponential backoff and a timed-out or otherwise failed one waits
        two seconds, each costing one of ``max_retries`` attempts. A bad
        request or a content-filter refusal returns :data:`ERROR_RESPONSE`
        at once, and so does running out of attempts. A reply with no content
        fails on ``strip`` and is retried like any other failure.
        """
        rate_limit, bad_request, timeout = _sdk_errors()
        kwargs: dict[str, Any] = {
            "model": self._deployment,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            self._cap(): max_tokens,
            "temperature": temperature,
        }
        if top_logprobs is not None:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout

        retries = 0
        while retries < max_retries:
            try:
                completion = self._client.chat.completions.create(**kwargs)
                if return_raw:
                    return completion
                return [c.message.content.strip() for c in completion.choices][0]
            except rate_limit:
                self._sleep(min(2 ** min(retries, 6) + random.uniform(0, 1), 60))
                retries += 1
            except bad_request:
                return ERROR_RESPONSE
            except timeout:
                retries += 1
                self._sleep(2)
            except Exception as error:  # noqa: BLE001 - every failure is retried or answered
                if _is_content_filter(error):
                    return ERROR_RESPONSE
                retries += 1
                self._sleep(2)
        return ERROR_RESPONSE
