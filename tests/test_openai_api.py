"""Unit tests for :class:`studentsim.baselines.openai_api.OpenAIClient`.

The client accepts an injected inner SDK client (the ``_inner`` seam the
Azure client already exposes), so these tests run entirely against fakes:
no network, no API key. The Azure client is exercised alongside it to pin
the two implementations to the same ``LLMClient`` contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from studentsim.baselines.azure_openai import AzureOpenAIClient
from studentsim.baselines.openai_api import OpenAIClient
from studentsim.core.llm import LLMResponse, Message

MESSAGES = [
    Message(role="system", content="You are a student."),
    Message(role="user", content="What is 2 + 2?"),
]


class _FakeCompletions:
    """Records ``create`` kwargs and returns a canned response."""

    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeInner:
    """Stands in for the OpenAI SDK client object both clients wrap."""

    def __init__(self, response):
        self.completions = _FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def _response(text="4", top_logprobs=None):
    logprobs = None
    if top_logprobs is not None:
        content = [
            SimpleNamespace(
                top_logprobs=[
                    SimpleNamespace(token=token, logprob=logprob) for token, logprob in top_logprobs
                ]
            )
        ]
        logprobs = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=SimpleNamespace(content=text), logprobs=logprobs)
    return SimpleNamespace(choices=[choice])


def _openai_client(model="gpt-4o", response=None):
    return OpenAIClient(model=model, _inner=_FakeInner(response or _response()))


def test_name_and_empty_model():
    assert _openai_client().name == "openai/gpt-4o"
    with pytest.raises(ValueError):
        OpenAIClient(model="", _inner=_FakeInner(_response()))


def test_request_shaping_and_text_normalization():
    client = _openai_client(response=_response(text="The answer is 4."))
    result = client.complete(MESSAGES, max_tokens=64, temperature=0.7, top_p=0.9)

    assert result == LLMResponse(text="The answer is 4.", top_logprobs=None)
    (kwargs,) = client._client.completions.calls
    assert kwargs == {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a student."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
    }


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5", "gpt5", "GPT-5-mini"])
def test_gpt5_models_use_max_completion_tokens(model):
    client = _openai_client(model=model)
    client.complete(MESSAGES, max_tokens=32)
    (kwargs,) = client._client.completions.calls
    assert kwargs["max_completion_tokens"] == 32
    assert "max_tokens" not in kwargs


def test_empty_content_becomes_empty_text():
    client = _openai_client(response=_response(text=None))
    assert client.complete(MESSAGES, max_tokens=8).text == ""


def test_content_parts_pass_through_untouched():
    parts = [{"type": "text", "text": "What move?"}]
    client = _openai_client()
    client.complete([Message(role="user", content=parts)], max_tokens=8)
    (kwargs,) = client._client.completions.calls
    assert kwargs["messages"] == [{"role": "user", "content": parts}]


def test_top_logprobs_requested_and_parsed():
    response = _response(top_logprobs=[("A", -0.25), ("B", -2)])
    client = _openai_client(model="gpt-5.4", response=response)
    result = client.complete(MESSAGES, max_tokens=1, top_logprobs=5)

    (kwargs,) = client._client.completions.calls
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] == 5
    assert result.top_logprobs == {"A": -0.25, "B": -2.0}
    assert all(isinstance(v, float) for v in result.top_logprobs.values())


def test_top_logprobs_missing_from_response_yields_none():
    client = _openai_client(response=_response(top_logprobs=None))
    result = client.complete(MESSAGES, max_tokens=1, top_logprobs=5)
    assert result.top_logprobs is None


def test_contract_parity_with_azure_client():
    response = _response(text="C", top_logprobs=[("C", -0.5), ("A", -1.5)])
    azure = AzureOpenAIClient(deployment="gpt-5.4", _inner=_FakeInner(response))
    openai_client = OpenAIClient(model="gpt-5.4", _inner=_FakeInner(response))

    kwargs = dict(max_tokens=1, temperature=0.0, top_p=1.0, top_logprobs=5)
    azure_result = azure.complete(MESSAGES, **kwargs)
    openai_result = openai_client.complete(MESSAGES, **kwargs)

    assert azure_result == openai_result
    assert azure._client.completions.calls == openai_client._client.completions.calls
    assert azure.name == "azure_openai/gpt-5.4"
    assert openai_client.name == "openai/gpt-5.4"
