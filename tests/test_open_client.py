"""Provider-selection tests for :func:`studentsim.core.llm.open_client`.

Selection rule under test (documented on ``open_client`` itself): an
explicit ``STUDENTSIM_LLM_PROVIDER`` wins; otherwise Azure is used when
``AZURE_OPENAI_ENDPOINT`` is set, and the OpenAI API otherwise. Clients are
really constructed with dummy credentials — construction makes no network
calls — and only their type and ``name`` are asserted.
"""

from __future__ import annotations

import pytest

from studentsim.baselines.azure_openai import AzureOpenAIClient
from studentsim.baselines.openai_api import OpenAIClient
from studentsim.core.llm import open_client

_ENV_VARS = [
    "STUDENTSIM_LLM_PROVIDER",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-azure-key")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")


def test_explicit_openai_wins_over_azure_env(monkeypatch):
    monkeypatch.setenv("STUDENTSIM_LLM_PROVIDER", "openai")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    client = open_client("gpt-4o")
    assert isinstance(client, OpenAIClient)
    assert client.name == "openai/gpt-4o"


def test_explicit_azure(monkeypatch):
    monkeypatch.setenv("STUDENTSIM_LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    client = open_client("gpt-5.4")
    assert isinstance(client, AzureOpenAIClient)
    assert client.name == "azure_openai/gpt-5.4"


def test_provider_value_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv("STUDENTSIM_LLM_PROVIDER", " OpenAI ")
    assert isinstance(open_client("gpt-4o"), OpenAIClient)


def test_default_is_azure_when_endpoint_set(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    assert isinstance(open_client("gpt-4o"), AzureOpenAIClient)


def test_default_is_openai_without_azure_endpoint():
    assert isinstance(open_client("gpt-4o"), OpenAIClient)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("STUDENTSIM_LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="STUDENTSIM_LLM_PROVIDER"):
        open_client("gpt-4o")
