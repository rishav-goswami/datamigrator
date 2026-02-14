from __future__ import annotations

import pytest
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from core.config import get_settings
from core.llm import LLMClient


@pytest.mark.parametrize(
    "provider,expected_type",
    [
        ("openai", ChatOpenAI),
        ("groq", ChatGroq),
        ("ollama", ChatOllama),
    ],
)
def test_llm_client_switches_providers(monkeypatch, provider, expected_type):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    get_settings.cache_clear()

    client_factory = LLMClient()
    client_factory.reset()
    client = client_factory.get_client()

    assert isinstance(client, expected_type)
