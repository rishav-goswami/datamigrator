from __future__ import annotations

import pytest
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from core.config import get_settings
from core.llm import LLMClient


@pytest.mark.parametrize(
    "provider,expected_type,env_extra",
    [
        ("openai", ChatOpenAI, {"OPENAI_API_KEY": "test-key", "GROQ_API_KEY": "test-key"}),
        ("groq", ChatGroq, {"OPENAI_API_KEY": "test-key", "GROQ_API_KEY": "test-key"}),
        ("ollama", ChatOllama, {"OPENAI_API_KEY": "test-key", "GROQ_API_KEY": "test-key"}),
        ("gemini", None, {"GEMINI_API_KEY": "test-key"}),  # None = ChatGoogleGenerativeAI (optional dep)
    ],
)
def test_llm_client_switches_providers(monkeypatch, provider, expected_type, env_extra):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    for k, v in env_extra.items():
        monkeypatch.setenv(k, v)

    get_settings.cache_clear()

    client_factory = LLMClient()
    client_factory.reset()
    raw = client_factory.get_client()
    inner = getattr(raw, "_client", raw)

    if expected_type is not None:
        assert isinstance(inner, expected_type)
    else:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            assert isinstance(inner, ChatGoogleGenerativeAI)
        except ImportError:
            pytest.skip("langchain-google-genai not installed")
