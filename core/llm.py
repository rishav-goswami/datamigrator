from __future__ import annotations

import time
from typing import Any, Optional

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from .config import get_settings

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


def _is_rate_limit_or_transient(exc: BaseException) -> bool:
    """True if we should retry (rate limit or transient error)."""
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "resource exhausted" in msg:
        return True
    if "503" in msg or "unavailable" in msg or "connection" in msg:
        return True
    if "timeout" in msg or "timed out" in msg:
        return True
    return False


class _RetryWrapper:
    """Wraps a chat model and retries invoke() on rate limit / transient errors."""

    def __init__(self, client: Any, max_retries: int = 4, min_wait: float = 2.0, max_wait: float = 60.0):
        self._client = client
        self._max_retries = max_retries
        self._min_wait = min_wait
        self._max_wait = max_wait

    def invoke(self, messages, **kwargs):
        last_exc = None
        for attempt in range(self._max_retries):
            try:
                return self._client.invoke(messages, **kwargs)
            except Exception as e:
                last_exc = e
                if not _is_rate_limit_or_transient(e) or attempt == self._max_retries - 1:
                    raise
                wait = min(
                    self._max_wait,
                    max(self._min_wait, (2 ** attempt) + 1),
                )
                time.sleep(wait)
        raise last_exc

    def __getattr__(self, name: str):
        return getattr(self._client, name)


class LLMClient:
    _instance: Optional["LLMClient"] = None

    def __new__(cls) -> "LLMClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
            cls._instance._provider = None
        return cls._instance

    def get_client(self):
        settings = get_settings()
        provider = settings.LLM_PROVIDER

        if self._client is None or self._provider != provider:
            if provider == "openai":
                client = ChatOpenAI(
                    model=settings.LLM_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                )
            elif provider == "groq":
                client = ChatGroq(
                    model=settings.LLM_MODEL,
                    api_key=settings.GROQ_API_KEY,
                )
            elif provider == "ollama":
                client = ChatOllama(model=settings.LLM_MODEL)
            elif provider == "gemini":
                if ChatGoogleGenerativeAI is None:
                    raise ImportError(
                        "Gemini support requires: pip install langchain-google-genai"
                    )
                api_key = settings.GEMINI_API_KEY or ""
                if not api_key:
                    raise ValueError(
                        "GEMINI_API_KEY is required when LLM_PROVIDER=gemini"
                    )
                client = ChatGoogleGenerativeAI(
                    model=settings.LLM_MODEL,
                    api_key=api_key,
                    max_retries=2,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

            self._client = _RetryWrapper(client)
            self._provider = provider

        return self._client

    def reset(self) -> None:
        self._client = None
        self._provider = None
