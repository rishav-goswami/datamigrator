from __future__ import annotations

from typing import Optional

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import get_settings


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
                self._client = ChatOpenAI(
                    model=settings.LLM_MODEL,
                    api_key=settings.OPENAI_API_KEY
                )
            elif provider == "groq":
                self._client = ChatGroq(
                    model=settings.LLM_MODEL,
                    api_key=settings.GROQ_API_KEY
                )
            elif provider == "ollama":
                self._client = ChatOllama(model=settings.LLM_MODEL)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

            self._provider = provider

        return self._client

    def reset(self) -> None:
        self._client = None
        self._provider = None
