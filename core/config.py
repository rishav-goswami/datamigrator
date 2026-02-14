from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_PROVIDER: Literal["openai", "groq", "ollama", "gemini"] = "groq"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    DATABASE_URL: str = ""
    REDIS_URL: str = ""
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OLLAMA_BASE_URL: str = ""
    GEMINI_API_KEY: str = ""
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
