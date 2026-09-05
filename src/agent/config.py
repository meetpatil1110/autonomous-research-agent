"""Environment-backed settings for the agent."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    tavily_api_key: str | None


@lru_cache
def get_settings() -> Settings:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return Settings(
        groq_api_key=api_key,
        groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
    )
