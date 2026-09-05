"""Thin wrapper around the Groq chat completion API."""
from __future__ import annotations

from groq import Groq

from .config import get_settings


def call_groq(system_prompt: str, user_prompt: str, *, json_mode: bool = True) -> str:
    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"} if json_mode else None,
        temperature=0.2,
    )
    return response.choices[0].message.content
