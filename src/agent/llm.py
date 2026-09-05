"""Thin wrapper around the Groq chat completion API."""
from __future__ import annotations

import time
from functools import lru_cache

from groq import Groq, GroqError

from .config import get_settings

_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2.0


@lru_cache
def _client() -> Groq:
    return Groq(api_key=get_settings().groq_api_key)


def call_groq(system_prompt: str, user_prompt: str, *, json_mode: bool = True) -> str:
    settings = get_settings()
    # gpt-oss models spend part of max_tokens on hidden reasoning before the
    # visible answer; low effort + a higher ceiling keeps JSON mode from
    # truncating to an empty completion (see console.groq.com/docs/reasoning).
    extra_body = {"reasoning_effort": "low"} if "gpt-oss" in settings.groq_model else {}

    last_error: GroqError | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = _client().chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"} if json_mode else None,
                temperature=0.2,
                max_tokens=2048,
                extra_body=extra_body,
            )
            return response.choices[0].message.content
        except GroqError as exc:
            # Groq's JSON mode occasionally fails generation transiently under load.
            last_error = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))
    raise last_error
