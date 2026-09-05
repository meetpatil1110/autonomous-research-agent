"""Produces and revises the agent's research plan via the LLM.

`llm_call` is injected rather than imported directly so these functions can be
unit-tested without hitting the real Groq API.
"""
from __future__ import annotations

import json
from typing import Callable

from typing_extensions import TypedDict


class Finding(TypedDict):
    step_id: int
    content: str


LLMCall = Callable[[str, str], str]

_PLAN_SYSTEM_PROMPT = """You are a research planning assistant. Given a research \
question, break it into 2 to 4 concrete, independently-answerable sub-questions \
that together would let someone answer the original question.

Respond with ONLY a JSON object of the form:
{"steps": ["sub-question 1", "sub-question 2", ...]}
No prose, no markdown fences."""

_REPLAN_SYSTEM_PROMPT = """You are a research planning assistant reviewing progress \
on a research question. You are given the original question and the findings \
gathered so far for each sub-question. Decide whether additional sub-questions \
are needed to fully answer the original question.

Respond with ONLY a JSON object of the form:
{"additional_steps": ["new sub-question", ...]}
Use an empty list if no further steps are needed. No prose, no markdown fences."""


def generate_plan(llm_call: LLMCall, question: str) -> list[str]:
    raw = llm_call(_PLAN_SYSTEM_PROMPT, question)
    steps = json.loads(raw)["steps"]
    if not steps:
        return [question]
    return steps[:4]


def replan(llm_call: LLMCall, question: str, findings: list[Finding]) -> list[str]:
    findings_text = (
        "\n".join(f"- {f['content']}" for f in findings) or "(no findings yet)"
    )
    user_prompt = f"Original question: {question}\n\nFindings so far:\n{findings_text}"
    raw = llm_call(_REPLAN_SYSTEM_PROMPT, user_prompt)
    return json.loads(raw)["additional_steps"]
