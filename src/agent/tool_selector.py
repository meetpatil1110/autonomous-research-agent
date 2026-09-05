"""Decides which tool answers a single research step, and with what input.

This is the agent's "Act" decision: given one sub-question, pick a tool and
produce its input via the LLM, rather than having the planner hard-wire a
tool per step up front.
"""
from __future__ import annotations

import json

from typing_extensions import TypedDict

from .planner import LLMCall
from .tools.registry import TOOL_NAMES

_TOOL_SELECT_SYSTEM_PROMPT = """You are a research agent deciding how to answer one \
sub-question using the tools available to you.

Tools:
- web_search: search the web. Use for anything requiring current facts, data, \
or general knowledge.
- calculator: evaluate a mathematical expression (numbers, + - * / ** % // \
parentheses, and functions sqrt/log/log10/exp/sin/cos/tan/abs/round). Use only \
when the sub-question requires a numeric computation.

Respond with ONLY a JSON object of the form:
{"tool": "web_search" | "calculator", "tool_input": "..."}
For web_search, tool_input is a concise, effective search query.
For calculator, tool_input is a valid arithmetic expression.
No prose, no markdown fences."""


class ToolDecision(TypedDict):
    tool: str
    tool_input: str


def select_tool(llm_call: LLMCall, sub_question: str) -> ToolDecision:
    raw = llm_call(_TOOL_SELECT_SYSTEM_PROMPT, sub_question)
    decision = json.loads(raw)
    if decision.get("tool") not in TOOL_NAMES:
        raise ValueError(f"Unknown tool selected: {decision.get('tool')}")
    if not decision.get("tool_input"):
        raise ValueError("Tool decision is missing tool_input")
    return decision
