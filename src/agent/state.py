"""Shared state schema passed between LangGraph nodes."""
from __future__ import annotations

from typing import Literal, Optional

from typing_extensions import TypedDict


class PlanStep(TypedDict):
    id: int
    description: str
    status: Literal["pending", "done"]


class Finding(TypedDict):
    step_id: int
    content: str
    sources: list[str]


class ToolCall(TypedDict):
    step_id: int
    tool: str
    input: str
    output: str
    sources: list[str]


class AgentState(TypedDict):
    question: str
    plan: list[PlanStep]
    findings: list[Finding]
    tool_calls: list[ToolCall]
    iteration: int
    max_iterations: int
    max_total_steps: int
    current_step_id: Optional[int]
