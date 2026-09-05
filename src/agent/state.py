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


class AgentState(TypedDict):
    question: str
    plan: list[PlanStep]
    findings: list[Finding]
    iteration: int
    max_iterations: int
    current_step_id: Optional[int]
