"""LangGraph state machine: plan -> act -> observe -> replan (loop) -> end.

`act` and `observe` are stubs in this phase — real tool calls (web search,
code execution) replace them in Phase 2.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .llm import call_groq
from .planner import generate_plan, replan
from .state import AgentState


def plan_node(state: AgentState) -> dict:
    step_descriptions = generate_plan(call_groq, state["question"])
    plan = [
        {"id": i, "description": desc, "status": "pending"}
        for i, desc in enumerate(step_descriptions)
    ]
    return {"plan": plan}


def act_node(state: AgentState) -> dict:
    next_step = next(s for s in state["plan"] if s["status"] == "pending")
    return {"current_step_id": next_step["id"]}


def observe_node(state: AgentState) -> dict:
    current_id = state["current_step_id"]
    plan = [
        {**s, "status": "done"} if s["id"] == current_id else s
        for s in state["plan"]
    ]
    step = next(s for s in plan if s["id"] == current_id)
    finding = {
        "step_id": current_id,
        "content": f"[stub observation for: {step['description']}]",
    }
    return {"plan": plan, "findings": state["findings"] + [finding]}


def replan_node(state: AgentState) -> dict:
    additional = replan(call_groq, state["question"], state["findings"])
    next_id = len(state["plan"])
    new_steps = [
        {"id": next_id + i, "description": desc, "status": "pending"}
        for i, desc in enumerate(additional)
    ]
    return {
        "plan": state["plan"] + new_steps,
        "iteration": state["iteration"] + 1,
    }


def _route_after_observe(state: AgentState) -> str:
    if any(s["status"] == "pending" for s in state["plan"]):
        return "act"
    return "replan"


def _route_after_replan(state: AgentState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return END
    if any(s["status"] == "pending" for s in state["plan"]):
        return "act"
    return END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("observe", observe_node)
    graph.add_node("replan", replan_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "act")
    graph.add_edge("act", "observe")
    graph.add_conditional_edges(
        "observe", _route_after_observe, {"act": "act", "replan": "replan"}
    )
    graph.add_conditional_edges(
        "replan", _route_after_replan, {"act": "act", END: END}
    )
    return graph.compile()
