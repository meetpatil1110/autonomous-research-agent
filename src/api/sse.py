"""Turns the agent's LangGraph execution into a live Server-Sent Events stream.

Uses stream_mode="updates" so each yielded item is {node_name: partial_state}
for exactly the node that just finished — the natural unit for "one step of
reasoning" the client should see as it happens, rather than a full state
snapshot the client would have to diff itself.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from agent.graph import DEFAULT_RECURSION_LIMIT, build_graph, initial_state

# planner/replan send the authoritative current plan (simpler for the client
# than merging deltas); act/observe send just the single item each produced.
_NODE_TO_EVENT = {
    "planner": "plan",
    "act": "tool_call",
    "observe": "finding",
    "replan": "plan",
    "reporter": "report",
}


def format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _payload_for(node_name: str, partial_state: dict[str, Any]) -> dict[str, Any]:
    if node_name in ("planner", "replan"):
        return {
            "plan": partial_state["plan"],
            "iteration": partial_state.get("iteration", 0),
        }
    if node_name == "act":
        return partial_state["tool_calls"][-1]
    if node_name == "observe":
        return partial_state["findings"][-1]
    if node_name == "reporter":
        return {"report": partial_state["report"]}
    return {}


async def stream_research(question: str, run_id: str) -> AsyncIterator[str]:
    graph = build_graph()
    state = initial_state(question, run_id)

    try:
        async for update in graph.astream(
            state,
            config={"recursion_limit": DEFAULT_RECURSION_LIMIT},
            stream_mode="updates",
        ):
            for node_name, partial_state in update.items():
                event = _NODE_TO_EVENT.get(node_name)
                if event is None:
                    continue
                yield format_sse(event, _payload_for(node_name, partial_state))
    except Exception as exc:
        yield format_sse("error", {"message": str(exc)})
    finally:
        yield format_sse("done", {})
