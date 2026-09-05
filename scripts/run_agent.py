"""CLI smoke test for the plan/act/observe/replan loop."""
from __future__ import annotations

import sys
import uuid

from agent.graph import DEFAULT_RECURSION_LIMIT, build_graph, initial_state


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Research question: ")
    graph = build_graph()
    result = graph.invoke(
        initial_state(question, uuid.uuid4().hex),
        config={"recursion_limit": DEFAULT_RECURSION_LIMIT},
    )

    print("\n=== Plan ===")
    for step in result["plan"]:
        print(f"[{step['status']}] {step['description']}")

    print("\n=== Reasoning trace ===")
    for call in result["tool_calls"]:
        print(f"- step {call['step_id']}: {call['tool']}({call['input']!r})")
        print(f"    -> {call['output'][:200]}")

    print("\n=== Findings ===")
    for finding in result["findings"]:
        sources = ", ".join(finding["sources"]) or "none"
        print(f"- (step {finding['step_id']}) [{sources}] {finding['content'][:200]}")

    print("\n=== Report ===")
    print(result["report"])


if __name__ == "__main__":
    main()
