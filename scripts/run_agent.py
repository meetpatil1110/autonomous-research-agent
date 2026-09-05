"""CLI smoke test for the plan/act/observe/replan loop."""
from __future__ import annotations

import sys

from agent.graph import build_graph


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Research question: ")
    graph = build_graph()
    result = graph.invoke(
        {
            "question": question,
            "plan": [],
            "findings": [],
            "tool_calls": [],
            "iteration": 0,
            "max_iterations": 3,
            "max_total_steps": 8,
            "current_step_id": None,
        },
        config={"recursion_limit": 50},
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


if __name__ == "__main__":
    main()
