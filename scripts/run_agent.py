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
            "iteration": 0,
            "max_iterations": 3,
            "current_step_id": None,
        }
    )

    print("\n=== Plan ===")
    for step in result["plan"]:
        print(f"[{step['status']}] {step['description']}")

    print("\n=== Findings ===")
    for finding in result["findings"]:
        print(f"- (step {finding['step_id']}) {finding['content']}")


if __name__ == "__main__":
    main()
