"""Example SSE client: streams a research run and prints it human-readably.

Shows how to consume /research from a plain script (no browser EventSource,
since that can't send a POST body) — also used to record the project's demo
recording.
"""
from __future__ import annotations

import json
import sys

import httpx

DEFAULT_URL = "http://127.0.0.1:8000/research"
_MAX_CHARS = 160


def _truncate(text: str, limit: int = _MAX_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit].rstrip() + "..."


def _render(event: str, payload: dict) -> None:
    if event == "plan":
        print(f"\n[PLAN] {len(payload['plan'])} sub-question(s)")
        for step in payload["plan"]:
            marker = "x" if step["status"] == "done" else " "
            print(f"  [{marker}] {step['description']}")
    elif event == "tool_call":
        print(f"\n[{payload['tool'].upper()}] {payload['input']}")
    elif event == "finding":
        sources = ", ".join(payload["sources"]) or "none"
        print(f"  -> {_truncate(payload['content'])}")
        print(f"     sources: {sources}")
    elif event == "report":
        print("\n[REPORT]\n")
        print(payload["report"])
    elif event == "error":
        print(f"\n[ERROR] {payload['message']}")
    elif event == "done":
        print("\n[DONE]")


def stream(question: str, url: str = DEFAULT_URL) -> None:
    with httpx.stream(
        "POST", url, json={"question": question}, timeout=None
    ) as response:
        event = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                payload = json.loads(line.removeprefix("data:").strip())
                _render(event, payload)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "What is the boiling point of water at sea level in Celsius?"
    )
    stream(question)
