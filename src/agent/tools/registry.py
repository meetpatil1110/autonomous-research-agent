"""Dispatches a tool name + input to its implementation and normalizes the result."""
from __future__ import annotations

from typing_extensions import TypedDict

from .calculator import CalculatorError, calculate
from .search import web_search

TOOL_NAMES = ("web_search", "calculator")


class ToolResult(TypedDict):
    summary: str
    sources: list[str]


def run_tool(tool_name: str, tool_input: str) -> ToolResult:
    if tool_name == "web_search":
        results = web_search(tool_input)
        if not results:
            return {"summary": "No search results found.", "sources": []}
        summary = "\n".join(f"- {r['title']}: {r['content']}" for r in results)
        sources = [r["url"] for r in results]
        return {"summary": summary, "sources": sources}

    if tool_name == "calculator":
        try:
            value = calculate(tool_input)
        except CalculatorError as exc:
            return {"summary": f"Calculator error: {exc}", "sources": []}
        return {"summary": f"{tool_input} = {value}", "sources": ["calculator"]}

    raise ValueError(f"Unknown tool: {tool_name}")
