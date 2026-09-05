import json

import pytest

from agent.tool_selector import select_tool


def _fake_llm(payload: dict):
    def call(_system: str, _user: str) -> str:
        return json.dumps(payload)

    return call


def test_select_tool_web_search():
    fake = _fake_llm({"tool": "web_search", "tool_input": "population of Tokyo 2024"})
    decision = select_tool(fake, "What is the population of Tokyo?")
    assert decision["tool"] == "web_search"
    assert decision["tool_input"] == "population of Tokyo 2024"


def test_select_tool_calculator():
    fake = _fake_llm({"tool": "calculator", "tool_input": "37 * 4.2"})
    decision = select_tool(fake, "What is 37 times 4.2?")
    assert decision["tool"] == "calculator"
    assert decision["tool_input"] == "37 * 4.2"


def test_select_tool_rejects_unknown_tool():
    fake = _fake_llm({"tool": "shell", "tool_input": "rm -rf /"})
    with pytest.raises(ValueError):
        select_tool(fake, "anything")


def test_select_tool_rejects_missing_input():
    fake = _fake_llm({"tool": "web_search", "tool_input": ""})
    with pytest.raises(ValueError):
        select_tool(fake, "anything")
