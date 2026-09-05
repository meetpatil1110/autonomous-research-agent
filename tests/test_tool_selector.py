import json

import pytest
from groq import GroqError

from agent.tool_selector import select_tool, select_tool_with_fallback


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


def test_fallback_returns_decision_unchanged_on_success():
    fake = _fake_llm({"tool": "calculator", "tool_input": "1 + 1"})
    decision = select_tool_with_fallback(fake, "anything")
    assert decision == {"tool": "calculator", "tool_input": "1 + 1"}


def test_fallback_defaults_to_web_search_on_groq_error():
    def raising_call(_system: str, _user: str) -> str:
        raise GroqError("boom")

    decision = select_tool_with_fallback(raising_call, "original sub-question")
    assert decision == {"tool": "web_search", "tool_input": "original sub-question"}


def test_fallback_defaults_to_web_search_on_invalid_json():
    def bad_json_call(_system: str, _user: str) -> str:
        return "not json"

    decision = select_tool_with_fallback(bad_json_call, "original sub-question")
    assert decision == {"tool": "web_search", "tool_input": "original sub-question"}


def test_fallback_defaults_to_web_search_on_unknown_tool():
    fake = _fake_llm({"tool": "shell", "tool_input": "rm -rf /"})
    decision = select_tool_with_fallback(fake, "original sub-question")
    assert decision == {"tool": "web_search", "tool_input": "original sub-question"}
