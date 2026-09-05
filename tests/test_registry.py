from unittest.mock import patch

import pytest

from agent.tools.registry import run_tool


def test_run_tool_calculator_success():
    result = run_tool("calculator", "6 * 7")
    assert result["summary"] == "6 * 7 = 42"
    assert result["sources"] == ["calculator"]


def test_run_tool_calculator_error_is_returned_not_raised():
    result = run_tool("calculator", "os.system('ls')")
    assert "Calculator error" in result["summary"]
    assert result["sources"] == []


def test_run_tool_web_search_success():
    fake_results = [
        {"title": "A", "url": "https://a.example", "content": "content a"},
        {"title": "B", "url": "https://b.example", "content": "content b"},
    ]
    with patch("agent.tools.registry.web_search", return_value=fake_results):
        result = run_tool("web_search", "some query")
    assert "content a" in result["summary"]
    assert result["sources"] == ["https://a.example", "https://b.example"]


def test_run_tool_web_search_no_results():
    with patch("agent.tools.registry.web_search", return_value=[]):
        result = run_tool("web_search", "some query")
    assert result["summary"] == "No search results found."
    assert result["sources"] == []


def test_run_tool_unknown_name_raises():
    with pytest.raises(ValueError):
        run_tool("not_a_tool", "x")
