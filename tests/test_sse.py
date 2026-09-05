from api.sse import format_sse, stream_research


def test_format_sse_produces_valid_event_block():
    line = format_sse("plan", {"plan": []})
    assert line == 'event: plan\ndata: {"plan": []}\n\n'


class _FakeGraph:
    def __init__(self, updates):
        self._updates = updates

    async def astream(self, _state, config=None, stream_mode=None):
        for update in self._updates:
            yield update


async def test_stream_research_emits_one_event_per_node(monkeypatch):
    fake_updates = [
        {"planner": {"plan": [{"id": 0, "description": "q1", "status": "pending"}]}},
        {
            "act": {
                "tool_calls": [
                    {
                        "step_id": 0,
                        "tool": "web_search",
                        "input": "q1",
                        "output": "result",
                        "sources": ["https://x.example"],
                    }
                ]
            }
        },
        {
            "observe": {
                "findings": [
                    {"step_id": 0, "content": "result", "sources": ["https://x.example"]}
                ]
            }
        },
        {"reporter": {"report": "final report"}},
    ]
    monkeypatch.setattr("api.sse.build_graph", lambda: _FakeGraph(fake_updates))

    events = [chunk async for chunk in stream_research("question", "run-1")]
    joined = "".join(events)

    assert "event: plan" in joined
    assert "event: tool_call" in joined
    assert "event: finding" in joined
    assert "event: report" in joined
    assert joined.rstrip().endswith(format_sse("done", {}).rstrip())


async def test_stream_research_emits_error_event_on_failure(monkeypatch):
    class _BrokenGraph:
        async def astream(self, *_args, **_kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("api.sse.build_graph", lambda: _BrokenGraph())

    events = [chunk async for chunk in stream_research("question", "run-1")]
    joined = "".join(events)

    assert "event: error" in joined
    assert "boom" in joined
    assert "event: done" in joined
