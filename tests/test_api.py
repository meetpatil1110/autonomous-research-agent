from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_lists_endpoints():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/health"


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_rejects_empty_question():
    response = client.post("/research", json={"question": ""})
    assert response.status_code == 422


def test_research_streams_sse_events(monkeypatch):
    async def fake_stream(_question, _run_id):
        yield "event: plan\ndata: {}\n\n"
        yield "event: done\ndata: {}\n\n"

    monkeypatch.setattr("api.main.stream_research", fake_stream)

    with client.stream("POST", "/research", json={"question": "test question"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: plan" in body
    assert "event: done" in body
