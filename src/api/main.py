"""FastAPI app exposing the research agent over a streaming SSE endpoint.

/research is POST (not GET), so it's consumed via fetch()'s streaming body
reader rather than the browser EventSource API — EventSource can't send a
JSON body anyway. This mirrors how most LLM streaming APIs shape this.
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import ResearchRequest
from api.sse import stream_research

app = FastAPI(title="Autonomous Research Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "autonomous-research-agent",
        "docs": "/docs",
        "health": "/health",
        "research": "POST /research",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research")
async def research(request: ResearchRequest) -> StreamingResponse:
    run_id = uuid.uuid4().hex
    return StreamingResponse(
        stream_research(request.question, run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
