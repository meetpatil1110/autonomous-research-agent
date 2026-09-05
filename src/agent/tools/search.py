"""Tavily-backed web search tool."""
from __future__ import annotations

from tavily import TavilyClient
from typing_extensions import TypedDict

from ..config import get_settings


class SearchResult(TypedDict):
    title: str
    url: str
    content: str


def web_search(query: str, *, max_results: int = 3) -> list[SearchResult]:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query, max_results=max_results)
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in response["results"]
    ]
