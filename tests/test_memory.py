import chromadb
import pytest

from agent.memory import add_finding, find_similar


@pytest.fixture
def collection():
    client = chromadb.Client()
    return client.get_or_create_collection(
        name="test-findings", metadata={"hnsw:space": "cosine"}
    )


def test_find_similar_returns_none_when_empty(collection):
    assert find_similar("run-1", "anything", collection=collection) is None


def test_find_similar_finds_near_duplicate_question(collection):
    add_finding(
        "run-1",
        0,
        "What is the population of Tokyo?",
        "Tokyo has about 14 million people.",
        ["https://example.com/a"],
        collection=collection,
    )
    cached = find_similar("run-1", "What is Tokyo's population?", collection=collection)
    assert cached is not None
    assert "14 million" in cached["content"]
    assert cached["sources"] == ["https://example.com/a"]


def test_find_similar_ignores_unrelated_question(collection):
    add_finding(
        "run-1",
        0,
        "What is the population of Tokyo?",
        "Tokyo has about 14 million people.",
        ["https://example.com/a"],
        collection=collection,
    )
    cached = find_similar("run-1", "How does photosynthesis work?", collection=collection)
    assert cached is None


def test_find_similar_scoped_to_run_id(collection):
    add_finding(
        "run-1",
        0,
        "What is the population of Tokyo?",
        "Tokyo has about 14 million people.",
        ["https://example.com/a"],
        collection=collection,
    )
    cached = find_similar(
        "run-2", "What is the population of Tokyo?", collection=collection
    )
    assert cached is None
