"""ChromaDB-backed memory: lets the agent skip re-searching a sub-question
it has already answered earlier in the same research run.

Scoped per run_id rather than shared globally across all research runs —
a stale match from an unrelated past topic would produce a wrong citation
in the final report, which matters more here than the extra cache hits a
global store would give.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import chromadb
from chromadb.api.models.Collection import Collection
from typing_extensions import TypedDict

_COLLECTION_NAME = "findings"
_PERSIST_DIR = ".chroma"
# Cosine distance (0 = identical). Kept conservative: reusing a finding for a
# question that wasn't really the same one would silently degrade research
# quality, which is worse than the occasional redundant search.
_SIMILARITY_THRESHOLD = 0.1


class CachedFinding(TypedDict):
    content: str
    sources: list[str]


@lru_cache
def _collection() -> Collection:
    client = chromadb.PersistentClient(path=_PERSIST_DIR)
    return client.get_or_create_collection(
        name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def add_finding(
    run_id: str,
    step_id: int,
    sub_question: str,
    content: str,
    sources: list[str],
    *,
    collection: Optional[Collection] = None,
) -> None:
    (collection or _collection()).add(
        ids=[f"{run_id}:{step_id}"],
        documents=[sub_question],
        metadatas=[
            {"run_id": run_id, "content": content, "sources": ",".join(sources)}
        ],
    )


def find_similar(
    run_id: str,
    sub_question: str,
    *,
    collection: Optional[Collection] = None,
) -> Optional[CachedFinding]:
    result = (collection or _collection()).query(
        query_texts=[sub_question],
        n_results=1,
        where={"run_id": run_id},
    )
    ids = result["ids"][0]
    if not ids or result["distances"][0][0] > _SIMILARITY_THRESHOLD:
        return None
    metadata = result["metadatas"][0][0]
    sources = metadata["sources"].split(",") if metadata["sources"] else []
    return {"content": metadata["content"], "sources": sources}
