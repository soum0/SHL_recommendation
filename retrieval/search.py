"""
retrieval/search.py
-------------------
Searches the FAISS index using OpenRouter embeddings.
No local model — uses API calls for query embedding only.

Imported by agent/llm.py at runtime.
"""

import json
import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CATALOG_FILE = "catalog/shl_catalog.json"
INDEX_FILE   = "retrieval/faiss.index"
MAP_FILE     = "retrieval/index_map.json"
EMBED_MODEL  = "openai/text-embedding-3-small"

# ── Load everything once when the module is first imported ────────────────────
print("Loading catalog and FAISS index...")

with open(CATALOG_FILE, encoding="utf-8") as f:
    _catalog = json.load(f)

_catalog_by_id = {item["id"]: item for item in _catalog}

with open(MAP_FILE) as f:
    _index_map = json.load(f)

_faiss_index = faiss.read_index(INDEX_FILE)

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

print(f"Ready: {len(_catalog)} assessments, {_faiss_index.ntotal} vectors indexed.")


def _embed_query(query: str) -> np.ndarray:
    """Embed a single query string using OpenRouter."""
    response = _client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    vec = np.array([response.data[0].embedding], dtype=np.float32)
    # Normalize for cosine similarity
    norm = np.linalg.norm(vec)
    return vec / max(norm, 1e-10)


def search(query: str, top_k: int = 15) -> list[dict]:
    """
    Search the catalog for assessments relevant to the query.

    Args:
        query:  free-text built from conversation
        top_k:  how many candidates to return

    Returns:
        List of assessment dicts with _score attached.
    """
    query_vec = _embed_query(query)
    distances, indices = _faiss_index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        assessment_id = _index_map.get(str(idx))
        if not assessment_id:
            continue

        assessment = _catalog_by_id.get(assessment_id)
        if not assessment:
            continue

        results.append({**assessment, "_score": float(dist)})

    return results


def filter_by(
    assessments: list[dict],
    test_types:   list[str] | None = None,
    job_level:    str | None = None,
    remote_only:  bool = False,
    adaptive_only: bool = False,
) -> list[dict]:
    """Optional hard filter on top of FAISS results."""
    out = assessments

    if test_types:
        out = [a for a in out if any(t in a.get("test_type", "") for t in test_types)]

    if job_level:
        out = [a for a in out if job_level in a.get("job_levels", [])]

    if remote_only:
        out = [a for a in out if a.get("remote_testing")]

    if adaptive_only:
        out = [a for a in out if a.get("adaptive")]

    return out