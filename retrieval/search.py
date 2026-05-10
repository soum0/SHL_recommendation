"""
retrieval/search.py
-------------------
Given a text query, returns the top-N most relevant assessments
from the FAISS index.

This file is imported by the agent — it doesn't run standalone.

Usage:
    from retrieval.search import search

    results = search("Java developer mid level stakeholder communication")
    # returns a list of assessment dicts from the catalog
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CATALOG_FILE = "catalog/shl_catalog.json"
INDEX_FILE   = "retrieval/faiss.index"
MAP_FILE     = "retrieval/index_map.json"
EMBED_MODEL  = "all-MiniLM-L6-v2"

# ── Load everything once when the module is imported ─────────────────────────
# This means the model and index stay in memory across requests (fast).
print("Loading catalog, FAISS index, and embedding model...")

with open(CATALOG_FILE, encoding="utf-8") as f:
    _catalog = json.load(f)

# Build a lookup dict: id → assessment dict
_catalog_by_id = {item["id"]: item for item in _catalog}

with open(MAP_FILE) as f:
    _index_map = json.load(f)   # {"0": "opq32r", "1": "verify-g-plus", ...}

_faiss_index = faiss.read_index(INDEX_FILE)
_model = SentenceTransformer(EMBED_MODEL)

print(f"Ready: {len(_catalog)} assessments, {_faiss_index.ntotal} vectors indexed.")


def search(query: str, top_k: int = 15) -> list[dict]:
    """
    Search the catalog for assessments relevant to the query.

    Args:
        query:  free-text query built from the conversation so far
        top_k:  how many candidates to return (LLM picks final 1-10 from these)

    Returns:
        List of assessment dicts (same shape as catalog entries).
    """
    # Embed the query the same way we embedded the catalog
    query_vec = _model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Search FAISS — returns distances and row indices
    distances, indices = _faiss_index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue   # FAISS returns -1 if fewer results than top_k exist

        assessment_id = _index_map.get(str(idx))
        if not assessment_id:
            continue

        assessment = _catalog_by_id.get(assessment_id)
        if not assessment:
            continue

        # Attach similarity score for debugging (not exposed to the user)
        results.append({**assessment, "_score": float(dist)})

    return results


def filter_by(
    assessments: list[dict],
    test_types:  list[str] | None = None,
    job_level:   str | None = None,
    remote_only: bool = False,
    adaptive_only: bool = False,
) -> list[dict]:
    """
    Optional hard filter on top of FAISS results.
    Call this after search() if the user has specified constraints.

    Args:
        assessments:   results from search()
        test_types:    e.g. ["K", "S"] to keep only Knowledge & Simulation tests
        job_level:     e.g. "Manager" to keep only assessments for that level
        remote_only:   only return remote-enabled assessments
        adaptive_only: only return adaptive assessments
    """
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