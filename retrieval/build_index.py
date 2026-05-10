"""
retrieval/build_index.py
------------------------
Reads catalog/shl_catalog.json, embeds each assessment description,
and saves a FAISS index to disk.

Run once (or whenever the catalog changes):
    python retrieval/build_index.py

Output:
    retrieval/faiss.index       <- the vector index
    retrieval/index_map.json    <- maps FAISS row number → assessment id
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CATALOG_FILE = "catalog/shl_catalog.json"
INDEX_FILE   = "retrieval/faiss.index"
MAP_FILE     = "retrieval/index_map.json"

# This model is small (~90MB), fast, and works well for short descriptions.
# It runs locally — no API key needed.
EMBED_MODEL  = "all-MiniLM-L6-v2"


def make_text(item: dict) -> str:
    """
    Combine all useful fields into one string for embedding.
    More information = better search results.
    """
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        "Test type: " + item.get("test_type", ""),
        "Job levels: " + ", ".join(item.get("job_levels", [])),
        "Languages: "  + ", ".join(item.get("languages", [])),
    ]
    return " | ".join(p for p in parts if p.strip())


def main():
    os.makedirs("retrieval", exist_ok=True)

    # ── Load catalog ─────────────────────────────────────────────────────────
    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"Loaded {len(catalog)} assessments from {CATALOG_FILE}")

    # ── Build text for each assessment ───────────────────────────────────────
    texts = [make_text(item) for item in catalog]

    # ── Load embedding model ─────────────────────────────────────────────────
    print(f"Loading embedding model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    # ── Embed all texts ──────────────────────────────────────────────────────
    print("Embedding assessments ...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # needed for cosine similarity with IndexFlatIP
    )

    print(f"Embedding shape: {embeddings.shape}")   # (num_assessments, 384)

    # ── Build FAISS index ────────────────────────────────────────────────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # IP = Inner Product = cosine similarity (after normalizing)
    index.add(embeddings.astype(np.float32))

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # ── Save index to disk ───────────────────────────────────────────────────
    faiss.write_index(index, INDEX_FILE)
    print(f"Saved index → {INDEX_FILE}")

    # ── Save row → assessment id mapping ────────────────────────────────────
    # FAISS returns row numbers (0, 1, 2, ...), we need to map those back to assessments
    index_map = {str(i): item["id"] for i, item in enumerate(catalog)}
    with open(MAP_FILE, "w") as f:
        json.dump(index_map, f, indent=2)
    print(f"Saved index map → {MAP_FILE}")

    print("\nDone! Files created:")
    print(f"  {INDEX_FILE}")
    print(f"  {MAP_FILE}")


if __name__ == "__main__":
    main()