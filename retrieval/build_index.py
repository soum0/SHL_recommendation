"""
retrieval/build_index.py
------------------------
Builds FAISS index using OpenRouter embeddings API.
No local model needed — runs in minimal RAM.

Run once on your LOCAL machine only:
    python retrieval/build_index.py

Then commit the output files to GitHub:
    git add retrieval/faiss.index retrieval/index_map.json
    git commit -m "add prebuilt faiss index"
    git push

Output:
    retrieval/faiss.index
    retrieval/index_map.json
"""

import json
import os
import time
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CATALOG_FILE = "catalog/shl_catalog.json"
INDEX_FILE   = "retrieval/faiss.index"
MAP_FILE     = "retrieval/index_map.json"
EMBED_MODEL  = "openai/text-embedding-3-small"

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def make_text(item: dict) -> str:
    """Combine all fields into one string for embedding."""
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        "Test type: " + item.get("test_type", ""),
        "Job levels: " + ", ".join(item.get("job_levels", [])),
        "Languages: "  + ", ".join(item.get("languages", [])),
    ]
    return " | ".join(p for p in parts if p.strip())


def get_embeddings(texts: list[str], batch_size: int = 50) -> np.ndarray:
    """Get embeddings from OpenRouter in batches."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch)} items)...")

        response = _client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        time.sleep(0.3)

    return np.array(all_embeddings, dtype=np.float32)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-10)


def main():
    os.makedirs("retrieval", exist_ok=True)

    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"Loaded {len(catalog)} assessments")

    texts = [make_text(item) for item in catalog]

    print(f"\nEmbedding {len(texts)} assessments via OpenRouter...")
    embeddings = get_embeddings(texts)
    embeddings = normalize(embeddings)

    print(f"Embedding shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, INDEX_FILE)
    print(f"Saved → {INDEX_FILE}")

    index_map = {str(i): item["id"] for i, item in enumerate(catalog)}
    with open(MAP_FILE, "w") as f:
        json.dump(index_map, f, indent=2)
    print(f"Saved → {MAP_FILE}")

    print("\nDone! Now commit these files to GitHub:")
    print("  git add retrieval/faiss.index retrieval/index_map.json")
    print("  git commit -m 'add prebuilt faiss index'")
    print("  git push")


if __name__ == "__main__":
    main()