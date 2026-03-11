"""
Explore the Chroma RAG store: count, sample documents, and similarity search.

Usage:
  python scripts/explore_embeddings.py
  python scripts/explore_embeddings.py "remote Python backend"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
CHROMA_PATH = ROOT / "data" / "chroma"
COLLECTION_NAME = "jobs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chromadb


def main() -> None:
    if not CHROMA_PATH.exists():
        print(f"Chroma path not found: {CHROMA_PATH}")
        print("Run: python scripts/run_rag_index.py")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    coll = client.get_collection(COLLECTION_NAME)

    # Count
    n = coll.count()
    print(f"Total jobs in collection: {n}\n")

    # Sample documents + metadata
    print("--- Sample (first 3) ---")
    sample = coll.get(limit=3, include=["documents", "metadatas"])
    for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"]), 1):
        title = meta.get("title", "")
        company = meta.get("company", "")
        print(f"[{i}] {title} | {company}")
        snippet = (doc[:200] + "...") if len(doc) > 200 else doc
        print(f"    {snippet}")
        print()
    print()

    # Embedding dimension (one vector)
    one = coll.get(limit=1, include=["embeddings"])
    embs = one.get("embeddings")
    if embs is not None and len(embs) > 0:
        dim = len(embs[0])
        print(f"Embedding dimension: {dim}\n")

    # Similarity search
    query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "remote Python backend developer"
    print(f"--- Query: \"{query_text}\" (top 5) ---")
    results = coll.query(
        query_texts=[query_text],
        n_results=min(5, n),
        include=["documents", "metadatas", "distances"],
    )
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  distance={dist:.4f} | {meta.get('title', '')} @ {meta.get('company', '')}")
        print(f"  {doc[:180]}{'...' if len(doc) > 180 else ''}")
        print()

    print("Done.")


# # Default query: "remote Python backend developer"
# python scripts/explore_embeddings.py

# # Custom query
# python scripts/explore_embeddings.py "remote Python backend"
# python scripts/explore_embeddings.py "data engineer"

if __name__ == "__main__":
    main()
