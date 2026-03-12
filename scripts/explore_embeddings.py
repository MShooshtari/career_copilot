"""
Explore the Chroma RAG store: list collections, count, sample documents, and similarity search.

Uses OpenAI embeddings for both jobs and user_profiles (career_copilot.rag.embedding).
Set OPENAI_API_KEY in .env.

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
JOBS_COLLECTION = "jobs"
USER_PROFILES_COLLECTION = "user_profiles"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chromadb

from career_copilot.rag.embedding import get_embedding_function


def _get_collection(client: chromadb.PersistentClient, name: str):
    ef = get_embedding_function()
    try:
        return client.get_or_create_collection(name, embedding_function=ef)
    except ValueError as e:
        if "embedding function" in str(e).lower() and "conflict" in str(e).lower():
            client.delete_collection(name=name)
            return client.get_or_create_collection(name, embedding_function=ef)
        raise


def _show_collection(
    coll, name: str, id_key: str = "title", meta_company: str = "company"
) -> int:
    n = coll.count()
    print(f"=== {name} (count: {n}) ===\n")
    if n == 0:
        print("  (empty)\n")
        return 0

    sample = coll.get(limit=min(5, n), include=["documents", "metadatas"])
    ids = sample.get("ids") or [f"#{j}" for j in range(len(sample["documents"]))]
    for i, (doc_id, doc, meta) in enumerate(
        zip(ids, sample["documents"], sample["metadatas"]), 1
    ):
        title = meta.get(id_key, doc_id)
        extra = f" | {meta.get(meta_company, '')}" if meta_company else ""
        print(f"  [{i}] id={doc_id} | {title}{extra}")
        snippet = (doc[:200] + "...") if len(doc) > 200 else doc
        print(f"      {snippet}")
        print()
    one = coll.get(limit=1, include=["embeddings"])
    embs = one.get("embeddings")
    if embs and len(embs) > 0:
        print(f"  Embedding dimension: {len(embs[0])}\n")
    return n


def main() -> None:
    if not CHROMA_PATH.exists():
        print(f"Chroma path not found: {CHROMA_PATH}")
        print("Run: python scripts/run_rag_index.py (jobs) or save a profile in the web app (user_profiles).")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # List what collections exist
    try:
        existing = client.list_collections()
        print("Collections in Chroma:", [c.name for c in existing], "\n")
    except Exception:
        existing = []

    # Show jobs collection
    coll_jobs = _get_collection(client, JOBS_COLLECTION)
    n_jobs = _show_collection(coll_jobs, JOBS_COLLECTION)

    # Show user_profiles collection
    coll_users = _get_collection(client, USER_PROFILES_COLLECTION)
    n_users = _show_collection(
        coll_users, USER_PROFILES_COLLECTION, id_key="user_id", meta_company=""
    )

    # Similarity search on jobs (only if non-empty)
    query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "remote Python backend developer"
    if n_jobs > 0:
        print(f"--- Query (jobs): \"{query_text}\" (top 5) ---")
        results = coll_jobs.query(
            query_texts=[query_text],
            n_results=min(5, n_jobs),
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
    else:
        print(f"--- Query skipped (jobs collection is empty). Run: python scripts/run_rag_index.py ---")

    print("Done.")


# # Default query: "remote Python backend developer"
# python scripts/explore_embeddings.py

# # Custom query
# python scripts/explore_embeddings.py "remote Python backend"
# python scripts/explore_embeddings.py "data engineer"

if __name__ == "__main__":
    main()
