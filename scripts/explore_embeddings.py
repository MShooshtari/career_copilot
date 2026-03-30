"""
Explore RAG stores: jobs and user profiles in Azure AI Search.

Uses OpenAI embeddings. Set OPENAI_API_KEY, and AZURE_SEARCH_* (see configs/config.example.env).

Usage:
  python scripts/explore_embeddings.py
  python scripts/explore_embeddings.py "remote Python backend"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from career_copilot.database.db import connect, load_env
from career_copilot.rag.azure_search_jobs import (
    CONTENT_FIELD,
    azure_search_configured,
    get_jobs_search_client,
    vector_search_jobs,
)
from career_copilot.rag.azure_search_users import (
    USER_CONTENT_FIELD,
    USER_KEY_FIELD,
    get_user_profiles_search_client,
    user_profiles_search_configured,
    vector_search_user_profiles,
)
from career_copilot.rag.embedding import EMBEDDING_VECTOR_DIMENSIONS, embed_texts
from career_copilot.resume_io import extract_resume_text

RESUME_SNIPPET_CHARS = 800


def _fetch_stored_resumes() -> list[tuple[int, str | None, str]]:
    """Fetch (user_id, filename, extracted_text) for all profiles that have a resume file."""
    out: list[tuple[int, str | None, str]] = []
    try:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, resume_filename, resume_file FROM profiles WHERE resume_file IS NOT NULL AND resume_file != ''"
            )
            for row in cur.fetchall():
                user_id, filename, content_bytes = row[0], row[1], bytes(row[2]) if row[2] else b""
                text = extract_resume_text(content_bytes, filename)
                out.append((user_id, filename, text))
        conn.close()
    except Exception as e:
        print(f"  (Could not load resumes from DB: {e})\n")
    return out


def _explore_azure_jobs(query_text: str) -> int:
    """Print Azure AI Search jobs sample and optional vector query. Returns document count or 0."""
    print("=== Jobs (Azure AI Search) ===\n")
    try:
        client = get_jobs_search_client()
    except Exception as e:
        print(f"  Could not connect: {e}\n")
        return 0

    total = 0
    try:
        page = client.search(
            search_text="*",
            include_total_count=True,
            top=5,
            select=["job_id", "title", "company", "location", CONTENT_FIELD],
        )
        total = int(page.get_count() or 0)
        print(f"  Document count: {total}\n")
        for i, doc in enumerate(page, 1):
            d = dict(doc)
            title = d.get("title") or ""
            company = d.get("company") or ""
            jid = d.get("job_id") or ""
            content = d.get(CONTENT_FIELD) or ""
            snippet = (content[:200] + "...") if len(content) > 200 else content
            print(f"  [{i}] job_id={jid} | {title} @ {company}")
            print(f"      {snippet}\n")
    except Exception as e:
        print(f"  Search failed (is the index created? run scripts/rag_index/run.py): {e}\n")
        return 0

    if total <= 0:
        print("  Run: python scripts/rag_index/run.py\n")
        return 0

    print(f'--- Vector query (jobs): "{query_text}" (top 5) ---')
    try:
        vec = embed_texts([query_text])[0]
        hits = vector_search_jobs(vec, top_k=min(5, max(1, total)))
        for h in hits:
            meta = h.get("metadata") or {}
            dist = h.get("distance")
            doc = (h.get("document") or "")[:180]
            dist_s = f"{dist:.6f}" if dist is not None else "n/a"
            print(f"  distance~={dist_s} | {meta.get('title', '')} @ {meta.get('company', '')}")
            print(f"  {doc}{'...' if len(h.get('document') or '') > 180 else ''}\n")
    except Exception as e:
        print(f"  Vector query failed: {e}\n")

    return int(total)


def _explore_azure_user_profiles(query_text: str) -> int:
    """Print user profile index sample and optional vector query."""
    print("=== User profiles (Azure AI Search) ===\n")
    try:
        client = get_user_profiles_search_client()
    except Exception as e:
        print(f"  Could not connect: {e}\n")
        return 0

    total = 0
    try:
        page = client.search(
            search_text="*",
            include_total_count=True,
            top=5,
            select=[USER_KEY_FIELD, USER_CONTENT_FIELD],
        )
        total = int(page.get_count() or 0)
        print(f"  Document count: {total}\n")
        for i, doc in enumerate(page, 1):
            d = dict(doc)
            uid = d.get(USER_KEY_FIELD) or ""
            content = d.get(USER_CONTENT_FIELD) or ""
            snippet = (content[:200] + "...") if len(content) > 200 else content
            print(f"  [{i}] user_id={uid}")
            print(f"      {snippet}\n")
    except Exception as e:
        print(
            f"  Search failed (save a profile in the app or ensure the user index exists): {e}\n"
        )
        return 0

    if total > 0:
        print(f"  Vector field size: {EMBEDDING_VECTOR_DIMENSIONS} dimensions\n")

    if total <= 0:
        print(
            "  No documents yet. Set AZURE_SEARCH_* and OPENAI_API_KEY, then save a profile in the web app.\n"
        )
        return 0

    print(f'--- Vector query (user profiles): "{query_text}" (top 5) ---')
    try:
        vec = embed_texts([query_text])[0]
        hits = vector_search_user_profiles(vec, top_k=min(5, max(1, total)))
        for h in hits:
            meta = h.get("metadata") or {}
            dist = h.get("distance")
            doc = (h.get("document") or "")[:180]
            dist_s = f"{dist:.6f}" if dist is not None else "n/a"
            print(f"  distance~={dist_s} | user_id={meta.get('user_id', '')}")
            print(f"  {doc}{'...' if len(h.get('document') or '') > 180 else ''}\n")
    except Exception as e:
        print(f"  Vector query failed: {e}\n")

    return int(total)


def main() -> None:
    load_env()
    query_text = (
        " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "remote Python backend developer"
    )

    n_jobs = 0
    if azure_search_configured():
        n_jobs = _explore_azure_jobs(query_text)
    else:
        print(
            "=== Jobs (Azure AI Search) ===\n"
            "  Not configured. Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY in .env, "
            "then run: python scripts/rag_index/run.py\n"
        )

    n_users = 0
    if user_profiles_search_configured():
        n_users = _explore_azure_user_profiles(query_text)
        if n_users > 0:
            print(
                "--- Stored resumes (from DB; this text is included in the embedding when present) ---\n"
            )
            stored = _fetch_stored_resumes()
            if not stored:
                print(
                    "  No resume files stored in DB. Upload a resume in the profile page and save to include it in the embedding.\n"
                )
            for user_id, filename, text in stored:
                print(f"  user_id={user_id}  file={filename or '(unknown)'}")
                if text.strip():
                    snippet = (
                        (text[:RESUME_SNIPPET_CHARS] + "...")
                        if len(text) > RESUME_SNIPPET_CHARS
                        else text
                    )
                    print(f"  {snippet}")
                else:
                    print("  (no text extracted)")
                print()
    else:
        print(
            "=== User profiles (Azure AI Search) ===\n"
            "  Not configured. Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY in .env, "
            "then save a profile in the web app to index user embeddings.\n"
        )

    if (
        n_jobs == 0
        and n_users == 0
        and not azure_search_configured()
        and not user_profiles_search_configured()
    ):
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
