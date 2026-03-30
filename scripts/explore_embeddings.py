"""
Explore pgvector embeddings: jobs and user profiles in PostgreSQL.

Requires: OPENAI_API_KEY, Postgres (see configs/config.example.env), and
``CREATE EXTENSION vector`` + schema from career_copilot.database.schema.init_schema.

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

import psycopg

from career_copilot.database.db import connect, load_env
from career_copilot.rag.embedding import EMBEDDING_VECTOR_DIMENSIONS, embed_texts
from career_copilot.rag.pgvector_rag import (
    vector_search_jobs,
    vector_search_user_profiles,
)
from career_copilot.resume_io import extract_resume_text

RESUME_SNIPPET_CHARS = 800


def _explore_jobs(conn: psycopg.Connection, query_text: str) -> int:
    from pgvector.psycopg import register_vector

    register_vector(conn)
    print("=== Jobs (pgvector on jobs_embeddings.embedding) ===\n")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FROM jobs_embeddings
            """
        )
        n = int(cur.fetchone()[0])
    print(f"  Rows with embeddings: {n}\n")
    if n <= 0:
        print("  Run: python scripts/job_embeddings_backfill/run.py after ingesting jobs.\n")
        return 0

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.id, left(
                coalesce(title, '') || ' @ ' || coalesce(company, ''),
                80
            )
            FROM jobs_embeddings e
            JOIN jobs j ON j.id = e.job_id
            ORDER BY j.id
            LIMIT 5
            """
        )
        for i, row in enumerate(cur.fetchall(), 1):
            print(f"  [{i}] job id={row[0]} | {row[1]}")

    print(f"\n  Vector dimensions: {EMBEDDING_VECTOR_DIMENSIONS}\n")

    print(f'--- Vector query (jobs): "{query_text}" (top 5) ---')
    try:
        vec = embed_texts([query_text])[0]
        hits = vector_search_jobs(conn, vec, top_k=min(5, max(1, n)))
        for h in hits:
            meta = h.get("metadata") or {}
            dist = h.get("distance")
            doc = (h.get("document") or "")[:180]
            dist_s = f"{dist:.6f}" if dist is not None else "n/a"
            print(f"  distance~={dist_s} | {meta.get('title', '')} @ {meta.get('company', '')}")
            print(f"  {doc}{'...' if len(h.get('document') or '') > 180 else ''}\n")
    except Exception as e:
        print(f"  Vector query failed: {e}\n")

    return n


def _explore_user_profiles(conn: psycopg.Connection, query_text: str) -> int:
    from pgvector.psycopg import register_vector

    register_vector(conn)
    print("=== User profiles (pgvector on user_embeddings.embedding) ===\n")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM user_embeddings")
        n = int(cur.fetchone()[0])
    print(f"  Rows: {n}\n")
    if n <= 0:
        print("  Save a profile in the web app (with OPENAI_API_KEY set) to index.\n")
        return 0

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT user_id, left(coalesce(content, ''), 200)
            FROM user_embeddings
            ORDER BY user_id
            LIMIT 5
            """
        )
        for i, row in enumerate(cur.fetchall(), 1):
            print(f"  [{i}] user_id={row[0]}")
            print(f"      {row[1]}{'...' if len(row[1] or '') >= 200 else ''}\n")

    print(f'--- Vector query (user profiles): "{query_text}" (top 5) ---')
    try:
        vec = embed_texts([query_text])[0]
        hits = vector_search_user_profiles(
            conn, vec, top_k=min(5, max(1, n))
        )
        for h in hits:
            meta = h.get("metadata") or {}
            dist = h.get("distance")
            doc = (h.get("document") or "")[:180]
            dist_s = f"{dist:.6f}" if dist is not None else "n/a"
            print(f"  distance~={dist_s} | user_id={meta.get('user_id', '')}")
            print(f"  {doc}{'...' if len(h.get('document') or '') > 180 else ''}\n")
    except Exception as e:
        print(f"  Vector query failed: {e}\n")

    return n


def _fetch_stored_resumes() -> list[tuple[int, str | None, str]]:
    out: list[tuple[int, str | None, str]] = []
    try:
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id, resume_filename, resume_file FROM profiles WHERE resume_file IS NOT NULL AND resume_file != ''"
                )
                for row in cur.fetchall():
                    user_id, filename, content_bytes = row[0], row[1], bytes(row[2]) if row[2] else b""
                    text = extract_resume_text(content_bytes, filename)
                    out.append((user_id, filename, text))
        finally:
            conn.close()
    except Exception as e:
        print(f"  (Could not load resumes from DB: {e})\n")
    return out


def main() -> None:
    load_env()
    query_text = (
        " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "remote Python backend developer"
    )

    n_jobs = 0
    n_users = 0
    try:
        with connect(dbname="career_copilot") as conn:
            n_jobs = _explore_jobs(conn, query_text)
            n_users = _explore_user_profiles(conn, query_text)
    except Exception as e:
        print(f"Database error: {e}\n")
        sys.exit(1)

    if n_users > 0:
        print(
            "--- Stored resumes (from DB; included in profile embedding when present) ---\n"
        )
        stored = _fetch_stored_resumes()
        if not stored:
            print("  No resume files in DB.\n")
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

    if n_jobs == 0 and n_users == 0:
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
