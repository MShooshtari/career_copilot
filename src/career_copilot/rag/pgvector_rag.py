"""
pgvector in PostgreSQL: job and user profile embeddings (cosine / HNSW).

Requires: CREATE EXTENSION vector; jobs_embeddings and user_embeddings (see database.schema).
Embeddings: OpenAI via career_copilot.rag.embedding.embed_texts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg

try:
    from pgvector.psycopg import register_vector
except ModuleNotFoundError:  # pragma: no cover
    # Allows importing this module in minimal test envs without pgvector installed.
    def register_vector(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError("pgvector is required (pip install pgvector)")


from career_copilot.constants import (
    RAG_DEFAULT_RECOMMENDATION_N_RESULTS,
    RAG_JOB_DOC_MAX_CHARS,
    RAG_JOB_UPSERT_BATCH_SIZE,
    RAG_SIMILAR_JOBS_N_RESULTS,
    RAG_SIMILAR_RESUMES_N_RESULTS,
)
from career_copilot.database.db import load_env
from career_copilot.ingestion.common import NormalizedJob
from career_copilot.rag.embedding import embed_texts
from career_copilot.rag.job_document import job_to_document, job_to_metadata

load_env()


def _register_vector(conn: psycopg.Connection) -> None:
    register_vector(conn)


def _row_to_job_hit(
    row: tuple[Any, ...],
) -> dict[str, Any]:
    """Build vector-hit dict matching format_recommendation_jobs / resume improvement."""
    (
        jid,
        source,
        source_id,
        title,
        company,
        location,
        salary_min,
        salary_max,
        description,
        skills,
        posted_at,
        url,
        distance,
    ) = row
    job = NormalizedJob(
        source=source or "",
        source_id=source_id,
        title=title,
        company=company,
        location=location,
        salary_min=salary_min,
        salary_max=salary_max,
        description=description,
        skills=list(skills) if skills else None,
        posted_at=posted_at,
        url=url,
        raw={},
        db_id=int(jid),
    )
    doc = job_to_document(job)
    meta = job_to_metadata(job)
    dist: float | None = None
    if distance is not None:
        try:
            dist = float(distance)
        except (TypeError, ValueError):
            dist = None
    return {
        "id": str(jid),
        "postgres_job_id": int(jid),
        "metadata": meta,
        "document": doc,
        "distance": dist,
    }


def vector_search_user_profiles(
    conn: psycopg.Connection,
    vector: list[float],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Nearest user profile rows by embedding (for scripts/explore_embeddings)."""
    _register_vector(conn)
    k = max(1, top_k)
    sql = """
        SELECT user_id, content,
               (embedding <=> %(q)s::vector) AS distance
        FROM user_embeddings
        ORDER BY embedding <=> %(q)s::vector
        LIMIT %(k)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"q": vector, "k": k})
        rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        uid, content, distance = row[0], row[1], row[2]
        try:
            uid_int = int(uid)
        except (TypeError, ValueError):
            uid_int = uid
        dist: float | None = None
        if distance is not None:
            try:
                dist = float(distance)
            except (TypeError, ValueError):
                dist = None
        out.append(
            {
                "id": f"user:{uid}",
                "metadata": {"user_id": uid_int},
                "document": content or "",
                "distance": dist,
            }
        )
    return out


def vector_search_jobs(
    conn: psycopg.Connection,
    vector: list[float],
    *,
    top_k: int,
    exclude_disliked_by_user: int | None = None,
) -> list[dict[str, Any]]:
    _register_vector(conn)
    k = max(1, top_k)
    params: dict[str, Any] = {"q": vector, "k": k}
    dislike_filter = ""
    if exclude_disliked_by_user is not None:
        dislike_filter = """
        WHERE NOT EXISTS (
            SELECT 1
            FROM user_job_interaction jf
            WHERE jf.user_id = %(exclude_user_id)s
              AND jf.job_source = 'ingested'
              AND jf.job_id = j.id
              AND jf.feedback = 'dislike'
        )
        """
        params["exclude_user_id"] = exclude_disliked_by_user
    sql = f"""
        SELECT j.id, j.source, j.source_id, j.title, j.company, j.location,
               j.salary_min, j.salary_max, j.description, j.skills, j.posted_at, j.url,
               (e.embedding <=> %(q)s::vector) AS distance
        FROM jobs_embeddings e
        JOIN jobs j ON j.id = e.job_id
        {dislike_filter}
        ORDER BY e.embedding <=> %(q)s::vector
        LIMIT %(k)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [_row_to_job_hit(tuple(r)) for r in rows]


def index_jobs_into_pgvector(conn: psycopg.Connection, jobs: list[NormalizedJob]) -> int:
    """
    Compute embeddings for normalized jobs and UPSERT into jobs_embeddings.

    Each job must have ``db_id`` set (Postgres ``jobs.id``).
    """
    if not jobs:
        return 0

    _register_vector(conn)
    batch_docs: list[tuple[int, str]] = []
    for job in jobs:
        if job.db_id is None:
            raise ValueError("index_jobs_into_pgvector requires NormalizedJob.db_id (Postgres id)")
        doc = job_to_document(job)
        if not doc.strip():
            continue
        batch_docs.append((int(job.db_id), doc))

    if not batch_docs:
        return 0

    total = 0
    for i in range(0, len(batch_docs), RAG_JOB_UPSERT_BATCH_SIZE):
        print(f"Processing batch {i} of {len(batch_docs)}")
        chunk = batch_docs[i : i + RAG_JOB_UPSERT_BATCH_SIZE]
        texts = [c[1] for c in chunk]
        embeddings = embed_texts(texts)
        if len(embeddings) != len(chunk):
            raise RuntimeError("Embedding count mismatch")
        with conn.cursor() as cur:
            for (jid, doc), emb in zip(chunk, embeddings, strict=True):
                cur.execute(
                    """
                    INSERT INTO jobs_embeddings (job_id, content, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (job_id) DO UPDATE SET
                      content = EXCLUDED.content,
                      embedding = EXCLUDED.embedding
                    """,
                    (jid, doc, emb),
                )
                total += 1
        # Persist each batch promptly (important for job/worker runs and debugging).
        conn.commit()
    return total


def upsert_user_profile_embedding(conn: psycopg.Connection, user_id: int, content: str) -> None:
    """Store profile text and embedding for ``user_id`` (overwrites if exists)."""
    if not content.strip():
        return
    _register_vector(conn)
    vec = embed_texts([content])[0]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_embeddings (user_id, content, embedding)
            VALUES (%s, %s, %s::vector)
            ON CONFLICT (user_id) DO UPDATE SET
              content = EXCLUDED.content,
              embedding = EXCLUDED.embedding
            """,
            (user_id, content, vec),
        )


def fetch_user_profile_embedding(conn: psycopg.Connection, user_id: int) -> list[float] | None:
    """Return stored embedding for ``user_id``, or None."""
    _register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding FROM user_embeddings WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    if not row or row[0] is None:
        return None
    emb = row[0]
    return list(emb)


def get_recommended_job_results(
    conn: psycopg.Connection,
    *,
    user_id: int,
    n_results: int = RAG_DEFAULT_RECOMMENDATION_N_RESULTS,
    jobs_collection_name: str = "jobs",
) -> list[dict[str, Any]]:
    _ = jobs_collection_name
    user_embedding = fetch_user_profile_embedding(conn, user_id)
    if user_embedding is None:
        return []
    k = max(1, n_results)
    return vector_search_jobs(conn, user_embedding, top_k=k, exclude_disliked_by_user=user_id)


def get_similar_jobs_for_resume_improvement(
    conn: psycopg.Connection,
    job_document: str,
    *,
    n_results: int = RAG_SIMILAR_JOBS_N_RESULTS,
    jobs_collection_name: str = "jobs",
) -> list[dict[str, Any]]:
    _ = jobs_collection_name
    if not job_document or not job_document.strip():
        return []
    vec = embed_texts([job_document.strip()])[0]
    k = max(1, n_results)
    return vector_search_jobs(conn, vec, top_k=k)


def get_similar_resumes_for_resume_improvement(
    conn: psycopg.Connection,
    job_document: str,
    *,
    exclude_user_id: int | None = None,
    n_results: int = RAG_SIMILAR_RESUMES_N_RESULTS,
    persist_path: str | Path | None = None,
    user_profiles_collection_name: str = "user_profiles",
) -> list[dict[str, Any]]:
    _ = persist_path, user_profiles_collection_name
    if not job_document or not job_document.strip():
        return []

    _register_vector(conn)
    text = job_document[:RAG_JOB_DOC_MAX_CHARS].strip()
    vec = embed_texts([text])[0]
    want = max(1, n_results)
    fetch_k = min(want + (1 if exclude_user_id is not None else 0), 50)

    if exclude_user_id is not None:
        sql = """
            SELECT user_id, content,
                   (embedding <=> %(q)s::vector) AS distance
            FROM user_embeddings
            WHERE user_id <> %(ex)s
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(k)s
        """
        params: dict[str, Any] = {"q": vec, "ex": exclude_user_id, "k": fetch_k}
    else:
        sql = """
            SELECT user_id, content,
                   (embedding <=> %(q)s::vector) AS distance
            FROM user_embeddings
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(k)s
        """
        params = {"q": vec, "k": fetch_k}

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        uid, content, distance = row[0], row[1], row[2]
        try:
            uid_int = int(uid)
        except (TypeError, ValueError):
            uid_int = uid
        meta = {"user_id": uid_int}
        dist: float | None = None
        if distance is not None:
            try:
                dist = float(distance)
            except (TypeError, ValueError):
                dist = None
        out.append(
            {
                "id": f"user:{uid}",
                "metadata": meta,
                "document": content or "",
                "distance": dist,
            }
        )
        if len(out) >= want:
            break
    return out
