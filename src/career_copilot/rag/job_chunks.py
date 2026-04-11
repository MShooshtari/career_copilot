"""Persist description chunks + embeddings for RAG (pgvector)."""

from __future__ import annotations

import psycopg

try:
    from pgvector.psycopg import register_vector
except ModuleNotFoundError:  # pragma: no cover

    def register_vector(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError("pgvector is required (pip install pgvector)")


from career_copilot.ingestion.common import NormalizedJob
from career_copilot.rag.chunk_text import chunk_description
from career_copilot.rag.embedding import embed_texts


def delete_job_chunks(conn: psycopg.Connection, job_id: int) -> None:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM job_description_chunks WHERE job_id = %s", (int(job_id),))
    conn.commit()


def rebuild_job_chunks_for_job(conn: psycopg.Connection, job: NormalizedJob) -> int:
    """
    Replace all chunks for ``job.db_id`` from ``job.description``.

    Returns number of chunk rows inserted (0 if no description).
    """
    if job.db_id is None:
        raise ValueError("rebuild_job_chunks_for_job requires NormalizedJob.db_id")
    register_vector(conn)
    jid = int(job.db_id)
    desc = (job.description or "").strip()

    with conn.cursor() as cur:
        cur.execute("DELETE FROM job_description_chunks WHERE job_id = %s", (jid,))
        if not desc:
            conn.commit()
            return 0

        parts = chunk_description(desc)
        if not parts:
            conn.commit()
            return 0

        embeddings = embed_texts(parts)
        if len(embeddings) != len(parts):
            raise RuntimeError("Chunk embedding count mismatch")

        for idx, (content, emb) in enumerate(zip(parts, embeddings, strict=True)):
            cur.execute(
                """
                INSERT INTO job_description_chunks (job_id, chunk_index, content, embedding, updated_at)
                VALUES (%s, %s, %s, %s::vector, now())
                """,
                (jid, idx, content, emb),
            )
    conn.commit()
    return len(parts)
