"""
Drain jobs_embedding_queue and UPSERT into jobs_embeddings (pgvector).

Intended for "triggered" embedding updates in online deployments:
- A Postgres trigger enqueues job ids on INSERT/UPDATE(description) in jobs.
- This worker runs as a background job (e.g. Azure Container Apps Job) and
  processes pending rows, computing embeddings via OpenAI and upserting into pgvector.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg


def _project_root() -> Path:
    env = os.environ.get("CAREER_COPILOT_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    # scripts/job_embeddings_worker/run.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from career_copilot.database.db import connect, load_env  # noqa: E402
from career_copilot.database.schema import init_schema  # noqa: E402
from career_copilot.ingestion.common import NormalizedJob  # noqa: E402
from career_copilot.rag.job_document import job_to_document  # noqa: E402
from career_copilot.rag.pgvector_rag import index_jobs_into_pgvector  # noqa: E402


CLAIM_LIMIT_DEFAULT = 50


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _claim_pending(
    conn: psycopg.Connection,
    *,
    limit: int,
    worker_id: str,
) -> list[tuple[int, str]]:
    """
    Claim up to `limit` rows and mark them processing.
    Returns list of (job_id, description_hash) claimed.
    """
    claimed: list[tuple[int, str]] = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT job_id, description_hash
            FROM jobs_embedding_queue
            WHERE status = 'pending'
            ORDER BY requested_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
        if not rows:
            return []
        for job_id, h in rows:
            claimed.append((int(job_id), str(h)))
        cur.executemany(
            """
            UPDATE jobs_embedding_queue
            SET status = 'processing',
                locked_at = now(),
                locked_by = %s,
                attempts = attempts + 1,
                last_error = NULL
            WHERE job_id = %s
              AND description_hash = %s
            """,
            [(worker_id, jid, h) for jid, h in claimed],
        )
    conn.commit()
    return claimed


def _load_jobs(conn: psycopg.Connection, job_ids: list[int]) -> dict[int, tuple]:
    if not job_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source, source_id, title, company, location,
                   salary_min, salary_max, description, skills,
                   posted_at, url, raw
            FROM jobs
            WHERE id = ANY(%s)
            """,
            (job_ids,),
        )
        rows = cur.fetchall()
    return {int(r[0]): tuple(r) for r in rows}


def _row_to_job(row: tuple) -> NormalizedJob:
    (
        db_id,
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
        raw,
    ) = row
    raw_dict = dict(raw) if raw else {}
    return NormalizedJob(
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
        raw=raw_dict,
        db_id=int(db_id),
    )


def _finalize_success(
    conn: psycopg.Connection,
    *,
    processed: list[tuple[int, str]],
    worker_id: str,
) -> int:
    if not processed:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            DELETE FROM jobs_embedding_queue
            WHERE job_id = %s
              AND description_hash = %s
              AND status = 'processing'
              AND locked_by = %s
            """,
            [(jid, h, worker_id) for jid, h in processed],
        )
    conn.commit()
    return len(processed)


def _finalize_error(
    conn: psycopg.Connection,
    *,
    job_id: int,
    description_hash: str,
    worker_id: str,
    error: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs_embedding_queue
            SET status = 'pending',
                locked_at = NULL,
                locked_by = NULL,
                last_error = %s
            WHERE job_id = %s
              AND description_hash = %s
              AND status = 'processing'
              AND locked_by = %s
            """,
            (error[:2000], job_id, description_hash, worker_id),
        )
    conn.commit()


def main() -> None:
    load_env()
    worker_id = (
        os.environ.get("EMBEDDING_WORKER_ID", "").strip()
        or os.environ.get("HOSTNAME", "").strip()
        or "embedding-worker"
    )
    claim_limit = max(1, _env_int("EMBEDDING_WORKER_CLAIM_LIMIT", CLAIM_LIMIT_DEFAULT))
    max_loops = _env_int("EMBEDDING_WORKER_MAX_LOOPS", 1)
    if max_loops <= 0:
        max_loops = 1

    with connect(dbname=os.environ.get("POSTGRES_DB") or "career_copilot") as conn:
        init_schema(conn)
        total_upserts = 0
        total_deleted = 0
        total_errors = 0

        for _ in range(max_loops):
            claimed = _claim_pending(conn, limit=claim_limit, worker_id=worker_id)
            if not claimed:
                break

            job_ids = [jid for jid, _h in claimed]
            rows_by_id = _load_jobs(conn, job_ids)

            to_index: list[NormalizedJob] = []
            processed: list[tuple[int, str]] = []

            for jid, h in claimed:
                row = rows_by_id.get(jid)
                if row is None:
                    # Job deleted between enqueue and processing.
                    processed.append((jid, h))
                    continue
                job = _row_to_job(row)
                doc = job_to_document(job)
                if not doc.strip():
                    # No content => delete embedding row (if any) and clear queue.
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM jobs_embeddings WHERE job_id = %s", (jid,))
                    conn.commit()
                    total_deleted += 1
                    processed.append((jid, h))
                    continue
                to_index.append(job)
                processed.append((jid, h))

            try:
                total_upserts += index_jobs_into_pgvector(conn, to_index)
                _finalize_success(conn, processed=processed, worker_id=worker_id)
            except Exception as e:
                total_errors += 1
                msg = str(e)
                # Requeue each claimed row; if any have changed hash since claim,
                # the UPDATE will no-op and the newer pending row remains.
                for jid, h in claimed:
                    _finalize_error(
                        conn,
                        job_id=jid,
                        description_hash=h,
                        worker_id=worker_id,
                        error=msg,
                    )
                break

    print("Embedding worker done.")


if __name__ == "__main__":
    main()

