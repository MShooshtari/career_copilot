"""
Index job listings into Postgres pgvector (jobs_embeddings) for RAG.

Requires: POSTGRES_* or POSTGRES_DSN, OPENAI_API_KEY (see configs/config.example.env).
Job embeddings are stored in Postgres (pgvector in ``jobs_embeddings``).

Run after ingestion so the search index matches the jobs table.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from career_copilot.database.db import connect, load_env  # noqa: E402
from career_copilot.database.schema import init_schema  # noqa: E402
from career_copilot.ingestion.common import NormalizedJob  # noqa: E402
from career_copilot.rag.job_chunks import rebuild_job_chunks_for_job  # noqa: E402
from career_copilot.rag.pgvector_rag import index_jobs_into_pgvector  # noqa: E402

LOAD_JOBS_SQL = """
SELECT id, source, source_id, title, company, location,
       salary_min, salary_max, description, skills, extracted_skills, ai_extracted_skills,
       posted_at, url, raw
FROM jobs
ORDER BY id;
"""


def _row_to_normalized_job(row: tuple) -> NormalizedJob:
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
        extracted_skills,
        ai_extracted_skills,
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
        extracted_skills=list(extracted_skills) if extracted_skills else None,
        ai_extracted_skills=list(ai_extracted_skills) if ai_extracted_skills else None,
        posted_at=posted_at,
        url=url,
        raw=raw_dict,
        db_id=int(db_id),
    )


def main() -> None:
    load_env()
    target_db = os.environ.get("POSTGRES_DB") or "career_copilot"
    with connect(dbname=target_db) as conn:
        init_schema(conn)
        with conn.cursor() as cur:
            cur.execute(LOAD_JOBS_SQL)
            rows = cur.fetchall()

        jobs = [_row_to_normalized_job(r) for r in rows]
        count = index_jobs_into_pgvector(conn, jobs)
        chunk_rows = 0
        print(f"[job-embeddings-backfill] rebuilding chunks for {len(jobs)} job(s)")
        for idx, job in enumerate(jobs, start=1):
            try:
                chunk_rows += rebuild_job_chunks_for_job(conn, job)
            except Exception as ex:
                print(f"[job-embeddings-backfill] chunk rebuild skipped job_id={job.db_id}: {ex}")
            if idx % 50 == 0 or idx == len(jobs):
                print(
                    f"[job-embeddings-backfill] rebuilt chunks for {idx}/{len(jobs)} job(s), "
                    f"{chunk_rows} chunk row(s)"
                )
        print(
            f"Job embeddings backfill: {count} job(s) upserted into jobs_embeddings; "
            f"{chunk_rows} chunk row(s) in job_description_chunks "
            f"(db={target_db}, total_jobs={len(jobs)})"
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
