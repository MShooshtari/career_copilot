"""
Index job listings into Postgres pgvector (jobs.embedding) for RAG.

Requires: POSTGRES_* or POSTGRES_DSN, OPENAI_API_KEY (see configs/config.example.env).
Job embeddings are stored in Postgres (pgvector on ``jobs.embedding``).

Run after ingestion so the search index matches the jobs table.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from career_copilot.database.db import load_env, connect  # noqa: E402
from career_copilot.database.schema import init_schema  # noqa: E402
from career_copilot.ingestion.common import NormalizedJob  # noqa: E402
from career_copilot.rag.pgvector_rag import index_jobs_into_pgvector  # noqa: E402

LOAD_JOBS_SQL = """
SELECT id, source, source_id, title, company, location,
       salary_min, salary_max, description, skills,
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


def main() -> None:
    load_env()
    with connect(dbname="career_copilot") as conn:
        init_schema(conn)
        with conn.cursor() as cur:
            cur.execute(LOAD_JOBS_SQL)
            rows = cur.fetchall()

        jobs = [_row_to_normalized_job(r) for r in rows]
        count = index_jobs_into_pgvector(conn, jobs)
        print(f"RAG index: {count} job(s) updated with embeddings in Postgres (pgvector)")


if __name__ == "__main__":
    main()
    sys.exit(0)
