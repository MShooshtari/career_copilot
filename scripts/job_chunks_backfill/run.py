"""
Rebuild job_description_chunks for all jobs with a non-empty description.

Requires: POSTGRES_* or POSTGRES_DSN, OPENAI_API_KEY.

Run after sql/005 or init_schema created the chunk table. Normally chunks are updated
by the job embeddings worker when descriptions change; this script backfills existing rows.
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

LOAD_JOBS_SQL = """
SELECT id, source, source_id, title, company, location,
       salary_min, salary_max, description, skills, extracted_skills,
       posted_at, url, raw
FROM jobs
WHERE description IS NOT NULL AND btrim(description) <> ''
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
        posted_at=posted_at,
        url=url,
        raw=raw_dict,
        db_id=int(db_id),
    )


def main() -> None:
    load_env()
    target_db = os.environ.get("POSTGRES_DB") or "career_copilot"
    total_chunks = 0
    n_jobs = 0
    with connect(dbname=target_db) as conn:
        init_schema(conn)
        with conn.cursor() as cur:
            cur.execute(LOAD_JOBS_SQL)
            rows = cur.fetchall()

        for row in rows:
            job = _row_to_normalized_job(row)
            n = rebuild_job_chunks_for_job(conn, job)
            total_chunks += n
            n_jobs += 1
            if n_jobs % 50 == 0:
                print(f"[job-chunks-backfill] processed {n_jobs} job(s), {total_chunks} chunk(s)…")

    print(
        f"Job chunks backfill: {n_jobs} job(s), {total_chunks} chunk row(s) "
        f"(db={target_db})"
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
