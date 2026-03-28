"""
Index job listings from the database into Chroma for RAG (local run).

Uses OpenAI embeddings (career_copilot.rag.embedding). Set OPENAI_API_KEY in .env.

Run after ingestion so that the RAG store is populated from the jobs table.
Does not modify the existing ingestion or DB schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from career_copilot.database.db import load_env, connect  # noqa: E402
from career_copilot.ingestion.common import NormalizedJob  # noqa: E402
from career_copilot.rag.chroma_store import index_jobs_into_chroma  # noqa: E402

LOAD_JOBS_SQL = """
SELECT source, source_id, title, company, location,
       salary_min, salary_max, description, skills,
       posted_at, url, raw
FROM jobs
ORDER BY id;
"""


def _row_to_normalized_job(row: tuple) -> NormalizedJob:
    (
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
    )


def main() -> None:
    load_env()
    persist_path = ROOT / "data" / "chroma"

    with connect(dbname="career_copilot") as conn:
        with conn.cursor() as cur:
            cur.execute(LOAD_JOBS_SQL)
            rows = cur.fetchall()

    jobs = [_row_to_normalized_job(r) for r in rows]
    count = index_jobs_into_chroma(
        jobs,
        persist_path=persist_path,
        collection_name="jobs",
    )
    print(f"RAG index: {count} job(s) indexed into Chroma at {persist_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
