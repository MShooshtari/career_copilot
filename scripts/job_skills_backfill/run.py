"""
Backfill jobs.extracted_skills from existing job descriptions.

Run after the jobs table has the extracted_skills column:

  python scripts/job_skills_backfill/run.py
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
from career_copilot.ingestion.skill_extraction import extract_skill_tags  # noqa: E402

LOAD_JOBS_SQL = """
SELECT id, description
FROM jobs
WHERE description IS NOT NULL AND btrim(description) <> ''
ORDER BY id;
"""

UPDATE_SQL = """
UPDATE jobs
SET extracted_skills = %s,
    updated_at = now()
WHERE id = %s;
"""


def main() -> None:
    load_env()
    target_db = os.environ.get("POSTGRES_DB") or "career_copilot"
    processed = 0
    updated = 0

    with connect(dbname=target_db) as conn:
        init_schema(conn)
        with conn.cursor() as cur:
            cur.execute(LOAD_JOBS_SQL)
            rows = cur.fetchall()

        with conn.cursor() as cur:
            for job_id, description in rows:
                extracted_skills = extract_skill_tags(description)
                cur.execute(UPDATE_SQL, (extracted_skills or None, job_id))
                processed += 1
                updated += int(bool(extracted_skills))
                if processed % 100 == 0:
                    conn.commit()
                    print(
                        f"[job-skills-backfill] processed {processed} job(s), "
                        f"{updated} with extracted skills"
                    )
        conn.commit()

    print(
        f"Job skills backfill: {processed} job(s) processed, "
        f"{updated} with extracted skills (db={target_db})"
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
