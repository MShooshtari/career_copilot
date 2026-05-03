"""
Backfill jobs.extracted_skills and jobs.ai_extracted_skills from existing job descriptions.

Run after the jobs table has the extracted_skills column:

  python scripts/job_skills_backfill/run.py
  python scripts/job_skills_backfill/run.py --skip-existing-ai-extracted-skills
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from career_copilot.database.db import connect, load_env  # noqa: E402
from career_copilot.database.schema import init_schema  # noqa: E402
from career_copilot.ingestion.skill_extraction import (  # noqa: E402
    extract_ai_skill_tags,
    extract_skill_tags,
)

LOAD_JOBS_SQL = """
SELECT id, description, skills
FROM jobs
WHERE description IS NOT NULL AND btrim(description) <> ''
ORDER BY id;
"""

LOAD_JOBS_MISSING_AI_SKILLS_SQL = """
SELECT id, description, skills
FROM jobs
WHERE description IS NOT NULL
  AND btrim(description) <> ''
  AND (ai_extracted_skills IS NULL OR array_length(ai_extracted_skills, 1) IS NULL)
ORDER BY id;
"""

UPDATE_SQL = """
UPDATE jobs
SET extracted_skills = %s,
    ai_extracted_skills = %s,
    updated_at = now()
WHERE id = %s;
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill jobs.extracted_skills and jobs.ai_extracted_skills."
    )
    parser.add_argument(
        "--skip-existing-ai-extracted-skills",
        action="store_true",
        help="Skip jobs where ai_extracted_skills already contains at least one skill.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env()
    target_db = os.environ.get("POSTGRES_DB") or "career_copilot"
    processed = 0
    updated = 0

    with connect(dbname=target_db) as conn:
        init_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                LOAD_JOBS_MISSING_AI_SKILLS_SQL
                if args.skip_existing_ai_extracted_skills
                else LOAD_JOBS_SQL
            )
            rows = cur.fetchall()

        with conn.cursor() as cur:
            ai_skill_extraction_available = True
            for job_id, description, skills in rows:
                extracted_skills = extract_skill_tags(description, source_skills=skills)
                ai_extracted_skills = None
                if ai_skill_extraction_available:
                    try:
                        ai_extracted_skills = extract_ai_skill_tags(description)
                    except RuntimeError as e:
                        print(f"AI skill extraction disabled: {e}")
                        ai_skill_extraction_available = False
                    except Exception as e:
                        print(f"AI skill extraction failed for job_id={job_id}: {e}")
                cur.execute(
                    UPDATE_SQL,
                    (extracted_skills or None, ai_extracted_skills or None, job_id),
                )
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
