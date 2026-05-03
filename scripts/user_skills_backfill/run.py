"""
Backfill user_skills.ai_extracted_skills from stored resumes.

Requires: POSTGRES_* or POSTGRES_DSN, OPENAI_API_KEY.

Run after the user_skills table has the ai_extracted_skills column:

  python scripts/user_skills_backfill/run.py
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
from career_copilot.database.profiles import (  # noqa: E402
    get_resume_file_by_user_id,
    replace_user_skills,
)
from career_copilot.database.schema import init_schema  # noqa: E402
from career_copilot.ingestion.skill_extraction import extract_ai_resume_skill_tags  # noqa: E402
from career_copilot.resume_io import extract_resume_text  # noqa: E402
from career_copilot.utils import strip_nul  # noqa: E402

LOAD_PROFILES_SQL = """
SELECT user_id, skill_tags
FROM profiles
WHERE resume_file IS NOT NULL
   OR (resume_blob_container IS NOT NULL AND resume_blob_name IS NOT NULL)
ORDER BY user_id;
"""


def main() -> None:
    load_env()
    target_db = os.environ.get("POSTGRES_DB") or "career_copilot"
    processed = 0
    updated = 0
    ai_skill_extraction_available = True

    with connect(dbname=target_db) as conn:
        init_schema(conn)
        with conn.cursor() as cur:
            cur.execute(LOAD_PROFILES_SQL)
            rows = cur.fetchall()

        for user_id, skill_tags in rows:
            data, filename = get_resume_file_by_user_id(conn, int(user_id))
            if data is None:
                continue

            resume_text = strip_nul(extract_resume_text(data, filename))
            ai_extracted_skills: list[str] = []
            if resume_text and ai_skill_extraction_available:
                try:
                    ai_extracted_skills = extract_ai_resume_skill_tags(resume_text)
                except RuntimeError as e:
                    print(f"AI resume skill extraction disabled: {e}")
                    ai_skill_extraction_available = False
                except Exception as e:
                    print(f"AI resume skill extraction failed for user_id={user_id}: {e}")

            replace_user_skills(
                conn,
                user_id=int(user_id),
                skill_tags=str(skill_tags or ""),
                ai_extracted_skills=ai_extracted_skills,
            )
            processed += 1
            updated += int(bool(ai_extracted_skills))
            if processed % 25 == 0:
                conn.commit()
                print(
                    f"[user-skills-backfill] processed {processed} user(s), "
                    f"{updated} with AI-extracted skills"
                )
        conn.commit()

    print(
        f"User skills backfill: {processed} user(s) processed, "
        f"{updated} with AI-extracted skills (db={target_db})"
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
