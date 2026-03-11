"""
One-off script: repair job descriptions in the database in-place.

Applies the same html_to_plain_text logic (strip HTML, fix mojibake, normalize
whitespace) to every existing row with a description, then UPDATEs the row
if the repaired text differs. Safe to run multiple times.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from career_copilot.database.db import load_env, connect  # noqa: E402
from career_copilot.ingestion.common import html_to_plain_text  # noqa: E402

LOAD_SQL = """
SELECT id, description
FROM jobs
WHERE description IS NOT NULL AND description != '';
"""

UPDATE_SQL = """
UPDATE jobs
SET description = %s, updated_at = now()
WHERE id = %s;
"""


def main() -> None:
    load_env()
    updated = 0
    skipped = 0
    errors = 0

    with connect(dbname="career_copilot") as conn:
        with conn.cursor() as cur:
            cur.execute(LOAD_SQL)
            rows = cur.fetchall()

        for job_id, description in rows:
            if description is None:
                skipped += 1
                continue
            repaired = html_to_plain_text(description)
            if repaired is None:
                repaired = ""
            if repaired == description:
                skipped += 1
                continue
            try:
                with conn.cursor() as cur:
                    cur.execute(UPDATE_SQL, (repaired, job_id))
                updated += 1
            except Exception as e:
                errors += 1
                print(f"  Error updating id={job_id}: {e}", file=sys.stderr)

        conn.commit()

    print(f"Repaired {updated} row(s), unchanged {skipped}, errors {errors}")


if __name__ == "__main__":
    main()
    sys.exit(0)
