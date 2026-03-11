from __future__ import annotations

import sys
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Json

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SQL_DIR = ROOT / "sql"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from career_copilot.database.db import load_env, connect  # noqa: E402
from career_copilot.ingestion.arbeitnow_api import (  # noqa: E402
    fetch_arbeitnow_jobs,
    normalize_arbeitnow_job,
)
from career_copilot.ingestion.remoteok_api import (  # noqa: E402
    fetch_remoteok_jobs,
    normalize_remoteok_job,
)
from career_copilot.ingestion.remotive_api import (  # noqa: E402
    fetch_remotive_jobs,
    normalize_remotive_job,
)
from career_copilot.ingestion.adzuna_api import (  # noqa: E402
    fetch_adzuna_jobs,
    normalize_adzuna_job,
)
from career_copilot.ingestion.common import NormalizedJob  # noqa: E402

UPSERT_SQL = """
INSERT INTO jobs (
  source, source_id, title, company, location,
  salary_min, salary_max, description, skills,
  posted_at, url, raw, updated_at
)
VALUES (
  %(source)s, %(source_id)s, %(title)s, %(company)s, %(location)s,
  %(salary_min)s, %(salary_max)s, %(description)s, %(skills)s,
  %(posted_at)s, %(url)s, %(raw)s, now()
)
ON CONFLICT (source, source_id) WHERE source_id IS NOT NULL
DO UPDATE SET
  title = EXCLUDED.title,
  company = EXCLUDED.company,
  location = EXCLUDED.location,
  salary_min = EXCLUDED.salary_min,
  salary_max = EXCLUDED.salary_max,
  description = EXCLUDED.description,
  skills = EXCLUDED.skills,
  posted_at = EXCLUDED.posted_at,
  url = EXCLUDED.url,
  raw = EXCLUDED.raw,
  updated_at = now();
"""


def apply_schema(conn) -> None:
    schema_path = SQL_DIR / "001_create_jobs.sql"
    sql_text = schema_path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql_text)
    conn.commit()

def ensure_database(dbname: str) -> None:
    """
    Create the target database if missing (best-effort).
    """
    for maintenance_db in ("postgres", "template1"):
        try:
            with connect(dbname=maintenance_db) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
                    exists = cur.fetchone() is not None
                    if not exists:
                        cur.execute(
                            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname))
                        )
                conn.commit()
            return
        except psycopg.OperationalError:
            continue
        except Exception:
            return


def _fetch_all_sources() -> list[tuple[str, list[NormalizedJob]]]:
    """Fetch from each source and return (source_name, normalized_jobs)."""
    out: list[tuple[str, list[NormalizedJob]]] = []
    # RemoteOK
    try:
        raw = fetch_remoteok_jobs()
        out.append(("remoteok", [normalize_remoteok_job(j) for j in raw]))
    except Exception as e:
        print(f"RemoteOK fetch failed: {e}")
        out.append(("remoteok", []))
    # Remotive (no key)
    try:
        raw = fetch_remotive_jobs()
        out.append(("remotive", [normalize_remotive_job(j) for j in raw]))
    except Exception as e:
        print(f"Remotive fetch failed: {e}")
        out.append(("remotive", []))
    # Arbeitnow (no key)
    try:
        raw = fetch_arbeitnow_jobs()
        out.append(("arbeitnow", [normalize_arbeitnow_job(j) for j in raw]))
    except Exception as e:
        print(f"Arbeitnow fetch failed: {e}")
        out.append(("arbeitnow", []))
    # Adzuna (needs ADZUNA_APP_ID + ADZUNA_APP_KEY)
    try:
        raw = fetch_adzuna_jobs()
        out.append(("adzuna", [normalize_adzuna_job(j) for j in raw]))
    except Exception as e:
        print(f"Adzuna fetch failed: {e}")
        out.append(("adzuna", []))
    return out


def main() -> None:
    from datetime import datetime, timezone
    load_env()
    run_at = datetime.now(timezone.utc).isoformat()
    source_batches = _fetch_all_sources()
    normalized: list[NormalizedJob] = []
    for _name, jobs in source_batches:
        normalized.extend(jobs)

    ensure_database("career_copilot")
    try:
        conn = connect(dbname="career_copilot")
    except psycopg.OperationalError as e:
        msg = str(e)
        if 'database "career_copilot" does not exist' in msg:
            ensure_database("career_copilot")
            try:
                conn = connect(dbname="career_copilot")
            except psycopg.OperationalError:
                raise RuntimeError(
                    'Database "career_copilot" does not exist and could not be created automatically.\n'
                    'Create it manually, then re-run:\n\n'
                    "  psql -U postgres -d postgres -c \"CREATE DATABASE career_copilot;\""
                ) from e
        else:
            raise

    with conn:
        apply_schema(conn)
        with conn.cursor() as cur:
            inserted = 0
            for job in normalized:
                if job.source_id is None:
                    continue
                cur.execute(
                    UPSERT_SQL,
                    {
                        "source": job.source,
                        "source_id": job.source_id,
                        "title": job.title,
                        "company": job.company,
                        "location": job.location,
                        "salary_min": job.salary_min,
                        "salary_max": job.salary_max,
                        "description": job.description,
                        "skills": job.skills,
                        "posted_at": job.posted_at,
                        "url": job.url,
                        "raw": Json(job.raw),
                    },
                )
                inserted += 1

            cur.execute("SELECT source, count(*) FROM jobs GROUP BY source ORDER BY source")
            per_source = dict(cur.fetchall())
            cur.execute("SELECT count(*) FROM jobs")
            total = cur.fetchone()[0]

        conn.commit()

    parts = " | ".join(f"{name}: {len(jobs)} fetched" for name, jobs in source_batches)
    print(f"[{run_at}] {parts}")
    print(f"  Upserted {inserted} | Total in DB: {total} (by source: {per_source})")


if __name__ == "__main__":
    main()

