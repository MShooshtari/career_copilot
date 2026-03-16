"""Job-related database operations and Chroma result resolution."""

from __future__ import annotations

import psycopg


def _norm_sid(sid: str | int | float | None) -> str | None:
    """Normalize source_id to str for consistent map keys."""
    if sid is None:
        return None
    if isinstance(sid, str):
        return sid or None
    return str(int(sid))


def _chroma_id_to_source_source_id(chroma_id: str) -> tuple[str | None, str | None]:
    """Parse Chroma doc id 'source:source_id' into (source, source_id)."""
    if ":" in chroma_id:
        a, b = chroma_id.split(":", 1)
        return (a or None, b or None)
    return (chroma_id or None, None)


def resolve_job_ids(
    conn: psycopg.Connection,
    results: list[dict],
) -> dict[tuple[str | None, str | None], int]:
    """Map (source, source_id) to Postgres job id for each result. Returns dict keyed by (source, source_id) -> id."""
    if not results:
        return {}
    pairs: list[tuple[str | None, str | None]] = []
    for r in results:
        meta = r.get("metadata") or {}
        src = meta.get("source")
        sid = _norm_sid(meta.get("source_id"))
        if src is None and sid is None:
            chroma_id = r.get("id") or ""
            src, sid = _chroma_id_to_source_source_id(chroma_id)
        pairs.append((src, sid))
    pairs = list(dict.fromkeys(pairs))  # unique
    out: dict[tuple[str | None, str | None], int] = {}
    with conn.cursor() as cur:
        for src, sid in pairs:
            if src is None:
                continue
            cur.execute(
                "SELECT id FROM jobs WHERE source = %s AND source_id = %s",
                (src, sid),
            )
            row = cur.fetchone()
            if row:
                out[(src, sid)] = int(row[0])
    return out


def get_job_by_id(conn: psycopg.Connection, job_id: int) -> tuple | None:
    """Fetch full job row by Postgres id. Returns DB row or None."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source, source_id, title, company, location,
                   salary_min, salary_max, description, skills,
                   posted_at, url
            FROM jobs
            WHERE id = %s
            """,
            (job_id,),
        )
        return cur.fetchone()


def row_to_job_dict(row: tuple) -> dict:
    """Convert a jobs table row to a dict for templates (full job detail)."""
    (
        id_,
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
    ) = row
    return {
        "id": id_,
        "source": source,
        "source_id": source_id,
        "title": title or "Job",
        "company": company or "",
        "location": location or "",
        "salary_min": salary_min,
        "salary_max": salary_max,
        "description": description or "",
        "skills": list(skills) if skills else [],
        "posted_at": posted_at,
        "url": url or "",
    }


def row_to_job_dict_snippet(row: tuple, description_max_chars: int = 500) -> dict:
    """Convert a jobs table row to a dict with truncated description (e.g. for improve-resume page)."""
    d = row_to_job_dict(row)
    desc = d.get("description") or ""
    if len(desc) > description_max_chars:
        d["description"] = desc[:description_max_chars].rstrip() + "…"
    return d


def get_user_job_by_id(
    conn: psycopg.Connection, user_id: int, job_id: int
) -> tuple | None:
    """Fetch a user-added job by id and user_id. Returns DB row or None."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, title, company, location,
                   salary_min, salary_max, description, skills, url
            FROM user_jobs
            WHERE id = %s AND user_id = %s
            """,
            (job_id, user_id),
        )
        return cur.fetchone()


def list_user_jobs(
    conn: psycopg.Connection, user_id: int
) -> list[tuple]:
    """List all user-added jobs for a user, newest first. Returns list of rows."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, title, company, location,
                   salary_min, salary_max, description, skills, url
            FROM user_jobs
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        return list(cur.fetchall())


def insert_user_job(
    conn: psycopg.Connection,
    user_id: int,
    *,
    title: str | None = None,
    company: str | None = None,
    location: str | None = None,
    salary_min: int | None = None,
    salary_max: int | None = None,
    description: str | None = None,
    skills: list[str] | None = None,
    url: str | None = None,
    raw: dict | None = None,
) -> int:
    """Insert a user job and return its id."""
    from psycopg.types.json import Json

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_jobs (
                user_id, title, company, location,
                salary_min, salary_max, description, skills, url, raw
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                title or None,
                company or None,
                location or None,
                salary_min,
                salary_max,
                description or None,
                skills if skills else None,
                url or None,
                Json(raw or {}),
            ),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def user_job_row_to_dict(row: tuple) -> dict:
    """Convert a user_jobs row to the same dict shape as row_to_job_dict (for templates/agents)."""
    (
        id_,
        user_id,
        title,
        company,
        location,
        salary_min,
        salary_max,
        description,
        skills,
        url,
    ) = row
    return {
        "id": id_,
        "source": "user",
        "source_id": str(id_),
        "title": title or "Job",
        "company": company or "",
        "location": location or "",
        "salary_min": salary_min,
        "salary_max": salary_max,
        "description": description or "",
        "skills": list(skills) if skills else [],
        "posted_at": None,
        "url": url or "",
    }


def format_user_jobs_for_recommendations(
    rows: list[tuple], snippet_max_chars: int = 400
) -> list[dict]:
    """Build list of job dicts for recommendations template from user_jobs rows."""
    out: list[dict] = []
    for row in rows:
        d = user_job_row_to_dict(row)
        desc = (d.get("description") or "")[:snippet_max_chars]
        if len(d.get("description") or "") > snippet_max_chars:
            desc = desc.rstrip() + "…"
        out.append(
            {
                "job_id": d["id"],
                "is_user_job": True,
                "title": d["title"],
                "company": d["company"],
                "location": d["location"],
                "url": d["url"],
                "snippet": desc,
                "salary_min": d["salary_min"],
                "salary_max": d["salary_max"],
                "skills": d["skills"],
            }
        )
    return out


def format_recommendation_jobs(
    raw: list[dict],
    id_map: dict[tuple[str | None, str | None], int],
    snippet_max_chars: int = 400,
) -> list[dict]:
    """Build list of job dicts for the recommendations template from Chroma results and Postgres id map."""
    jobs_for_template: list[dict] = []
    for r in raw:
        meta = r.get("metadata") or {}
        src = meta.get("source")
        sid = _norm_sid(meta.get("source_id"))
        postgres_id = id_map.get((src, sid)) if src is not None else None
        doc = (r.get("document") or "")[:snippet_max_chars]
        if len(r.get("document") or "") > snippet_max_chars:
            doc = doc.rstrip() + "…"
        jobs_for_template.append(
            {
                "job_id": postgres_id,
                "is_user_job": False,
                "title": meta.get("title") or "Job",
                "company": meta.get("company") or "",
                "location": meta.get("location") or "",
                "url": meta.get("url") or "",
                "snippet": doc,
                "distance": r.get("distance"),
                "salary_min": meta.get("salary_min"),
                "salary_max": meta.get("salary_max"),
                "skills": (meta.get("skills") or "").split(",") if meta.get("skills") else [],
            }
        )
    return jobs_for_template
