"""Application tracking: resume improvement or interview preparation for a job."""

from __future__ import annotations

import json
import psycopg


def list_applications(
    conn: psycopg.Connection,
    user_id: int,
    *,
    stage: str | None = None,
    status: str | None = None,
) -> list[tuple]:
    """
    List applications for a user, newest first.
    Returns list of rows: (id, user_id, job_id, job_source, stage, status, created_at, updated_at).
    Optionally filter by stage ('resume_improvement' | 'interview_preparation') and/or status ('active' | 'done').
    """
    with conn.cursor() as cur:
        sql = """
            SELECT id, user_id, job_id, job_source, stage, status, history, application_memory, last_resume_text, created_at, updated_at
            FROM applications
            WHERE user_id = %s
        """
        params: list = [user_id]
        if stage:
            sql += " AND stage = %s"
            params.append(stage)
        if status:
            sql += " AND status = %s"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        cur.execute(sql, tuple(params))
        return list(cur.fetchall())


def add_application(
    conn: psycopg.Connection,
    user_id: int,
    job_id: int,
    job_source: str,
    stage: str,
    *,
    status: str = "active",
) -> int | None:
    """
    Add an application (resume_improvement or interview_preparation for a job).
    job_source must be 'ingested' or 'user'.
    Returns application id if inserted, None if duplicate (unique on user_id, job_id, job_source, stage).
    """
    if job_source not in ("ingested", "user") or stage not in (
        "resume_improvement",
        "interview_preparation",
    ):
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO applications (user_id, job_id, job_source, stage, status)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, job_id, job_source, stage) DO UPDATE SET
                updated_at = now(),
                status = EXCLUDED.status
            RETURNING id
            """,
            (user_id, job_id, job_source, stage, status),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None


def get_application(
    conn: psycopg.Connection,
    user_id: int,
    application_id: int,
) -> tuple | None:
    """Fetch one application by id and user_id. Returns row or None."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, job_id, job_source, stage, status, history, application_memory, last_resume_text, created_at, updated_at
            FROM applications
            WHERE id = %s AND user_id = %s
            """,
            (application_id, user_id),
        )
        return cur.fetchone()


def get_application_by_key(
    conn: psycopg.Connection,
    user_id: int,
    job_id: int,
    job_source: str,
    stage: str,
) -> tuple | None:
    """Fetch one application by (user_id, job_id, job_source, stage). Returns row or None."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, job_id, job_source, stage, status, history, application_memory, last_resume_text, created_at, updated_at
            FROM applications
            WHERE user_id = %s AND job_id = %s AND job_source = %s AND stage = %s
            """,
            (user_id, job_id, job_source, stage),
        )
        return cur.fetchone()


def set_application_history(
    conn: psycopg.Connection,
    user_id: int,
    application_id: int,
    history: list[dict],
) -> bool:
    """Persist chat history JSON for an application. Returns True if updated."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE applications
            SET history = %s::jsonb, updated_at = now()
            WHERE id = %s AND user_id = %s
            """,
            (json.dumps(history or []), application_id, user_id),
        )
        return cur.rowcount > 0


def set_application_memory(
    conn: psycopg.Connection,
    user_id: int,
    application_id: int,
    memory: dict,
) -> bool:
    """Persist compact memory JSON for an application. Returns True if updated."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE applications
            SET application_memory = %s::jsonb, updated_at = now()
            WHERE id = %s AND user_id = %s
            """,
            (json.dumps(memory or {}), application_id, user_id),
        )
        return cur.rowcount > 0


def set_application_last_resume_text(
    conn: psycopg.Connection,
    user_id: int,
    application_id: int,
    last_resume_text: str | None,
) -> bool:
    """Persist last generated resume text for an application. Returns True if updated."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE applications
            SET last_resume_text = %s, updated_at = now()
            WHERE id = %s AND user_id = %s
            """,
            (last_resume_text, application_id, user_id),
        )
        return cur.rowcount > 0


def remove_application(conn: psycopg.Connection, user_id: int, application_id: int) -> bool:
    """Delete an application. Returns True if a row was deleted."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM applications WHERE id = %s AND user_id = %s",
            (application_id, user_id),
        )
        return cur.rowcount > 0


def application_row_to_dict(row: tuple) -> dict:
    """Convert an applications row to a dict for API/templates."""
    (
        id_,
        user_id,
        job_id,
        job_source,
        stage,
        status,
        history,
        application_memory,
        last_resume_text,
        created_at,
        updated_at,
    ) = row

    def _dt_to_iso(v):
        try:
            return v.isoformat() if v is not None else None
        except Exception:
            return None

    return {
        "id": id_,
        "user_id": user_id,
        "job_id": job_id,
        "job_source": job_source,
        "stage": stage,
        "status": status,
        "history": history or [],
        "application_memory": application_memory or {},
        "last_resume_text": last_resume_text,
        # JSONResponse can't serialize datetime objects; store as ISO strings.
        "created_at": _dt_to_iso(created_at),
        "updated_at": _dt_to_iso(updated_at),
    }


def enrich_applications_with_job_info(
    conn: psycopg.Connection,
    rows: list[tuple],
) -> list[dict]:
    """
    Take list of application rows and return list of dicts with application fields
    plus job title, company, and action_url (link to improve-resume or prepare-interview).
    """
    from career_copilot.database.jobs import get_job_by_id, get_user_job_by_id, row_to_job_dict, user_job_row_to_dict

    result: list[dict] = []
    for row in rows:
        app = application_row_to_dict(row)
        job_id = app["job_id"]
        job_source = app["job_source"]
        stage = app["stage"]

        job_title = "Job"
        job_company = ""
        if job_source == "user":
            uj = get_user_job_by_id(conn, app["user_id"], job_id)
            if uj:
                job_d = user_job_row_to_dict(uj)
                job_title = job_d.get("title") or job_title
                job_company = job_d.get("company") or job_company
            base = "/my-jobs"
        else:
            j = get_job_by_id(conn, job_id)
            if j:
                job_d = row_to_job_dict(j)
                job_title = job_d.get("title") or job_title
                job_company = job_d.get("company") or job_company
            base = "/jobs"

        if stage == "resume_improvement":
            action_path = f"{base}/{job_id}/improve-resume"
        else:
            action_path = f"{base}/{job_id}/prepare-interview"

        app["job_title"] = job_title
        app["job_company"] = job_company
        app["action_url"] = action_path
        result.append(app)
    return result
