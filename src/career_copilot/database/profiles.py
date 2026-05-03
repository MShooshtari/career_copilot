"""Profile and user skills database operations."""

from __future__ import annotations

from collections.abc import Iterable

import psycopg


def replace_user_skills(
    conn: psycopg.Connection,
    *,
    user_id: int,
    skill_tags: str,
    ai_extracted_skills: list[str] | None = None,
) -> None:
    """Replace manual skill rows and store the resume-derived AI skill snapshot."""
    manual_skills = _split_skill_tags(skill_tags)
    ai_skills = _dedupe_skills(ai_extracted_skills or [])

    with conn.cursor() as cur:
        cur.execute("DELETE FROM user_skills WHERE user_id = %s", (user_id,))
        if manual_skills:
            for index, skill in enumerate(manual_skills):
                cur.execute(
                    """
                    INSERT INTO user_skills (user_id, skill, ai_extracted_skills)
                    VALUES (%s, %s, %s)
                    """,
                    (user_id, skill, ai_skills if index == 0 and ai_skills else None),
                )
        elif ai_skills:
            cur.execute(
                """
                INSERT INTO user_skills (user_id, skill, ai_extracted_skills)
                VALUES (%s, %s, %s)
                """,
                (user_id, "", ai_skills),
            )


def upsert_user_profile(
    conn: psycopg.Connection,
    *,
    user_id: int,
    skill_tags: str,
    years_experience: int | None,
    current_location: str,
    preferred_roles: str,
    industries: str,
    work_mode: str,
    employment_type: str,
    preferred_locations: str,
    salary_min: int | None,
    salary_max: int | None,
    resume_file: bytes | None,
    resume_filename: str | None,
    resume_blob_container: str | None = None,
    resume_blob_name: str | None = None,
    ai_extracted_skills: list[str] | None = None,
) -> None:
    replace_user_skills(
        conn,
        user_id=user_id,
        skill_tags=skill_tags,
        ai_extracted_skills=ai_extracted_skills,
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO profiles (
                user_id,
                skill_tags,
                years_experience,
                current_location,
                preferred_roles,
                industries,
                work_mode,
                employment_type,
                preferred_locations,
                salary_min,
                salary_max,
                resume_file,
                resume_filename,
                resume_blob_container,
                resume_blob_name
            )
            VALUES (%(user_id)s, %(skill_tags)s, %(years_experience)s, %(current_location)s,
                    %(preferred_roles)s, %(industries)s, %(work_mode)s, %(employment_type)s,
                    %(preferred_locations)s, %(salary_min)s, %(salary_max)s, %(resume_file)s, %(resume_filename)s,
                    %(resume_blob_container)s, %(resume_blob_name)s)
            ON CONFLICT (user_id) DO UPDATE
            SET
                skill_tags = EXCLUDED.skill_tags,
                years_experience = EXCLUDED.years_experience,
                current_location = EXCLUDED.current_location,
                preferred_roles = EXCLUDED.preferred_roles,
                industries = EXCLUDED.industries,
                work_mode = EXCLUDED.work_mode,
                employment_type = EXCLUDED.employment_type,
                preferred_locations = EXCLUDED.preferred_locations,
                salary_min = EXCLUDED.salary_min,
                salary_max = EXCLUDED.salary_max,
                resume_file = EXCLUDED.resume_file,
                resume_filename = EXCLUDED.resume_filename,
                resume_blob_container = EXCLUDED.resume_blob_container,
                resume_blob_name = EXCLUDED.resume_blob_name;
            """,
            {
                "user_id": user_id,
                "skill_tags": skill_tags,
                "years_experience": years_experience,
                "current_location": current_location,
                "preferred_roles": preferred_roles,
                "industries": industries,
                "work_mode": work_mode,
                "employment_type": employment_type,
                "preferred_locations": preferred_locations,
                "salary_min": salary_min,
                "salary_max": salary_max,
                "resume_file": resume_file,
                "resume_filename": resume_filename or None,
                "resume_blob_container": resume_blob_container or None,
                "resume_blob_name": resume_blob_name or None,
            },
        )
    conn.commit()


def get_profile_by_user_id(conn: psycopg.Connection, user_id: int) -> tuple | None:
    """Fetch profile row for a user. Returns DB row or None."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                p.skill_tags,
                p.years_experience,
                p.current_location,
                p.preferred_roles,
                p.industries,
                p.work_mode,
                p.employment_type,
                p.preferred_locations,
                p.salary_min,
                p.salary_max,
                p.resume_filename
            FROM profiles p
            WHERE p.user_id = %s
            """,
            (user_id,),
        )
        return cur.fetchone()


def list_user_skills_lower(conn: psycopg.Connection, user_id: int) -> list[str]:
    """Return normalized manual and AI-extracted skills for the user."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT lower(trim(value)) AS skill
            FROM (
                SELECT skill AS value
                FROM user_skills
                WHERE user_id = %s

                UNION ALL

                SELECT unnest(ai_extracted_skills) AS value
                FROM user_skills
                WHERE user_id = %s AND ai_extracted_skills IS NOT NULL
            ) AS skills
            WHERE value IS NOT NULL AND trim(value) <> ''
            ORDER BY skill
            """,
            (user_id, user_id),
        )
        rows = cur.fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def get_resume_file_by_user_id(
    conn: psycopg.Connection, user_id: int
) -> tuple[bytes | None, str | None]:
    """Return (resume_file_bytes, resume_filename) for the user. (None, None) if not found."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT resume_file, resume_filename, resume_blob_container, resume_blob_name FROM profiles WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    if not row:
        return (None, None)
    resume_file, resume_filename, blob_container, blob_name = row
    if blob_container and blob_name:
        try:
            from career_copilot.storage.resumes import get_resume, resume_storage_mode

            if resume_storage_mode() == "blob":
                data = get_resume(container=str(blob_container), blob_name=str(blob_name))
                return (data, (resume_filename or "resume").replace('"', ""))
        except Exception:
            # Fall back to DB bytes if present; otherwise treat as missing.
            pass
    if resume_file is None:
        return (None, None)
    return (bytes(resume_file), (resume_filename or "resume").replace('"', ""))


def _split_skill_tags(skill_tags: str) -> list[str]:
    return _dedupe_skills(raw.strip() for raw in skill_tags.split(","))


def _dedupe_skills(skills: Iterable[object]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for raw in skills:
        skill = str(raw).strip()
        if not skill:
            continue
        key = skill.casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(skill)
    return found
