"""Profile and user skills database operations."""
from __future__ import annotations

import psycopg


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
) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM user_skills WHERE user_id = %s", (user_id,))
        for raw in skill_tags.split(","):
            skill = raw.strip()
            if not skill:
                continue
            cur.execute(
                "INSERT INTO user_skills (user_id, skill) VALUES (%s, %s)",
                (user_id, skill),
            )

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
                resume_filename
            )
            VALUES (%(user_id)s, %(skill_tags)s, %(years_experience)s, %(current_location)s,
                    %(preferred_roles)s, %(industries)s, %(work_mode)s, %(employment_type)s,
                    %(preferred_locations)s, %(salary_min)s, %(salary_max)s, %(resume_file)s, %(resume_filename)s)
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
                resume_filename = EXCLUDED.resume_filename;
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


def get_resume_file_by_user_id(
    conn: psycopg.Connection, user_id: int
) -> tuple[bytes | None, str | None]:
    """Return (resume_file_bytes, resume_filename) for the user. (None, None) if not found."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT resume_file, resume_filename FROM profiles WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    if not row or row[0] is None:
        return (None, None)
    return (bytes(row[0]), (row[1] or "resume").replace('"', ""))
