from __future__ import annotations

import psycopg


def get_or_create_user(
    conn: psycopg.Connection,
    *,
    external_provider: str,
    external_subject: str,
    email: str | None,
) -> int:
    """
    Map an external identity (provider+subject) to an internal integer user id.
    Creates the user row on first login.
    """
    if not external_provider or not external_subject:
        raise ValueError("external_provider and external_subject are required")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM users
            WHERE external_provider = %s AND external_subject = %s
            """,
            (external_provider, external_subject),
        )
        row = cur.fetchone()
        if row:
            return int(row[0])

        cur.execute(
            """
            INSERT INTO users (external_provider, external_subject, email)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (external_provider, external_subject, email or None),
        )
        row2 = cur.fetchone()
        assert row2 is not None
        return int(row2[0])
