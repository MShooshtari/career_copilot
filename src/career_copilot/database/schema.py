"""Database schema initialization and migrations."""
from __future__ import annotations

import psycopg


def init_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        # Basic users table (demo-only; assume user_id=1 is logged in)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE
            );
            """
        )
        # One profile per user (resume stored as file bytes + filename)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                skill_tags TEXT,
                years_experience INTEGER,
                current_location TEXT,
                preferred_roles TEXT,
                industries TEXT,
                work_mode TEXT,
                employment_type TEXT,
                preferred_locations TEXT,
                salary_min INTEGER,
                salary_max INTEGER,
                resume_file BYTEA,
                resume_filename TEXT
            );
            """
        )
        # Commit so CREATE TABLE is persisted; then run migrations in separate transactions
        conn.commit()
        for sql in (
            "ALTER TABLE profiles ADD COLUMN resume_file BYTEA",
            "ALTER TABLE profiles ADD COLUMN resume_filename TEXT",
            "ALTER TABLE profiles DROP COLUMN IF EXISTS resume_text",
        ):
            try:
                cur.execute(sql)
                conn.commit()
            except psycopg.ProgrammingError as e:
                conn.rollback()
                if e.sqlstate != "42701":  # duplicate_column
                    raise
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_skills (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                skill TEXT NOT NULL
            );
            """
        )
        # Embeddings live only in Chroma (user_profiles collection); no vector storage in Postgres
        cur.execute("DROP TABLE IF EXISTS user_embeddings")

        # Ensure demo user exists
        cur.execute("INSERT INTO users (email) VALUES (%s) ON CONFLICT (email) DO NOTHING", ("demo@example.com",))
    conn.commit()
