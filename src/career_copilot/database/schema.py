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

        # User-added jobs (separate from ingested jobs table)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_jobs (
                id BIGSERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NULL,
                company TEXT NULL,
                location TEXT NULL,
                salary_min INTEGER NULL,
                salary_max INTEGER NULL,
                description TEXT NULL,
                skills TEXT[] NULL,
                url TEXT NULL,
                raw JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS user_jobs_user_id_idx ON user_jobs (user_id)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS user_jobs_created_at_idx ON user_jobs (created_at DESC)"
        )

        # Applications: resume improvement or interview preparation for a job (ingested or user-added)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id BIGSERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                job_id BIGINT NOT NULL,
                job_source TEXT NOT NULL CHECK (job_source IN ('ingested', 'user')),
                stage TEXT NOT NULL CHECK (stage IN ('resume_improvement', 'interview_preparation')),
                status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'done')),
                history JSONB NOT NULL DEFAULT '[]'::jsonb,
                application_memory JSONB NOT NULL DEFAULT '{}'::jsonb,
                last_resume_text TEXT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (user_id, job_id, job_source, stage)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS applications_user_id_idx ON applications (user_id)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS applications_created_at_idx ON applications (created_at DESC)"
        )
        # Migrations for applications (safe if columns already exist)
        for sql in (
            "ALTER TABLE applications ADD COLUMN history JSONB NOT NULL DEFAULT '[]'::jsonb",
            "ALTER TABLE applications ADD COLUMN application_memory JSONB NOT NULL DEFAULT '{}'::jsonb",
            "ALTER TABLE applications ADD COLUMN last_resume_text TEXT",
        ):
            try:
                cur.execute(sql)
                conn.commit()
            except psycopg.ProgrammingError as e:
                conn.rollback()
                if e.sqlstate != "42701":  # duplicate_column
                    raise

        # Ensure demo user exists
        cur.execute(
            "INSERT INTO users (email) VALUES (%s) ON CONFLICT (email) DO NOTHING",
            ("demo@example.com",),
        )
    conn.commit()
