"""Database schema initialization and migrations."""

from __future__ import annotations

import psycopg

from career_copilot.rag.embedding import EMBEDDING_VECTOR_DIMENSIONS


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
        # Ingested jobs table (used by RAG indexer). Kept in sync with sql/001_create_jobs.sql.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id BIGSERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                source_id TEXT NULL,
                title TEXT NULL,
                company TEXT NULL,
                location TEXT NULL,
                salary_min INTEGER NULL,
                salary_max INTEGER NULL,
                description TEXT NULL,
                skills TEXT[] NULL,
                posted_at TIMESTAMPTZ NULL,
                url TEXT NULL,
                raw JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        # Idempotency + common query indexes (same intent as sql/001_create_jobs.sql).
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS jobs_source_source_id_uniq
              ON jobs (source, source_id)
              WHERE source_id IS NOT NULL;
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS jobs_source_url_uniq
              ON jobs (source, url)
              WHERE url IS NOT NULL;
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS jobs_posted_at_idx ON jobs (posted_at DESC);")
        conn.commit()

        # pgvector: job embeddings in jobs_embeddings; user profiles in user_embeddings
        # NOTE: HNSW has a dimensions limit on common pgvector builds; keep embedding dims <= 2000.
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        except psycopg.Error:
            conn.rollback()
            raise
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS jobs_embeddings (
                job_id BIGINT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                content TEXT,
                embedding vector({EMBEDDING_VECTOR_DIMENSIONS}) NOT NULL
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS user_embeddings (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                content TEXT,
                embedding vector({EMBEDDING_VECTOR_DIMENSIONS}) NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS jobs_embeddings_job_id_idx ON jobs_embeddings (job_id)")
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw_idx
            ON jobs_embeddings USING hnsw (embedding vector_cosine_ops)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS user_embeddings_hnsw_idx
            ON user_embeddings USING hnsw (embedding vector_cosine_ops)
            """
        )
        conn.commit()

        # Embedding queue: enqueue job ids when description changes, then process out-of-band
        # (e.g. Azure Container Apps Job / background worker).
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs_embedding_queue (
                job_id BIGINT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                description_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing')),
                requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                locked_at TIMESTAMPTZ NULL,
                locked_by TEXT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS jobs_embedding_queue_status_requested_idx
              ON jobs_embedding_queue (status, requested_at);
            """
        )
        conn.commit()

        # Trigger: enqueue when a job is inserted or when its description changes.
        # Uses an UPSERT so repeated updates collapse to a single pending row.
        cur.execute(
            """
            CREATE OR REPLACE FUNCTION enqueue_job_embedding() RETURNS trigger AS $$
            DECLARE
              h TEXT;
            BEGIN
              IF TG_OP = 'INSERT' THEN
                IF NEW.description IS NULL OR btrim(NEW.description) = '' THEN
                  RETURN NEW;
                END IF;
              ELSIF TG_OP = 'UPDATE' THEN
                IF NEW.description IS NOT DISTINCT FROM OLD.description THEN
                  RETURN NEW;
                END IF;
                IF NEW.description IS NULL OR btrim(NEW.description) = '' THEN
                  RETURN NEW;
                END IF;
              END IF;

              h := md5(coalesce(NEW.description, ''));
              INSERT INTO jobs_embedding_queue (job_id, description_hash, status, requested_at)
              VALUES (NEW.id, h, 'pending', now())
              ON CONFLICT (job_id) DO UPDATE SET
                description_hash = EXCLUDED.description_hash,
                status = 'pending',
                requested_at = now(),
                locked_at = NULL,
                locked_by = NULL;
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        cur.execute("DROP TRIGGER IF EXISTS jobs_embedding_enqueue_trg ON jobs;")
        cur.execute(
            """
            CREATE TRIGGER jobs_embedding_enqueue_trg
            AFTER INSERT OR UPDATE OF description ON jobs
            FOR EACH ROW
            EXECUTE FUNCTION enqueue_job_embedding();
            """
        )
        conn.commit()

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
        # Commit so CREATE TABLE is persisted; then run migrations in separate transactions
        conn.commit()
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

        # Commit so CREATE TABLE is persisted; then run migrations in separate transactions
        conn.commit()
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
