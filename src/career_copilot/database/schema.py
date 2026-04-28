"""Database schema initialization and migrations."""

from __future__ import annotations

import psycopg

from career_copilot.rag.embedding import EMBEDDING_VECTOR_DIMENSIONS


def init_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        # Users table: maps external identities (Entra External ID, etc.) to an internal user id.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT,
                external_provider TEXT,
                external_subject TEXT
            );
            """
        )
        # Migrate older schemas (safe if constraints/columns already match).
        for sql in (
            "ALTER TABLE users ADD COLUMN email TEXT",
            "ALTER TABLE users ADD COLUMN external_provider TEXT",
            "ALTER TABLE users ADD COLUMN external_subject TEXT",
            "ALTER TABLE users DROP CONSTRAINT IF EXISTS users_email_key",
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
            CREATE UNIQUE INDEX IF NOT EXISTS users_external_identity_uniq
              ON users (external_provider, external_subject)
              WHERE external_provider IS NOT NULL AND external_subject IS NOT NULL;
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
                resume_filename TEXT,
                resume_blob_container TEXT,
                resume_blob_name TEXT
            );
            """
        )
        # Commit so CREATE TABLE is persisted; then run migrations in separate transactions
        conn.commit()
        for sql in (
            "ALTER TABLE profiles ADD COLUMN resume_file BYTEA",
            "ALTER TABLE profiles ADD COLUMN resume_filename TEXT",
            "ALTER TABLE profiles ADD COLUMN resume_blob_container TEXT",
            "ALTER TABLE profiles ADD COLUMN resume_blob_name TEXT",
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
        cur.execute(
            "CREATE INDEX IF NOT EXISTS jobs_embeddings_job_id_idx ON jobs_embeddings (job_id)"
        )
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

        # Chunk-level embeddings for RAG (job descriptions split for retrieval).
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS job_description_chunks (
                id BIGSERIAL PRIMARY KEY,
                job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector({EMBEDDING_VECTOR_DIMENSIONS}) NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (job_id, chunk_index)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS job_description_chunks_job_id_idx "
            "ON job_description_chunks (job_id)"
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS job_description_chunks_hnsw_idx
            ON job_description_chunks USING hnsw (embedding vector_cosine_ops)
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

        # User interactions with shown jobs. `job_id` points at jobs.id for ingested jobs and
        # user_jobs.id for user-added jobs, distinguished by job_source.
        cur.execute(
            """
            DO $$
            BEGIN
                IF to_regclass('user_job_interaction') IS NULL
                   AND to_regclass('job_feedback') IS NOT NULL THEN
                    ALTER TABLE job_feedback RENAME TO user_job_interaction;
                END IF;
            END $$;
            """
        )
        cur.execute(
            """
            DO $$
            BEGIN
                IF to_regclass('job_feedback_user_source_feedback_idx') IS NOT NULL
                   AND to_regclass('user_job_interaction_user_source_feedback_idx') IS NULL THEN
                    ALTER INDEX job_feedback_user_source_feedback_idx
                    RENAME TO user_job_interaction_user_source_feedback_idx;
                END IF;

                IF to_regclass('job_feedback_user_job_idx') IS NOT NULL
                   AND to_regclass('user_job_interaction_user_job_idx') IS NULL THEN
                    ALTER INDEX job_feedback_user_job_idx
                    RENAME TO user_job_interaction_user_job_idx;
                END IF;
            END $$;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_job_interaction (
                id BIGSERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                job_id BIGINT NOT NULL,
                job_source TEXT NOT NULL CHECK (job_source IN ('ingested', 'user')),
                feedback TEXT NOT NULL CHECK (feedback IN ('liked', 'disliked', 'applied', 'deleted')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS user_job_interaction_user_source_feedback_idx "
            "ON user_job_interaction (user_id, job_source, feedback)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS user_job_interaction_user_job_idx "
            "ON user_job_interaction (user_id, job_source, job_id)"
        )
        cur.execute(
            """
            ALTER TABLE user_job_interaction
            DROP CONSTRAINT IF EXISTS job_feedback_user_id_job_id_job_source_key
            """
        )
        cur.execute(
            """
            ALTER TABLE user_job_interaction
            DROP CONSTRAINT IF EXISTS user_job_interaction_user_id_job_id_job_source_key
            """
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS user_job_interaction_unique_feedback_idx "
            "ON user_job_interaction (user_id, job_id, job_source, feedback)"
        )
        cur.execute(
            """
            DO $$
            BEGIN
                ALTER TABLE user_job_interaction
                DROP CONSTRAINT IF EXISTS job_feedback_feedback_check;
                ALTER TABLE user_job_interaction
                DROP CONSTRAINT IF EXISTS user_job_interaction_feedback_check;

                UPDATE user_job_interaction
                SET feedback = CASE feedback
                    WHEN 'like' THEN 'liked'
                    WHEN 'dislike' THEN 'disliked'
                    ELSE feedback
                END
                WHERE feedback IN ('like', 'dislike');

                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'user_job_interaction_feedback_check'
                      AND conrelid = 'user_job_interaction'::regclass
                ) THEN
                    ALTER TABLE user_job_interaction
                    ADD CONSTRAINT user_job_interaction_feedback_check
                    CHECK (feedback IN ('liked', 'disliked', 'applied', 'deleted'));
                END IF;
            END $$;
            """
        )
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

    conn.commit()
