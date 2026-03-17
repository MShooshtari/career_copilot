-- User-added jobs (manual, file, or URL). Separate from ingested jobs.
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

CREATE INDEX IF NOT EXISTS user_jobs_user_id_idx ON user_jobs (user_id);
CREATE INDEX IF NOT EXISTS user_jobs_created_at_idx ON user_jobs (created_at DESC);
