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

-- Keep the table idempotent across re-ingests.
CREATE UNIQUE INDEX IF NOT EXISTS jobs_source_source_id_uniq
  ON jobs (source, source_id)
  WHERE source_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS jobs_source_url_uniq
  ON jobs (source, url)
  WHERE url IS NOT NULL;

CREATE INDEX IF NOT EXISTS jobs_posted_at_idx ON jobs (posted_at DESC);
