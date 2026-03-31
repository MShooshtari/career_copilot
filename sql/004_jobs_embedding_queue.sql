-- Queue for incremental job-embedding updates.
-- Enqueued by a DB trigger on jobs insert/update(description); drained by a worker job.

CREATE TABLE IF NOT EXISTS jobs_embedding_queue (
  job_id BIGINT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
  description_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing')),
  requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  locked_at TIMESTAMPTZ NULL,
  locked_by TEXT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT NULL
);

CREATE INDEX IF NOT EXISTS jobs_embedding_queue_status_requested_idx
  ON jobs_embedding_queue (status, requested_at);

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

DROP TRIGGER IF EXISTS jobs_embedding_enqueue_trg ON jobs;
CREATE TRIGGER jobs_embedding_enqueue_trg
AFTER INSERT OR UPDATE OF description ON jobs
FOR EACH ROW
EXECUTE FUNCTION enqueue_job_embedding();

