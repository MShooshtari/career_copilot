-- Chunk-level embeddings for market analysis RAG and other description-grounded features.
-- Applied automatically via init_schema on app startup; use this file for manual psql runs.

CREATE TABLE IF NOT EXISTS job_description_chunks (
  id BIGSERIAL PRIMARY KEY,
  job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  embedding vector(1536) NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (job_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS job_description_chunks_job_id_idx
  ON job_description_chunks (job_id);

CREATE INDEX IF NOT EXISTS job_description_chunks_hnsw_idx
  ON job_description_chunks USING hnsw (embedding vector_cosine_ops);
