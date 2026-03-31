-- pgvector setup for RAG (also applied by career_copilot.database.schema.init_schema).
-- Requires PostgreSQL with the pgvector extension available (e.g. Azure PostgreSQL: allowlist "vector").

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS jobs_embeddings (
  job_id BIGINT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
  content TEXT,
  embedding vector(1536) NOT NULL
);

CREATE INDEX IF NOT EXISTS jobs_embeddings_job_id_idx
  ON jobs_embeddings (job_id);

CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw_idx
  ON jobs_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS user_embeddings (
  user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  content TEXT,
  embedding vector(1536) NOT NULL
);

CREATE INDEX IF NOT EXISTS user_embeddings_hnsw_idx
  ON user_embeddings USING hnsw (embedding vector_cosine_ops);
