# RAG indexing image: Postgres jobs → pgvector embeddings (Azure Container Apps Jobs / cron).
#
# Required env (or use POSTGRES_DSN): POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
# Azure Database: set POSTGRES_SSLMODE=require and POSTGRES_SKIP_ENSURE_DATABASE=1
# Embeddings: OPENAI_API_KEY (text-embedding-3-large)
# Database must allow CREATE EXTENSION vector (or run sql/003_pgvector.sql once).
#
# Build from repository root (uses /.dockerignore; no --ignorefile needed):
#   docker build -f scripts/rag_index/Dockerfile -t career-copilot-rag-index:local .
# Run:
#   docker run --rm -e POSTGRES_HOST=... -e POSTGRES_PASSWORD=... -e OPENAI_API_KEY=... career-copilot-rag-index:local

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CAREER_COPILOT_ROOT=/app

COPY scripts/rag_index/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY . .

CMD ["python", "scripts/rag_index/run.py"]
