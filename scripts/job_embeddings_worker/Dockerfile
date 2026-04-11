# Embedding worker image: drains jobs_embedding_queue → upserts jobs_embeddings.
#
# Required env (or use POSTGRES_DSN): POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
# Azure Database: set POSTGRES_SSLMODE=require
# Embeddings: OPENAI_API_KEY
#
# Build from repository root:
#   docker build -f scripts/job_embeddings_worker/Dockerfile -t career-copilot-embedding-worker:local .
#
# Run once (process a batch and exit):
#   docker run --rm -e POSTGRES_HOST=... -e POSTGRES_PASSWORD=... -e OPENAI_API_KEY=... career-copilot-embedding-worker:local

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CAREER_COPILOT_ROOT=/app

COPY scripts/job_embeddings_worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY . .

CMD ["python", "scripts/job_embeddings_worker/run.py"]

