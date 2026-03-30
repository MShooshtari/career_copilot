# Ingestion-only image for scheduled runs (Azure Container Apps Jobs / cron).
#
# Required env (or use POSTGRES_DSN): POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
# Azure Database: set POSTGRES_SSLMODE=require and POSTGRES_SKIP_ENSURE_DATABASE=1
# Optional: ADZUNA_APP_ID, ADZUNA_APP_KEY
#
# Build from repository root (context must include src/, sql/, and this folder):
#   docker buildx build -f scripts/ingestion/Dockerfile --ignorefile scripts/ingestion/.dockerignore -t career-copilot-ingestion .
# Run:
#   docker run --rm -e POSTGRES_HOST=... -e POSTGRES_USER=... ... career-copilot-ingestion

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CAREER_COPILOT_ROOT=/app

COPY scripts/ingestion/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY . .

CMD ["python", "scripts/ingestion/run.py"]
