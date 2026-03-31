FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# System deps for PDF/text extraction libs (PyMuPDF, etc.) are bundled in wheels for many platforms,
# but keep image slim; add OS deps later only if your build/runtime needs them.

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY src/ src/
COPY templates/ templates/

EXPOSE 8000

# Azure Container Apps commonly provides $PORT; default to 8000 for local.
CMD ["sh", "-c", "uvicorn career_copilot.web_app:app --host 0.0.0.0 --port ${PORT:-8000}"]

