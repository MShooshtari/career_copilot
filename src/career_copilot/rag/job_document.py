"""Build searchable text, metadata, and keys for job RAG (pgvector / embedding text)."""

from __future__ import annotations

from typing import Any

from career_copilot.constants import RAG_JOB_DOC_MAX_CHARS
from career_copilot.ingestion.common import NormalizedJob

JOB_DOC_MAX_CHARS = RAG_JOB_DOC_MAX_CHARS


def job_to_document(job: NormalizedJob, max_chars: int = JOB_DOC_MAX_CHARS) -> str:
    """Build a single searchable document string from a normalized job (truncated for API limit)."""
    parts = []
    if job.title:
        parts.append(job.title)
    if job.company:
        parts.append(f"Company: {job.company}")
    if job.location:
        parts.append(f"Location: {job.location}")
    if job.description:
        parts.append(job.description)
    if job.skills:
        parts.append("Skills: " + ", ".join(job.skills))
    doc = "\n\n".join(parts) if parts else ""
    if len(doc) > max_chars:
        doc = doc[:max_chars].rstrip() + "…"
    return doc


def job_to_metadata(job: NormalizedJob) -> dict[str, str | int | float | bool]:
    """Metadata for templates and Postgres resolution (source, source_id, title, …)."""
    meta: dict[str, str | int | float | bool] = {
        "source": job.source,
        "title": job.title or "",
        "company": job.company or "",
        "location": job.location or "",
        "url": job.url or "",
    }
    if job.source_id:
        meta["source_id"] = job.source_id
    if job.salary_min is not None:
        meta["salary_min"] = job.salary_min
    if job.salary_max is not None:
        meta["salary_max"] = job.salary_max
    if job.posted_at is not None:
        meta["posted_at"] = job.posted_at.isoformat()
    if job.skills:
        meta["skills"] = ", ".join(job.skills)
    return meta


def job_to_document_key(job: NormalizedJob) -> str:
    """Stable document id when Postgres id is unavailable (e.g. pre-insert)."""
    sid = job.source_id or ""
    return f"{job.source}:{sid}" if sid else f"{job.source}:{job.url or id(job)}"
