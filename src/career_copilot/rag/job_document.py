"""Build searchable text, metadata, and keys for job RAG (pgvector / embedding text)."""

from __future__ import annotations

from career_copilot.constants import RAG_JOB_DOC_MAX_CHARS
from career_copilot.ingestion.common import NormalizedJob

JOB_DOC_MAX_CHARS = RAG_JOB_DOC_MAX_CHARS


def _analysis_skills(job: NormalizedJob) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    source_skills = job.ai_extracted_skills or job.extracted_skills or job.skills or []
    for skill in source_skills:
        key = skill.casefold()
        if key not in seen:
            seen.add(key)
            out.append(skill)
    return out


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
    skills = _analysis_skills(job)
    if skills:
        parts.append("Skills: " + ", ".join(skills))
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
    skills = _analysis_skills(job)
    if skills:
        meta["skills"] = ", ".join(skills)
        meta["ai_extracted_skills"] = ", ".join(skills)
        meta["extracted_skills"] = ", ".join(skills)
    return meta


def job_to_document_key(job: NormalizedJob) -> str:
    """Stable document id when Postgres id is unavailable (e.g. pre-insert)."""
    sid = job.source_id or ""
    return f"{job.source}:{sid}" if sid else f"{job.source}:{job.url or id(job)}"
