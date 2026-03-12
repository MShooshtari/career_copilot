"""Chroma-based RAG store for job listings (local persistence)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from career_copilot.ingestion.common import NormalizedJob
from career_copilot.rag.embedding import get_embedding_function


def _job_to_document(job: NormalizedJob) -> str:
    """Build a single searchable document string from a normalized job."""
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
    return "\n\n".join(parts) if parts else ""


def _job_to_metadata(job: NormalizedJob) -> dict[str, str | int | float | bool]:
    """Convert job to Chroma-safe metadata (str, int, float, bool only)."""
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


def _job_to_id(job: NormalizedJob) -> str:
    """Stable document id for upsert behavior."""
    sid = job.source_id or ""
    return f"{job.source}:{sid}" if sid else f"{job.source}:{job.url or id(job)}"


def index_jobs_into_chroma(
    jobs: list[NormalizedJob],
    *,
    persist_path: str | Path | None = None,
    collection_name: str = "jobs",
) -> int:
    """
    Index normalized jobs into a local Chroma collection for RAG.

    Uses OpenAI text-embedding-3-large (set OPENAI_API_KEY). Data is persisted
    under `persist_path` for local runs.

    Args:
        jobs: List of normalized job records to index.
        persist_path: Directory for Chroma persistence. Defaults to
            project_root / "data" / "chroma".
        collection_name: Name of the Chroma collection.

    Returns:
        Number of documents added/updated in the collection.
    """
    import chromadb

    if not jobs:
        return 0

    if persist_path is None:
        # Default: project root / data / chroma
        root = Path(__file__).resolve().parents[3]
        persist_path = root / "data" / "chroma"
    persist_path = Path(persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    ef = get_embedding_function()
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Career Copilot job listings for RAG"},
            embedding_function=ef,
        )
    except ValueError as e:
        if "embedding function" in str(e).lower() and "conflict" in str(e).lower():
            client.delete_collection(name=collection_name)
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Career Copilot job listings for RAG"},
                embedding_function=ef,
            )
        else:
            raise

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for job in jobs:
        doc = _job_to_document(job)
        if not doc.strip():
            continue
        ids.append(_job_to_id(job))
        documents.append(doc)
        metadatas.append(_job_to_metadata(job))

    if not ids:
        return 0

    # Upsert: add or replace by id
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)
