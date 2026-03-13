"""Chroma-based RAG store for job listings (local persistence)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from career_copilot.ingestion.common import NormalizedJob
from career_copilot.rag.embedding import get_embedding_function

# OpenAI allows max 300k tokens per request; batch to stay under that
# ~4 chars/token → cap doc length and batch size
JOB_DOC_MAX_CHARS = 6_000
JOB_UPSERT_BATCH_SIZE = 50


def _job_to_document(job: NormalizedJob, max_chars: int = JOB_DOC_MAX_CHARS) -> str:
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

    # Upsert in batches to stay under OpenAI max tokens per request (300k)
    total = 0
    for i in range(0, len(ids), JOB_UPSERT_BATCH_SIZE):
        batch_ids = ids[i : i + JOB_UPSERT_BATCH_SIZE]
        batch_docs = documents[i : i + JOB_UPSERT_BATCH_SIZE]
        batch_meta = metadatas[i : i + JOB_UPSERT_BATCH_SIZE]
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
        total += len(batch_ids)
    return total


def get_recommended_job_results(
    *,
    user_id: int = 1,
    n_results: int = 100,
    persist_path: str | Path | None = None,
    jobs_collection_name: str = "jobs",
    user_profiles_collection_name: str = "user_profiles",
) -> list[dict[str, Any]]:
    """
    Candidate retrieval: get top-n jobs by cosine similarity to the user's profile embedding.

    Fetches the user embedding from Chroma user_profiles, then queries the jobs
    collection with that vector. Chroma uses L2 distance; relative ordering is
    used for recommendations.

    Returns a list of dicts, each with:
      - id: Chroma document id (e.g. "remotive:123")
      - metadata: Chroma metadata (title, company, location, url, etc.)
      - document: full indexed document text (for snippet)
      - distance: Chroma distance (lower = more similar)
    """
    import chromadb

    if persist_path is None:
        root = Path(__file__).resolve().parents[3]
        persist_path = root / "data" / "chroma"
    persist_path = Path(persist_path)
    if not persist_path.exists():
        return []

    client = chromadb.PersistentClient(path=str(persist_path))
    ef = get_embedding_function()

    # Get user embedding
    try:
        user_coll = client.get_or_create_collection(
            name=user_profiles_collection_name,
            metadata={"description": "Career Copilot user profiles"},
            embedding_function=ef,
        )
    except ValueError as e:
        if "embedding function" in str(e).lower() and "conflict" in str(e).lower():
            client.delete_collection(name=user_profiles_collection_name)
            user_coll = client.get_or_create_collection(
                name=user_profiles_collection_name,
                metadata={"description": "Career Copilot user profiles"},
                embedding_function=ef,
            )
        else:
            raise

    user_doc_id = f"user:{user_id}"
    user_get = user_coll.get(ids=[user_doc_id], include=["embeddings"])
    ids_list = user_get.get("ids") or []
    embs = user_get.get("embeddings")
    if not ids_list or embs is None or len(embs) == 0:
        return []

    user_embedding = embs[0]

    # Get jobs collection and query by user embedding
    try:
        jobs_coll = client.get_or_create_collection(
            name=jobs_collection_name,
            metadata={"description": "Career Copilot job listings for RAG"},
            embedding_function=ef,
        )
    except ValueError as e:
        if "embedding function" in str(e).lower() and "conflict" in str(e).lower():
            client.delete_collection(name=jobs_collection_name)
            jobs_coll = client.get_or_create_collection(
                name=jobs_collection_name,
                metadata={"description": "Career Copilot job listings for RAG"},
                embedding_function=ef,
            )
        else:
            raise

    n = min(n_results, jobs_coll.count())
    if n == 0:
        return []

    results = jobs_coll.query(
        query_embeddings=[user_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    out: list[dict[str, Any]] = []
    ids_ = results["ids"][0]
    metadatas_ = results["metadatas"][0]
    documents_ = results["documents"][0]
    distances_ = results["distances"][0]

    for i, doc_id in enumerate(ids_):
        out.append(
            {
                "id": doc_id,
                "metadata": metadatas_[i] if i < len(metadatas_) else {},
                "document": documents_[i] if i < len(documents_) else "",
                "distance": distances_[i] if i < len(distances_) else None,
            }
        )
    return out


def get_similar_jobs_for_resume_improvement(
    job_document: str,
    *,
    n_results: int = 5,
    persist_path: str | Path | None = None,
    jobs_collection_name: str = "jobs",
) -> list[dict[str, Any]]:
    """
    Retrieve jobs similar to the given job document (for RAG context in resume improvement).
    Uses the jobs Chroma collection; query by text so the embedding is computed from job_document.
    """
    import chromadb

    if not job_document or not job_document.strip():
        return []
    if persist_path is None:
        root = Path(__file__).resolve().parents[3]
        persist_path = root / "data" / "chroma"
    persist_path = Path(persist_path)
    if not persist_path.exists():
        return []

    client = chromadb.PersistentClient(path=str(persist_path))
    ef = get_embedding_function()
    try:
        jobs_coll = client.get_or_create_collection(
            name=jobs_collection_name,
            metadata={"description": "Career Copilot job listings for RAG"},
            embedding_function=ef,
        )
    except ValueError as e:
        if "embedding function" in str(e).lower() and "conflict" in str(e).lower():
            client.delete_collection(name=jobs_collection_name)
            jobs_coll = client.get_or_create_collection(
                name=jobs_collection_name,
                metadata={"description": "Career Copilot job listings for RAG"},
                embedding_function=ef,
            )
        else:
            raise

    n = min(n_results, max(1, jobs_coll.count()))
    if n == 0:
        return []
    results = jobs_coll.query(
        query_texts=[job_document[:JOB_DOC_MAX_CHARS]],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict[str, Any]] = []
    ids_ = results["ids"][0]
    metadatas_ = results["metadatas"][0]
    documents_ = results["documents"][0]
    distances_ = results["distances"][0]
    for i, doc_id in enumerate(ids_):
        out.append(
            {
                "id": doc_id,
                "metadata": metadatas_[i] if i < len(metadatas_) else {},
                "document": documents_[i] if i < len(documents_) else "",
                "distance": distances_[i] if i < len(distances_) else None,
            }
        )
    return out


def get_similar_resumes_for_resume_improvement(
    job_document: str,
    *,
    exclude_user_id: int | None = None,
    n_results: int = 5,
    persist_path: str | Path | None = None,
    user_profiles_collection_name: str = "user_profiles",
) -> list[dict[str, Any]]:
    """
    Retrieve user profiles (resume-like documents) similar to the job for RAG context.
    Optionally exclude one user_id (e.g. current user). Returns list of dicts with id, metadata, document, distance.
    """
    import chromadb

    if not job_document or not job_document.strip():
        return []
    if persist_path is None:
        root = Path(__file__).resolve().parents[3]
        persist_path = root / "data" / "chroma"
    persist_path = Path(persist_path)
    if not persist_path.exists():
        return []

    client = chromadb.PersistentClient(path=str(persist_path))
    ef = get_embedding_function()
    try:
        user_coll = client.get_or_create_collection(
            name=user_profiles_collection_name,
            metadata={"description": "Career Copilot user profiles"},
            embedding_function=ef,
        )
    except ValueError as e:
        if "embedding function" in str(e).lower() and "conflict" in str(e).lower():
            client.delete_collection(name=user_profiles_collection_name)
            user_coll = client.get_or_create_collection(
                name=user_profiles_collection_name,
                metadata={"description": "Career Copilot user profiles"},
                embedding_function=ef,
            )
        else:
            raise

    # Request extra so we can drop excluded user and still have n_results
    n_query = n_results + (1 if exclude_user_id is not None else 0)
    n_query = min(n_query, max(1, user_coll.count()))
    if n_query == 0:
        return []
    results = user_coll.query(
        query_texts=[job_document[:JOB_DOC_MAX_CHARS]],
        n_results=n_query,
        include=["documents", "metadatas", "distances"],
    )
    ids_ = results["ids"][0]
    metadatas_ = results["metadatas"][0]
    documents_ = results["documents"][0]
    distances_ = results["distances"][0]
    out: list[dict[str, Any]] = []
    for i, doc_id in enumerate(ids_):
        if exclude_user_id is not None:
            meta = metadatas_[i] if i < len(metadatas_) else {}
            if str(meta.get("user_id")) == str(exclude_user_id):
                continue
        out.append(
            {
                "id": doc_id,
                "metadata": metadatas_[i] if i < len(metadatas_) else {},
                "document": documents_[i] if i < len(documents_) else "",
                "distance": distances_[i] if i < len(distances_) else None,
            }
        )
        if len(out) >= n_results:
            break
    return out
