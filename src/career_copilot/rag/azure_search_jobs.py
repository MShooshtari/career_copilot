"""
Azure AI Search: job index (vector + metadata) for RAG.

Env:
  AZURE_SEARCH_ENDPOINT — https://<service>.search.windows.net
  AZURE_SEARCH_API_KEY — admin key (index + upload)
  AZURE_SEARCH_INDEX_NAME — optional, default career-copilot-jobs

Uses OpenAI text-embedding-3-large (same as user profiles in Azure AI Search); set OPENAI_API_KEY.
"""

from __future__ import annotations

import os
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from career_copilot.constants import (
    DEFAULT_USER_ID,
    RAG_DEFAULT_RECOMMENDATION_N_RESULTS,
    RAG_JOB_UPSERT_BATCH_SIZE,
    RAG_SIMILAR_JOBS_N_RESULTS,
)
from career_copilot.database.db import load_env
from career_copilot.ingestion.common import NormalizedJob
from career_copilot.rag.embedding import EMBEDDING_VECTOR_DIMENSIONS, embed_texts
from career_copilot.rag.job_document import job_to_document, job_to_metadata
from career_copilot.rag.user_embedding import get_user_profile_embedding_vector

load_env()

DEFAULT_INDEX_NAME = "career-copilot-jobs"
VECTOR_PROFILE = "jobs-hnsw-profile"
HNSW_CONFIG = "jobs-hnsw-config"
VECTOR_FIELD = "embedding"
CONTENT_FIELD = "content"

_ENV_ENDPOINT = "AZURE_SEARCH_ENDPOINT"
_ENV_KEY = "AZURE_SEARCH_API_KEY"
_ENV_INDEX = "AZURE_SEARCH_INDEX_NAME"


def _azure_config() -> tuple[str, str, str]:
    endpoint = (os.environ.get(_ENV_ENDPOINT) or "").strip().rstrip("/")
    key = (os.environ.get(_ENV_KEY) or "").strip()
    name = (os.environ.get(_ENV_INDEX) or "").strip() or DEFAULT_INDEX_NAME
    if not endpoint or not key:
        raise RuntimeError(
            f"{_ENV_ENDPOINT} and {_ENV_KEY} must be set (see configs/config.example.env)."
        )
    return endpoint, key, name


def azure_search_configured() -> bool:
    """True when endpoint and API key are set (read path can run)."""
    try:
        _azure_config()
    except RuntimeError:
        return False
    return True


def _index_client() -> SearchIndexClient:
    endpoint, key, _ = _azure_config()
    return SearchIndexClient(endpoint, AzureKeyCredential(key))


def _search_client() -> SearchClient:
    endpoint, key, index_name = _azure_config()
    return SearchClient(endpoint, index_name, AzureKeyCredential(key))


def get_jobs_search_client() -> SearchClient:
    """Return a ``SearchClient`` for the configured jobs index (requires Azure env vars)."""
    return _search_client()


def jobs_index_definition(index_name: str) -> SearchIndex:
    fields = [
        SimpleField(name="job_id", type=SearchFieldDataType.String, key=True),
        SearchableField(
            name=CONTENT_FIELD,
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="en.microsoft",
        ),
        SearchableField(name="title", type=SearchFieldDataType.String, searchable=True),
        SearchableField(name="company", type=SearchFieldDataType.String, searchable=True),
        SearchableField(name="location", type=SearchFieldDataType.String, searchable=True),
        SimpleField(name="url", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="salary_min", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="salary_max", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="posted_at", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="skills", type=SearchFieldDataType.String, searchable=True),
        SearchField(
            name=VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_VECTOR_DIMENSIONS,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=HNSW_CONFIG,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric="cosine",
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE,
                algorithm_configuration_name=HNSW_CONFIG,
            )
        ],
    )
    return SearchIndex(name=index_name, fields=fields, vector_search=vector_search)


def ensure_jobs_index() -> None:
    """Create the jobs index if it does not exist."""
    _, _, index_name = _azure_config()
    client = _index_client()
    names = {x.name for x in client.list_indexes()}
    if index_name in names:
        return
    client.create_index(jobs_index_definition(index_name))


def _score_to_distance(score: float | None) -> float | None:
    """Map Azure @search.score (higher = better) to a distance-like score (lower = better)."""
    if score is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if s <= 0:
        return None
    return 1.0 / (s + 1e-9)


def _hit_to_result(hit: dict[str, Any]) -> dict[str, Any]:
    """Normalize a search result to the shape used by recommendations / resume improvement."""
    score = hit.get("@search.score")
    meta = {
        "source": hit.get("source") or "",
        "source_id": hit.get("source_id") or "",
        "title": hit.get("title") or "",
        "company": hit.get("company") or "",
        "location": hit.get("location") or "",
        "url": hit.get("url") or "",
        "salary_min": hit.get("salary_min"),
        "salary_max": hit.get("salary_max"),
        "skills": hit.get("skills") or "",
    }
    if hit.get("posted_at"):
        meta["posted_at"] = hit["posted_at"]
    doc = hit.get(CONTENT_FIELD) or ""
    job_id = hit.get("job_id") or ""
    postgres_job_id: int | None = None
    if job_id:
        try:
            postgres_job_id = int(job_id)
        except (TypeError, ValueError):
            postgres_job_id = None
    return {
        "id": job_id,
        "postgres_job_id": postgres_job_id,
        "metadata": meta,
        "document": doc,
        "distance": _score_to_distance(score),
    }


def vector_search_jobs(
    vector: list[float],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    client = _search_client()
    vq = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top_k,
        fields=VECTOR_FIELD,
    )
    results = client.search(
        search_text=None,
        vector_queries=[vq],
        select=[
            "job_id",
            CONTENT_FIELD,
            "title",
            "company",
            "location",
            "url",
            "source",
            "source_id",
            "salary_min",
            "salary_max",
            "posted_at",
            "skills",
        ],
        top=top_k,
    )
    out: list[dict[str, Any]] = []
    for hit in results:
        d = dict(hit)
        out.append(_hit_to_result(d))
    return out


def index_jobs_into_azure_search(jobs: list[NormalizedJob]) -> int:
    """
    Upsert normalized jobs into Azure AI Search (OpenAI embeddings).

    Each job must have ``db_id`` set (Postgres ``jobs.id``) for the document key.
    """
    if not jobs:
        return 0

    ensure_jobs_index()
    client = _search_client()

    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for job in jobs:
        if job.db_id is None:
            raise ValueError("index_jobs_into_azure_search requires NormalizedJob.db_id (Postgres id)")
        doc = job_to_document(job)
        if not doc.strip():
            continue
        meta = job_to_metadata(job)
        doc_row: dict[str, Any] = {
            "job_id": str(job.db_id),
            CONTENT_FIELD: doc,
            "title": meta.get("title", ""),
            "company": meta.get("company", ""),
            "location": meta.get("location", ""),
            "url": meta.get("url", ""),
            "source": meta.get("source", ""),
            "source_id": str(meta.get("source_id", "") or ""),
            "skills": meta.get("skills", "") or "",
        }
        if meta.get("salary_min") is not None:
            doc_row["salary_min"] = int(meta["salary_min"])
        if meta.get("salary_max") is not None:
            doc_row["salary_max"] = int(meta["salary_max"])
        if meta.get("posted_at"):
            doc_row["posted_at"] = str(meta["posted_at"])
        rows.append(doc_row)
        texts.append(doc)

    if not rows:
        return 0

    embeddings = embed_texts(texts)
    if len(embeddings) != len(rows):
        raise RuntimeError("Embedding count mismatch")

    for r, emb in zip(rows, embeddings, strict=True):
        r[VECTOR_FIELD] = emb

    total = 0
    for i in range(0, len(rows), RAG_JOB_UPSERT_BATCH_SIZE):
        batch = rows[i : i + RAG_JOB_UPSERT_BATCH_SIZE]
        client.merge_or_upload_documents(batch)
        total += len(batch)
    return total


def get_recommended_job_results(
    *,
    user_id: int = DEFAULT_USER_ID,
    n_results: int = RAG_DEFAULT_RECOMMENDATION_N_RESULTS,
    jobs_collection_name: str = "jobs",  # unused; kept for API compatibility
) -> list[dict[str, Any]]:
    """
    Top job documents by vector similarity to the user's profile embedding (Azure user index).
    """
    _ = jobs_collection_name
    if not azure_search_configured():
        return []
    user_embedding = get_user_profile_embedding_vector(user_id)
    if user_embedding is None:
        return []
    k = max(1, n_results)
    return vector_search_jobs(user_embedding, top_k=k)


def get_similar_jobs_for_resume_improvement(
    job_document: str,
    *,
    n_results: int = RAG_SIMILAR_JOBS_N_RESULTS,
    jobs_collection_name: str = "jobs",  # unused
) -> list[dict[str, Any]]:
    """Jobs similar to the given document text (embedding computed via OpenAI)."""
    _ = jobs_collection_name
    if not azure_search_configured():
        return []
    if not job_document or not job_document.strip():
        return []
    vec = embed_texts([job_document.strip()])[0]
    k = max(1, n_results)
    return vector_search_jobs(vec, top_k=k)
