"""
Azure AI Search: user profile index (vectors for recommendations and resume-improvement RAG).

Uses the same service credentials as job search:
  AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY
Optional: AZURE_SEARCH_USER_INDEX_NAME (default career-copilot-user-profiles)

Embeddings: OpenAI text-embedding-3-large via career_copilot.rag.embedding.embed_texts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
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

from career_copilot.constants import RAG_JOB_DOC_MAX_CHARS, RAG_SIMILAR_RESUMES_N_RESULTS
from career_copilot.database.db import load_env
from career_copilot.rag.embedding import EMBEDDING_VECTOR_DIMENSIONS, embed_texts

load_env()

_ENV_ENDPOINT = "AZURE_SEARCH_ENDPOINT"
_ENV_KEY = "AZURE_SEARCH_API_KEY"


def _endpoint_and_key() -> tuple[str, str]:
    endpoint = (os.environ.get(_ENV_ENDPOINT) or "").strip().rstrip("/")
    key = (os.environ.get(_ENV_KEY) or "").strip()
    if not endpoint or not key:
        raise RuntimeError(
            f"{_ENV_ENDPOINT} and {_ENV_KEY} must be set (see configs/config.example.env)."
        )
    return endpoint, key


def user_profiles_search_configured() -> bool:
    """True when Azure Search endpoint and API key are set (same credentials as the jobs index)."""
    try:
        _endpoint_and_key()
    except RuntimeError:
        return False
    return True

DEFAULT_USER_INDEX_NAME = "career-copilot-user-profiles"
_ENV_USER_INDEX = "AZURE_SEARCH_USER_INDEX_NAME"
USER_VECTOR_FIELD = "embedding"
USER_CONTENT_FIELD = "content"
USER_KEY_FIELD = "user_id"
USER_HNSW_CONFIG = "user-profiles-hnsw-config"
USER_VECTOR_PROFILE = "user-profiles-hnsw-profile"


def _user_index_name() -> str:
    return (os.environ.get(_ENV_USER_INDEX) or "").strip() or DEFAULT_USER_INDEX_NAME


def _user_index_client() -> SearchIndexClient:
    endpoint, key = _endpoint_and_key()
    return SearchIndexClient(endpoint, AzureKeyCredential(key))


def _user_search_client() -> SearchClient:
    endpoint, key = _endpoint_and_key()
    return SearchClient(endpoint, _user_index_name(), AzureKeyCredential(key))


def get_user_profiles_search_client() -> SearchClient:
    """Return a ``SearchClient`` for the configured user profiles index (requires Azure env vars)."""
    if not user_profiles_search_configured():
        raise RuntimeError(
            f"{_ENV_ENDPOINT} and {_ENV_KEY} must be set (see configs/config.example.env)."
        )
    return _user_search_client()


def vector_search_user_profiles(vector: list[float], *, top_k: int) -> list[dict[str, Any]]:
    """Nearest user profile documents by vector (same score shape as job vector search)."""
    if not user_profiles_search_configured():
        return []
    client = _user_search_client()
    vq = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top_k,
        fields=USER_VECTOR_FIELD,
    )
    results = client.search(
        search_text=None,
        vector_queries=[vq],
        select=[USER_KEY_FIELD, USER_CONTENT_FIELD],
        top=top_k,
    )
    out: list[dict[str, Any]] = []
    for hit in results:
        d = dict(hit)
        out.append(_hit_to_similar_resume(d))
    return out


def users_index_definition(index_name: str) -> SearchIndex:
    fields = [
        SimpleField(
            name=USER_KEY_FIELD,
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name=USER_CONTENT_FIELD,
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="en.microsoft",
        ),
        SearchField(
            name=USER_VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_VECTOR_DIMENSIONS,
            vector_search_profile_name=USER_VECTOR_PROFILE,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=USER_HNSW_CONFIG,
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
                name=USER_VECTOR_PROFILE,
                algorithm_configuration_name=USER_HNSW_CONFIG,
            )
        ],
    )
    return SearchIndex(name=index_name, fields=fields, vector_search=vector_search)


def ensure_users_index() -> None:
    """Create the user profiles index if it does not exist."""
    name = _user_index_name()
    client = _user_index_client()
    names = {x.name for x in client.list_indexes()}
    if name in names:
        return
    client.create_index(users_index_definition(name))


def upsert_user_profile_document(user_id: int, content: str) -> None:
    """Embed ``content`` and merge-or-upload one document keyed by ``user_id``."""
    if not user_profiles_search_configured():
        return
    ensure_users_index()
    client = _user_search_client()
    embeddings = embed_texts([content])
    row = {
        USER_KEY_FIELD: str(user_id),
        USER_CONTENT_FIELD: content,
        USER_VECTOR_FIELD: embeddings[0],
    }
    client.merge_or_upload_documents([row])


def fetch_user_profile_embedding(user_id: int) -> list[float] | None:
    """Return the stored embedding vector for ``user_id``, or None if missing."""
    if not user_profiles_search_configured():
        return None
    client = _user_search_client()
    try:
        doc = client.get_document(key=str(user_id), selected_fields=[USER_VECTOR_FIELD])
    except ResourceNotFoundError:
        return None
    except Exception:
        return None
    emb = doc.get(USER_VECTOR_FIELD)
    if not emb:
        return None
    return list(emb)


def _score_to_distance(score: float | None) -> float | None:
    if score is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if s <= 0:
        return None
    return 1.0 / (s + 1e-9)


def _hit_to_similar_resume(hit: dict[str, Any]) -> dict[str, Any]:
    uid = hit.get(USER_KEY_FIELD) or ""
    doc = hit.get(USER_CONTENT_FIELD) or ""
    score = hit.get("@search.score")
    meta: dict[str, Any] = {}
    if uid:
        try:
            meta["user_id"] = int(uid)
        except (TypeError, ValueError):
            meta["user_id"] = uid
    return {
        "id": f"user:{uid}" if uid else "",
        "metadata": meta,
        "document": doc,
        "distance": _score_to_distance(score),
    }


def get_similar_resumes_for_resume_improvement(
    job_document: str,
    *,
    exclude_user_id: int | None = None,
    n_results: int = RAG_SIMILAR_RESUMES_N_RESULTS,
    persist_path: str | Path | None = None,
    user_profiles_collection_name: str = "user_profiles",
) -> list[dict[str, Any]]:
    """
    User profiles most similar to ``job_document`` (vector search on the user index).

    ``persist_path`` and ``user_profiles_collection_name`` are unused; kept for API compatibility.
    """
    _ = persist_path, user_profiles_collection_name
    if not user_profiles_search_configured():
        return []
    if not job_document or not job_document.strip():
        return []

    ensure_users_index()

    text = job_document[:RAG_JOB_DOC_MAX_CHARS].strip()
    vec = embed_texts([text])[0]
    client = _user_search_client()

    want = max(1, n_results)
    fetch_k = want + 1 if exclude_user_id is not None else want
    fetch_k = min(fetch_k, 50)

    flt: str | None = None
    if exclude_user_id is not None:
        flt = f"{USER_KEY_FIELD} ne '{int(exclude_user_id)}'"

    vq = VectorizedQuery(
        vector=vec,
        k_nearest_neighbors=fetch_k,
        fields=USER_VECTOR_FIELD,
    )
    results = client.search(
        search_text=None,
        vector_queries=[vq],
        filter=flt,
        select=[USER_KEY_FIELD, USER_CONTENT_FIELD],
        top=fetch_k,
    )

    out: list[dict[str, Any]] = []
    for hit in results:
        d = dict(hit)
        uid_raw = d.get(USER_KEY_FIELD)
        if exclude_user_id is not None and uid_raw is not None:
            try:
                if int(uid_raw) == int(exclude_user_id):
                    continue
            except (TypeError, ValueError):
                pass
        out.append(_hit_to_similar_resume(d))
        if len(out) >= want:
            break
    return out
