"""User profile embedding indexing in Azure AI Search."""

from __future__ import annotations

from career_copilot.rag.azure_search_users import (
    DEFAULT_USER_INDEX_NAME,
    fetch_user_profile_embedding,
    upsert_user_profile_document,
)

# OpenAI text-embedding-3-large max context is 8192 tokens; ~4 chars/token → safe limit
EMBEDDING_MAX_CHARS = 28_000


def truncate_for_embedding(text: str, max_chars: int = EMBEDDING_MAX_CHARS) -> str:
    """Truncate text so it fits within the embedding model's token limit."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def index_user_embedding(
    *,
    user_id: int,
    resume_text: str,
    skill_tags: str,
    preferred_roles: str,
    industries: str,
    work_mode: str,
    employment_type: str,
    preferred_locations: str,
) -> tuple[str, str]:
    """
    Store the user's profile embedding in the Azure AI Search user profiles index.

    Uses OpenAI text-embedding-3-large (same as jobs; see career_copilot.rag.embedding).
    If Azure Search is not configured, this is a no-op.

    Returns (index_name, document_id) where document_id is ``user:{user_id}``.
    """
    pieces = [
        resume_text or "",
        f"Skills: {skill_tags}",
        f"Preferred roles: {preferred_roles}",
        f"Industries: {industries}",
        f"Work mode: {work_mode}",
        f"Employment type: {employment_type}",
        f"Preferred locations: {preferred_locations}",
    ]
    document = "\n\n".join(p for p in pieces if p.strip())
    document = truncate_for_embedding(document)
    document_id = f"user:{user_id}"

    upsert_user_profile_document(user_id, document)
    return DEFAULT_USER_INDEX_NAME, document_id


def get_user_profile_embedding_vector(user_id: int) -> list[float] | None:
    """Return the stored embedding vector for ``user:{user_id}`` in Azure AI Search, or None."""
    return fetch_user_profile_embedding(user_id)
