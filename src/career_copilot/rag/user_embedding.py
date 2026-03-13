"""User profile embedding indexing in Chroma."""

from __future__ import annotations

from pathlib import Path

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
    Store the user's profile embedding in a Chroma collection.

    Uses OpenAI text-embedding-3-large (same as jobs; see career_copilot.rag.embedding).
    Returns (collection_name, document_id).
    """
    import chromadb

    from career_copilot.rag.embedding import get_embedding_function
    from career_copilot.rag.user_embedding import truncate_for_embedding

    root = Path(__file__).resolve().parents[3]
    persist_path = root / "data" / "chroma"
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    collection_name = "user_profiles"
    ef = get_embedding_function()
    try:
        coll = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Career Copilot user profiles"},
            embedding_function=ef,
        )
    except ValueError as e:
        if "embedding function" in str(e).lower() and "conflict" in str(e).lower():
            client.delete_collection(name=collection_name)
            coll = client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Career Copilot user profiles"},
                embedding_function=ef,
            )
        else:
            raise

    # Resume is first so it is a central part of the user embedding; then preferences
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

    coll.upsert(ids=[document_id], documents=[document], metadatas=[{"user_id": user_id}])
    return collection_name, document_id
