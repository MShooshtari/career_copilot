"""
Single source of truth for embeddings.

We use OpenAI's text-embedding-3-small (1536 dims) so pgvector HNSW indexes work
on common pgvector builds (HNSW has a dimensions limit).

Used by: pgvector in Postgres (jobs + user_embeddings), explore_embeddings.
"""

from __future__ import annotations

import os

from career_copilot.database.db import load_env

# So OPENAI_API_KEY from .env is available
load_env()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
# Vector size for text-embedding-3-small (pgvector columns + indexes must match).
EMBEDDING_VECTOR_DIMENSIONS = 1536

_EMBED_BATCH = 100


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed each non-empty string with text-embedding-3-small.

    Requires OPENAI_API_KEY. Order matches input (empty strings are skipped in batching
    only when entire batch is built per job — caller should not pass empty docs).
    """
    if not texts:
        return []
    if not os.environ.get(OPENAI_API_KEY_ENV):
        raise RuntimeError(
            f"{OPENAI_API_KEY_ENV} is not set. Add it to .env (see configs/config.example.env) "
            "so embeddings can be computed."
        )
    from openai import OpenAI

    client = OpenAI()
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        chunk = texts[i : i + _EMBED_BATCH]
        resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=chunk)
        ordered = sorted(resp.data, key=lambda d: d.index)
        out.extend(item.embedding for item in ordered)
    return out
