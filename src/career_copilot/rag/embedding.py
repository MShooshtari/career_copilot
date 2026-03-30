"""
Single source of truth for embeddings: OpenAI text-embedding-3-large.

Used by: Azure AI Search (jobs and user profiles), explore_embeddings.
"""

from __future__ import annotations

import os

from career_copilot.database.db import load_env

# So OPENAI_API_KEY from .env is available
load_env()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
# Default vector size for text-embedding-3-large (Azure Search index definitions must match).
EMBEDDING_VECTOR_DIMENSIONS = 3072

_EMBED_BATCH = 100


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed each non-empty string with text-embedding-3-large.

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
