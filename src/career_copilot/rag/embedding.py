"""
Single source of truth for embeddings: all Chroma collections (jobs and user
profiles) use OpenAI text-embedding-3-large via this module.

Used by: chroma_store (job indexing), web_app (profile embedding), explore_embeddings.
"""
from __future__ import annotations

import os

from career_copilot.database.db import load_env

# So OPENAI_API_KEY from .env is available
load_env()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


def get_embedding_function():
    """
    Return Chroma embedding function using OpenAI text-embedding-3-large.

    All embeddings (jobs and user profiles) use this; requires OPENAI_API_KEY
    in the environment (e.g. in .env).
    """
    if not os.environ.get(OPENAI_API_KEY_ENV):
        raise RuntimeError(
            f"{OPENAI_API_KEY_ENV} is not set. Add it to .env (see .env.example) "
            "so that job and profile embeddings can use the OpenAI API."
        )
    import chromadb.utils.embedding_functions as embedding_functions

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key_env_var=OPENAI_API_KEY_ENV,
        model_name=OPENAI_EMBEDDING_MODEL,
    )
