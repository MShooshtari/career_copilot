"""Shared embedding function for Chroma (OpenAI text-embedding-3-large)."""
from __future__ import annotations

from career_copilot.database.db import load_env

# So OPENAI_API_KEY from .env is available
load_env()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


def get_embedding_function():
    """
    Return Chroma embedding function using OpenAI text-embedding-3-large.

    Requires OPENAI_API_KEY to be set (e.g. in .env).
    """
    import chromadb.utils.embedding_functions as embedding_functions

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key_env_var=OPENAI_API_KEY_ENV,
        model_name=OPENAI_EMBEDDING_MODEL,
    )
