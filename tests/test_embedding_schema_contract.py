"""Contract tests: pgvector column dimensions stay aligned with embedding config."""

from __future__ import annotations

import inspect

from career_copilot.database.schema import init_schema
from career_copilot.rag.embedding import EMBEDDING_VECTOR_DIMENSIONS


def test_embedding_dimension_is_openai_small_model() -> None:
    assert EMBEDDING_VECTOR_DIMENSIONS == 1536


def test_init_schema_defines_vector_columns_with_embedding_constant() -> None:
    """DDL must interpolate EMBEDDING_VECTOR_DIMENSIONS (not a stray literal)."""
    src = inspect.getsource(init_schema)
    assert "vector({EMBEDDING_VECTOR_DIMENSIONS})" in src
