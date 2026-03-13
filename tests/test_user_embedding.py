"""Tests for career_copilot.rag.user_embedding."""

from __future__ import annotations

from career_copilot.rag.user_embedding import EMBEDDING_MAX_CHARS, truncate_for_embedding


def test_truncate_empty() -> None:
    assert truncate_for_embedding("") == ""


def test_truncate_none_length_under_limit() -> None:
    short = "hello"
    assert truncate_for_embedding(short) == short


def test_truncate_under_limit() -> None:
    text = "x" * (EMBEDDING_MAX_CHARS - 1)
    assert truncate_for_embedding(text) == text


def test_truncate_exactly_at_limit() -> None:
    text = "x" * EMBEDDING_MAX_CHARS
    assert truncate_for_embedding(text) == text


def test_truncate_over_limit() -> None:
    text = "a" * (EMBEDDING_MAX_CHARS + 100)
    out = truncate_for_embedding(text)
    assert len(out) == EMBEDDING_MAX_CHARS + 1  # max_chars + "…"
    assert out.endswith("…")
    assert out == "a" * EMBEDDING_MAX_CHARS + "…"


def test_truncate_custom_max() -> None:
    text = "hello world"
    assert truncate_for_embedding(text, max_chars=5) == "hello…"
