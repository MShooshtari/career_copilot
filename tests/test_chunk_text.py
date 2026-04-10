"""Tests for description chunking."""

from __future__ import annotations

from career_copilot.rag.chunk_text import chunk_description


def test_chunk_description_empty() -> None:
    assert chunk_description("") == []
    assert chunk_description("   ") == []


def test_chunk_description_single_short() -> None:
    assert chunk_description("Hello world") == ["Hello world"]


def test_chunk_description_splits_long_paragraph() -> None:
    long = "word " * 800
    parts = chunk_description(long, max_chars=400, overlap=40)
    assert len(parts) >= 2
    assert all(len(p) <= 450 for p in parts)
