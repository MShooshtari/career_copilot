"""Tests for career_copilot.ingestion.common."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from career_copilot.ingestion.common import coerce_int, html_to_plain_text, parse_datetime


# --- html_to_plain_text ---


def test_html_to_plain_text_none() -> None:
    assert html_to_plain_text(None) is None


def test_html_to_plain_text_empty() -> None:
    assert html_to_plain_text("") is None
    assert html_to_plain_text("   ") is None


def test_html_to_plain_text_plain_only() -> None:
    assert html_to_plain_text("No HTML here") == "No HTML here"


def test_html_to_plain_text_strips_tags() -> None:
    assert html_to_plain_text("<p>Hello</p>") == "Hello"


def test_html_to_plain_text_block_tags_newline() -> None:
    out = html_to_plain_text("<p>One</p><p>Two</p>")
    assert "One" in out and "Two" in out
    assert "\n" in out


def test_html_to_plain_text_unescapes_entities() -> None:
    assert "&" in html_to_plain_text("&amp;")
    assert html_to_plain_text("&#x26;").strip() == "&" or "&" in html_to_plain_text("&#x26;")


# --- parse_datetime ---


def test_parse_datetime_none() -> None:
    assert parse_datetime(None) is None


def test_parse_datetime_empty_string() -> None:
    assert parse_datetime("") is None


def test_parse_datetime_iso_string() -> None:
    dt = parse_datetime("2025-01-15T12:00:00Z")
    assert dt is not None
    assert dt.year == 2025
    assert dt.month == 1
    assert dt.day == 15


def test_parse_datetime_iso_with_offset() -> None:
    dt = parse_datetime("2025-01-15T12:00:00+00:00")
    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_datetime_timestamp() -> None:
    ts = 1736942400  # 2025-01-15 12:00:00 UTC
    dt = parse_datetime(ts)
    assert dt is not None
    assert dt.tzinfo == timezone.utc


def test_parse_datetime_invalid_string() -> None:
    assert parse_datetime("not a date") is None


# --- coerce_int ---


def test_coerce_int_none() -> None:
    assert coerce_int(None) is None


def test_coerce_int_already_int() -> None:
    assert coerce_int(42) == 42


def test_coerce_int_string() -> None:
    assert coerce_int("99") == 99


def test_coerce_int_float() -> None:
    assert coerce_int(3.14) == 3


def test_coerce_int_invalid() -> None:
    assert coerce_int("abc") is None
    assert coerce_int({}) is None
