"""Tests for career_copilot.database.jobs (pure helpers and with mocked DB)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from career_copilot.database.jobs import (
    _chroma_id_to_source_source_id,
    _norm_sid,
    format_recommendation_jobs,
    resolve_job_ids,
    row_to_job_dict,
    row_to_job_dict_snippet,
)

# --- _norm_sid (test via import; used by resolve_job_ids and format_recommendation_jobs) ---


def test_norm_sid_none() -> None:
    assert _norm_sid(None) is None


def test_norm_sid_empty_string() -> None:
    assert _norm_sid("") is None


def test_norm_sid_string() -> None:
    assert _norm_sid("123") == "123"


def test_norm_sid_int() -> None:
    assert _norm_sid(42) == "42"


def test_norm_sid_float() -> None:
    assert _norm_sid(3.14) == "3"


# --- _chroma_id_to_source_source_id ---


def test_chroma_id_to_source_source_id_with_colon() -> None:
    assert _chroma_id_to_source_source_id("remotive:123") == ("remotive", "123")


def test_chroma_id_to_source_source_id_no_colon() -> None:
    assert _chroma_id_to_source_source_id("abc") == ("abc", None)


def test_chroma_id_to_source_source_id_multiple_colons() -> None:
    assert _chroma_id_to_source_source_id("a:b:c") == ("a", "b:c")


# --- row_to_job_dict ---


def _job_row(
    id_=1,
    source="remotive",
    source_id="123",
    title="Engineer",
    company="Acme",
    location="Remote",
    salary_min=100,
    salary_max=200,
    description="Do stuff",
    skills=None,
    posted_at=None,
    url="https://example.com",
) -> tuple:
    if skills is None:
        skills = ["Python", "SQL"]
    if posted_at is None:
        posted_at = datetime(2025, 1, 15)
    return (
        id_,
        source,
        source_id,
        title,
        company,
        location,
        salary_min,
        salary_max,
        description,
        skills,
        posted_at,
        url,
    )


def test_row_to_job_dict_full() -> None:
    row = _job_row()
    d = row_to_job_dict(row)
    assert d["id"] == 1
    assert d["source"] == "remotive"
    assert d["title"] == "Engineer"
    assert d["company"] == "Acme"
    assert d["description"] == "Do stuff"
    assert d["skills"] == ["Python", "SQL"]
    assert d["url"] == "https://example.com"


def test_row_to_job_dict_none_fields() -> None:
    row = (1, None, None, None, None, None, None, None, None, None, None, None)
    d = row_to_job_dict(row)
    assert d["title"] == "Job"
    assert d["company"] == ""
    assert d["description"] == ""
    assert d["skills"] == []
    assert d["url"] == ""


def test_row_to_job_dict_snippet_short_description() -> None:
    row = _job_row(description="Short.")
    d = row_to_job_dict_snippet(row, description_max_chars=500)
    assert d["description"] == "Short."


def test_row_to_job_dict_snippet_long_description() -> None:
    long_desc = "x" * 600
    row = _job_row(description=long_desc)
    d = row_to_job_dict_snippet(row, description_max_chars=500)
    assert len(d["description"]) == 501  # 500 + "…"
    assert d["description"].endswith("…")


# --- format_recommendation_jobs ---


def test_format_recommendation_jobs_empty() -> None:
    assert format_recommendation_jobs([], {}) == []


def test_format_recommendation_jobs_one() -> None:
    raw = [
        {
            "id": "remotive:99",
            "metadata": {"source": "remotive", "source_id": "99", "title": "Dev", "company": "Co"},
            "document": "Full text here.",
            "distance": 0.1,
        }
    ]
    id_map = {("remotive", "99"): 42}
    out = format_recommendation_jobs(raw, id_map)
    assert len(out) == 1
    assert out[0]["job_id"] == 42
    assert out[0]["title"] == "Dev"
    assert out[0]["company"] == "Co"
    assert out[0]["snippet"] == "Full text here."


def test_format_recommendation_jobs_snippet_truncated() -> None:
    long_doc = "a" * 500
    raw = [
        {
            "id": "remotive:1",
            "metadata": {"source": "remotive", "source_id": "1", "title": "Job"},
            "document": long_doc,
            "distance": 0.0,
        }
    ]
    id_map = {("remotive", "1"): 1}
    out = format_recommendation_jobs(raw, id_map, snippet_max_chars=100)
    assert len(out[0]["snippet"]) == 101
    assert out[0]["snippet"].endswith("…")


def test_format_recommendation_jobs_skills_from_metadata() -> None:
    raw = [
        {
            "metadata": {"source": "x", "source_id": "1", "title": "J", "skills": "Python, Go"},
        }
    ]
    id_map = {("x", "1"): 1}
    out = format_recommendation_jobs(raw, id_map)
    assert out[0]["skills"] == ["Python", " Go"]  # comma split keeps space after comma


# --- resolve_job_ids (mocked connection) ---


def test_resolve_job_ids_empty() -> None:
    conn = MagicMock()
    assert resolve_job_ids(conn, []) == {}
    conn.cursor.assert_not_called()


def test_resolve_job_ids_mock_returns_ids() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None

    # First call returns (10,), second returns (20,)
    cur.fetchone = MagicMock(side_effect=[(10,), (20,)])

    results = [
        {"metadata": {"source": "remotive", "source_id": "1"}},
        {"metadata": {"source": "remotive", "source_id": "2"}},
    ]
    out = resolve_job_ids(conn, results)
    assert out.get(("remotive", "1")) == 10
    assert out.get(("remotive", "2")) == 20
