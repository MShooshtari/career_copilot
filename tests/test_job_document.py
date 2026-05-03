"""Tests for career_copilot.rag.job_document."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from career_copilot.constants import RAG_JOB_DOC_MAX_CHARS
from career_copilot.ingestion.common import NormalizedJob
from career_copilot.rag.job_document import (
    JOB_DOC_MAX_CHARS,
    job_to_document,
    job_to_document_key,
    job_to_metadata,
)


def _nj(**kwargs: Any) -> NormalizedJob:
    base: dict[str, Any] = {
        "source": "remoteok",
        "source_id": "abc",
        "title": None,
        "company": None,
        "location": None,
        "salary_min": None,
        "salary_max": None,
        "description": None,
        "skills": None,
        "posted_at": None,
        "url": None,
        "raw": {},
        "db_id": None,
    }
    base.update(kwargs)
    return NormalizedJob(**base)


def test_job_to_document_empty() -> None:
    assert job_to_document(_nj()) == ""


def test_job_to_document_joins_sections() -> None:
    j = _nj(
        title="Engineer",
        company="Acme",
        location="Remote",
        description="Build things.",
        skills=["Python", "SQL"],
    )
    doc = job_to_document(j)
    assert "Engineer" in doc
    assert "Company: Acme" in doc
    assert "Location: Remote" in doc
    assert "Build things." in doc
    assert "Skills: Python, SQL" in doc


def test_job_to_document_includes_extracted_skills_without_duplicates() -> None:
    j = _nj(skills=["Python"], extracted_skills=["Python", "CI/CD"])

    assert "Skills: Python, CI/CD" in job_to_document(j)
    assert job_to_metadata(j)["extracted_skills"] == "Python, CI/CD"


def test_job_to_document_truncates_over_max_chars() -> None:
    long_desc = "x" * (JOB_DOC_MAX_CHARS + 50)
    j = _nj(title="T", description=long_desc)
    out = job_to_document(j, max_chars=100)
    assert len(out) == 101
    assert out.endswith("…")


def test_job_doc_max_chars_matches_constant() -> None:
    assert JOB_DOC_MAX_CHARS == RAG_JOB_DOC_MAX_CHARS


def test_job_to_metadata_minimal() -> None:
    j = _nj(source="s", title="T", url="https://example.com/j")
    m = job_to_metadata(j)
    assert m["source"] == "s"
    assert m["title"] == "T"
    assert m["company"] == ""
    assert m["location"] == ""
    assert m["url"] == "https://example.com/j"
    assert "source_id" in m and m["source_id"] == "abc"


def test_job_to_metadata_optional_fields() -> None:
    posted = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    j = _nj(
        salary_min=100_000,
        salary_max=150_000,
        posted_at=posted,
        skills=["Go"],
    )
    m = job_to_metadata(j)
    assert m["salary_min"] == 100_000
    assert m["salary_max"] == 150_000
    assert m["posted_at"] == posted.isoformat()
    assert m["skills"] == "Go"


def test_job_to_document_key_with_source_id() -> None:
    j = _nj(source="remotive", source_id="job-42")
    assert job_to_document_key(j) == "remotive:job-42"


def test_job_to_document_key_without_source_id_uses_url() -> None:
    j = _nj(source="x", source_id=None, url="https://u")
    assert job_to_document_key(j) == "x:https://u"
