"""Tests for career_copilot.database.jobs (pure helpers and with mocked DB)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from career_copilot.database.jobs import (
    _norm_sid,
    _rag_doc_id_to_source_source_id,
    delete_user_job,
    format_recommendation_jobs,
    get_job_feedback_map,
    get_job_interactions_map,
    insert_user_job,
    list_jobs_with_feedback,
    remove_job_feedback,
    resolve_job_ids,
    row_to_job_dict,
    row_to_job_dict_snippet,
    set_job_feedback,
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


# --- _rag_doc_id_to_source_source_id ---


def test_rag_doc_id_to_source_source_id_with_colon() -> None:
    assert _rag_doc_id_to_source_source_id("remotive:123") == ("remotive", "123")


def test_rag_doc_id_to_source_source_id_no_colon() -> None:
    assert _rag_doc_id_to_source_source_id("abc") == ("abc", None)


def test_rag_doc_id_to_source_source_id_multiple_colons() -> None:
    assert _rag_doc_id_to_source_source_id("a:b:c") == ("a", "b:c")


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
    extracted_skills=None,
    posted_at=None,
    url="https://example.com",
) -> tuple:
    if skills is None:
        skills = ["SourceTag"]
    if extracted_skills is None:
        extracted_skills = ["Python", "SQL"]
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
        extracted_skills,
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


def test_row_to_job_dict_uses_extracted_skills() -> None:
    row = _job_row(skills=["SourceTag"], extracted_skills=["Python", "SQL"])
    d = row_to_job_dict(row)
    assert d["skills"] == ["Python", "SQL"]


def test_row_to_job_dict_none_fields() -> None:
    row = (1, None, None, None, None, None, None, None, None, None, None, None, None)
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


def test_format_recommendation_jobs_prefers_postgres_job_id() -> None:
    raw = [
        {
            "id": "7",
            "postgres_job_id": 7,
            "metadata": {"source": "remotive", "source_id": "99", "title": "Dev", "company": "Co"},
            "document": "Text",
            "distance": 0.2,
        }
    ]
    out = format_recommendation_jobs(raw, {})
    assert out[0]["job_id"] == 7
    assert out[0]["title"] == "Dev"
    assert out[0]["company"] == "Co"
    assert out[0]["snippet"] == "Text"


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
            "metadata": {
                "source": "x",
                "source_id": "1",
                "title": "J",
                "skills": "SourceTag",
                "extracted_skills": "Python, Go",
                "ai_extracted_skills": "Python, Go",
            },
        }
    ]
    id_map = {("x", "1"): 1}
    out = format_recommendation_jobs(raw, id_map)
    assert out[0]["skills"] == ["SourceTag"]


def test_format_recommendation_jobs_falls_back_to_ai_extracted_skills_metadata() -> None:
    raw = [
        {
            "metadata": {
                "source": "x",
                "source_id": "1",
                "title": "J",
                "ai_extracted_skills": "Python, Go",
            },
        }
    ]
    id_map = {("x", "1"): 1}
    out = format_recommendation_jobs(raw, id_map)
    assert out[0]["skills"] == ["Python", " Go"]  # comma split keeps space after comma


def test_format_recommendation_jobs_falls_back_to_legacy_extracted_skills_metadata() -> None:
    raw = [
        {
            "metadata": {
                "source": "x",
                "source_id": "1",
                "title": "J",
                "extracted_skills": "LegacyTag",
            },
        }
    ]
    id_map = {("x", "1"): 1}
    out = format_recommendation_jobs(raw, id_map)
    assert out[0]["skills"] == ["LegacyTag"]


def test_format_recommendation_jobs_falls_back_to_source_skills_metadata() -> None:
    raw = [
        {
            "metadata": {
                "source": "x",
                "source_id": "1",
                "title": "J",
                "skills": "SourceTag",
            },
        }
    ]
    id_map = {("x", "1"): 1}
    out = format_recommendation_jobs(raw, id_map)
    assert out[0]["skills"] == ["SourceTag"]


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


# --- insert_user_job (mocked connection) ---


def test_insert_user_job_returns_id() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None
    cur.fetchone.return_value = (99,)

    out = insert_user_job(
        conn,
        1,
        title="Engineer",
        company="Acme",
        location="Remote",
        description="Do stuff.",
        skills=["Python"],
        url="https://example.com/job",
    )
    assert out == 99
    cur.execute.assert_called_once()
    query = cur.execute.call_args[0][0]
    assert "user_jobs" in (query if isinstance(query, str) else query.decode())


# --- delete_user_job (mocked connection) ---


def test_delete_user_job_returns_true_when_deleted() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None
    cur.rowcount = 1

    out = delete_user_job(conn, 1, 42)
    assert out is True
    cur.execute.assert_called_once()
    assert cur.execute.call_args[0][1] == (42, 1)


def test_delete_user_job_returns_false_when_none_deleted() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None
    cur.rowcount = 0

    out = delete_user_job(conn, 1, 999)
    assert out is False


def test_set_job_feedback_upserts_valid_feedback() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None

    set_job_feedback(conn, 1, 42, "ingested", "disliked")

    assert cur.execute.call_count == 2
    delete_query = cur.execute.call_args_list[0][0][0]
    assert "DELETE FROM user_job_interaction" in (
        delete_query if isinstance(delete_query, str) else delete_query.decode()
    )
    assert cur.execute.call_args_list[0][0][1] == (1, 42, "ingested", "liked")
    query = cur.execute.call_args_list[1][0][0]
    assert "user_job_interaction" in (query if isinstance(query, str) else query.decode())
    assert cur.execute.call_args_list[1][0][1] == (1, 42, "ingested", "disliked")


def test_set_job_feedback_does_not_clear_liked_for_applied() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None

    set_job_feedback(conn, 1, 42, "ingested", "applied")

    cur.execute.assert_called_once()
    query = cur.execute.call_args[0][0]
    assert "ON CONFLICT (user_id, job_id, job_source, feedback)" in (
        query if isinstance(query, str) else query.decode()
    )


def test_set_job_feedback_allows_deleted_marker() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None

    set_job_feedback(conn, 1, 42, "ingested", "deleted")

    cur.execute.assert_called_once()
    assert cur.execute.call_args[0][1] == (1, 42, "ingested", "deleted")


@pytest.mark.parametrize(
    "feedback",
    [
        "details_viewed",
        "resume_improvement_opened",
        "interview_preparation_opened",
    ],
)
def test_set_job_feedback_allows_navigation_interactions(feedback: str) -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None

    set_job_feedback(conn, 1, 42, "ingested", feedback)

    cur.execute.assert_called_once()
    assert cur.execute.call_args[0][1] == (1, 42, "ingested", feedback)


def test_remove_job_feedback_deletes_one_interaction() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None
    cur.rowcount = 1

    removed = remove_job_feedback(conn, 1, 42, "ingested", "applied")

    assert removed is True
    cur.execute.assert_called_once()
    assert cur.execute.call_args[0][1] == (1, 42, "ingested", "applied")


def test_get_job_feedback_map_returns_feedback_by_job_id() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None
    cur.fetchall.return_value = [(42, "liked"), (99, "disliked")]

    out = get_job_feedback_map(conn, 1, "ingested", [42, 99, 42])

    assert out == {42: "liked", 99: "disliked"}
    cur.execute.assert_called_once()
    assert cur.execute.call_args[0][1] == (1, "ingested", [42, 99])


def test_get_job_interactions_map_returns_multiple_interactions() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None
    cur.fetchall.return_value = [(42, "liked"), (42, "applied"), (99, "disliked")]

    out = get_job_interactions_map(conn, 1, "ingested", [42, 99, 42])

    assert out == {42: {"liked", "applied"}, 99: {"disliked"}}
    cur.execute.assert_called_once()
    assert cur.execute.call_args[0][1] == (1, "ingested", [42, 99])


def test_get_job_feedback_map_empty_ids_skips_db() -> None:
    conn = MagicMock()

    assert get_job_feedback_map(conn, 1, "ingested", []) == {}
    conn.cursor.assert_not_called()


def test_list_jobs_with_feedback_returns_action_urls() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None
    cur.fetchall.return_value = [
        (42, "ingested", "ML Engineer", "Acme", "Remote", "https://example.com/job", None),
        (7, "user", "Data Scientist", "Personal Co", "", "", None),
    ]

    out = list_jobs_with_feedback(conn, 1, "applied")

    assert out == [
        {
            "job_id": 42,
            "job_source": "ingested",
            "title": "ML Engineer",
            "company": "Acme",
            "location": "Remote",
            "url": "https://example.com/job",
            "detail_url": "/jobs/42",
            "resume_url": "/jobs/42/improve-resume",
            "interview_url": "/jobs/42/prepare-interview",
            "updated_at": None,
        },
        {
            "job_id": 7,
            "job_source": "user",
            "title": "Data Scientist",
            "company": "Personal Co",
            "location": "",
            "url": "",
            "detail_url": "/my-jobs/7",
            "resume_url": "/my-jobs/7/improve-resume",
            "interview_url": "/my-jobs/7/prepare-interview",
            "updated_at": None,
        },
    ]
    cur.execute.assert_called_once()
    assert cur.execute.call_args[0][1] == (1, "applied", 1, "applied")
