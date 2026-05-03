"""Tests for market analysis in-memory caches."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from career_copilot import market_analysis_service as service
from career_copilot.market_analysis_service import MarketCohortFilters


def setup_function() -> None:
    service.clear_market_analysis_caches()


def teardown_function() -> None:
    service.clear_market_analysis_caches()


def test_build_market_analysis_report_caches_by_user_filters_limit_and_rag_flag() -> None:
    conn = MagicMock()
    aggregates = {
        "weekly_posted": [{"week_start": "2026-04-27", "count": 1}],
        "top_skills": [{"skill": "python", "count": 1}],
        "salary": {},
        "top_locations": [],
    }

    with (
        patch.object(
            service, "cohort_job_ids", return_value=([123], {"filtered_count": 1})
        ) as cohort,
        patch.object(service, "_aggregates_for_cohort", return_value=aggregates) as aggregate,
        patch.object(service, "list_user_skills_lower", return_value=["python"]),
    ):
        first = service.build_market_analysis_report(
            conn,
            user_id=7,
            filters=MarketCohortFilters(title_contains=" Engineer "),
            cohort_limit=500,
            include_rag=False,
        )
        first["cohort"]["size"] = 99

        second = service.build_market_analysis_report(
            conn,
            user_id=7,
            filters=MarketCohortFilters(title_contains="engineer"),
            cohort_limit=500,
            include_rag=False,
        )

    assert second["cohort"]["size"] == 1
    cohort.assert_called_once()
    aggregate.assert_called_once()


def test_build_market_analysis_report_keeps_include_rag_in_cache_key() -> None:
    conn = MagicMock()
    aggregates = {
        "weekly_posted": [],
        "top_skills": [],
        "salary": {},
        "top_locations": [],
    }

    with (
        patch.object(service, "cohort_job_ids", return_value=([], {"filtered_count": 0})) as cohort,
        patch.object(service, "_aggregates_for_cohort", return_value=aggregates),
        patch.object(service, "list_user_skills_lower", return_value=[]),
    ):
        service.build_market_analysis_report(
            conn,
            user_id=7,
            filters=MarketCohortFilters(),
            cohort_limit=500,
            include_rag=False,
        )
        service.build_market_analysis_report(
            conn,
            user_id=7,
            filters=MarketCohortFilters(),
            cohort_limit=500,
            include_rag=True,
        )

    assert cohort.call_count == 2


def test_rag_query_embedding_uses_cache_for_same_profile_blurb() -> None:
    with patch.object(service, "embed_texts", return_value=[[0.1, 0.2]]) as embed:
        first = service._rag_query_embedding("Python backend engineer")
        first.append(0.3)
        second = service._rag_query_embedding("Python backend engineer")

    assert second == [0.1, 0.2]
    embed.assert_called_once()


def test_rag_query_embedding_reuses_stored_profile_version() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = ([0.1, 0.2],)

    with (
        patch.object(service, "_register"),
        patch.object(service, "embed_texts") as embed,
    ):
        result = service._rag_query_embedding(
            "Python backend engineer",
            conn=conn,
            user_id=7,
            profile_version="profile-v1",
        )

    assert result == [0.1, 0.2]
    embed.assert_not_called()


def test_rag_query_embedding_stores_profile_version_after_miss() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = None

    with (
        patch.object(service, "_register"),
        patch.object(service, "embed_texts", return_value=[[0.1, 0.2]]) as embed,
    ):
        result = service._rag_query_embedding(
            "Python backend engineer",
            conn=conn,
            user_id=7,
            profile_version="profile-v1",
        )

    assert result == [0.1, 0.2]
    embed.assert_called_once()
    assert cur.execute.call_count == 2
    conn.commit.assert_called_once()
