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


def test_build_market_analysis_report_keeps_remote_mode_in_cache_key() -> None:
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
            filters=MarketCohortFilters(remote_mode="remote_only"),
            cohort_limit=500,
            include_rag=False,
        )
        service.build_market_analysis_report(
            conn,
            user_id=7,
            filters=MarketCohortFilters(remote_mode="no_remote"),
            cohort_limit=500,
            include_rag=False,
        )

    assert cohort.call_count == 2


def test_aggregates_top_skills_prefers_ai_then_legacy_skill_columns() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.side_effect = [
        [],
        [(1, "python"), (2, "python")],
        [("python", 2)],
        [],
    ]
    cur.fetchone.side_effect = [
        (10,),
        (0, 0, None, None, None),
    ]

    with patch.object(service, "_register"):
        aggregates = service._aggregates_for_cohort(conn, [1, 2])

    top_skills_sql = cur.execute.call_args_list[1].args[0]
    assert "j.ai_extracted_skills" in top_skills_sql
    assert "j.extracted_skills" in top_skills_sql
    assert "j.skills" in top_skills_sql
    assert top_skills_sql.index("j.ai_extracted_skills") < top_skills_sql.index(
        "j.extracted_skills"
    )
    assert top_skills_sql.index("j.extracted_skills") < top_skills_sql.index("j.skills")
    assert aggregates["top_skills"][0]["skill"] == "python"
    assert aggregates["top_skills"][0]["count"] == 2
    assert aggregates["top_skills"][0]["market_score"] > 0


def test_filtered_top_skills_removes_generic_terms_and_merges_variants() -> None:
    raw = [
        *[(idx, "engineering") for idx in range(10)],
        *[(idx, "engineer") for idx in range(9)],
        *[(idx, "software") for idx in range(8)],
        *[(idx, "digital nomad") for idx in range(7)],
        *[(idx, "code") for idx in range(6)],
        *[(idx, "python") for idx in range(5)],
        *[(idx + 5, "Python") for idx in range(3)],
        *[(idx, "langchain") for idx in range(4)],
    ]

    filtered = service._filtered_top_skills(raw)

    assert [(item["skill"], item["count"]) for item in filtered] == [
        ("python", 8),
        ("langchain", 4),
    ]


def test_filtered_top_skills_demotes_vague_terms_below_specific_skills() -> None:
    raw = [
        *[(idx, "problem solving") for idx in range(100)],
        *[(idx, "attention to detail") for idx in range(90)],
        *[(idx, "python") for idx in range(5)],
        *[(idx, "vector databases") for idx in range(4)],
    ]

    filtered = service._filtered_top_skills(raw)

    assert set(item["skill"] for item in filtered[:2]) == {"python", "vector databases"}
    assert {item["skill"] for item in filtered} >= {"problem solving", "attention to detail"}


def test_filtered_top_skills_uses_profile_weight_and_idf_over_raw_frequency() -> None:
    raw = [
        *[(idx, "customer service") for idx in range(100, 220)],
        *[(idx, "python") for idx in range(8)],
        *[(idx, "machine learning") for idx in range(6)],
    ]

    filtered = service._filtered_top_skills(
        raw,
        job_ids=list(range(220)),
        user_skills={"python", "machine learning"},
    )

    top_names = [item["skill"] for item in filtered[:3]]
    assert set(top_names[:2]) == {"python", "machine learning"}
    assert top_names.index("customer service") > top_names.index("machine learning")


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
