"""Unit tests for recommendations routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from career_copilot.database import deps as db_deps
from career_copilot.routers import recommendations as recommendations_router
from career_copilot.routers.recommendations import _attach_feedback, _drop_hidden_interactions
from career_copilot.web_app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def setup_function() -> None:
    recommendations_router.clear_recommendation_caches()


def teardown_function() -> None:
    recommendations_router.clear_recommendation_caches()


def test_recommendations_page_caches_online_candidates(client: TestClient) -> None:
    mock_conn = MagicMock()
    jobs_online = [
        {
            "job_id": 123,
            "is_user_job": False,
            "title": "AI Engineer",
            "company": "Acme",
            "location": "Remote",
            "url": "",
            "snippet": "Build AI systems.",
            "salary_min": None,
            "salary_max": None,
            "skills": ["RawTag"],
        }
    ]
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        with (
            patch("career_copilot.routers.recommendations.list_user_jobs", return_value=[]),
            patch(
                "career_copilot.routers.recommendations.get_job_interactions_map",
                return_value={},
            ) as interactions,
            patch(
                "career_copilot.routers.recommendations.get_recommended_job_results",
                return_value=[{"metadata": {"title": "AI Engineer"}}],
            ) as get_results,
            patch(
                "career_copilot.routers.recommendations.score_candidates_by_distance",
                side_effect=lambda raw: raw,
            ),
            patch(
                "career_copilot.routers.recommendations.rerank_with_diversity_and_exploration",
                side_effect=lambda raw, **_kwargs: raw,
            ),
            patch("career_copilot.routers.recommendations.resolve_job_ids", return_value={}),
            patch(
                "career_copilot.routers.recommendations.format_recommendation_jobs",
                return_value=jobs_online,
            ) as format_jobs,
        ):
            first = client.get("/recommendations")
            second = client.get("/recommendations?page=1&page_size=10")
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)

    assert first.status_code == 200
    assert second.status_code == 200
    get_results.assert_called_once()
    format_jobs.assert_called_once()
    assert interactions.call_count == 4
    assert mock_conn.close.call_count == 2


def test_post_job_feedback_records_choice_and_redirects(client: TestClient) -> None:
    mock_conn = MagicMock()
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        with (
            patch("career_copilot.routers.recommendations.set_job_feedback") as mock_set,
            patch(
                "career_copilot.routers.recommendations.clear_recommendation_caches"
            ) as clear_caches,
        ):
            response = client.post(
                "/recommendations/ingested/123/feedback/disliked?page=2&page_size=10",
                follow_redirects=False,
            )
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)

    assert response.status_code == 303
    assert response.headers.get("location") == "/recommendations?page=2&page_size=10"
    mock_set.assert_called_once_with(mock_conn, 1, 123, "ingested", "disliked")
    mock_conn.commit.assert_called_once()
    clear_caches.assert_called_once()
    mock_conn.close.assert_called_once()


def test_post_job_feedback_records_already_applied(client: TestClient) -> None:
    mock_conn = MagicMock()
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        with patch("career_copilot.routers.recommendations.set_job_feedback") as mock_set:
            response = client.post(
                "/recommendations/ingested/123/feedback/applied?page=2&page_size=10",
                follow_redirects=False,
            )
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)

    assert response.status_code == 303
    mock_set.assert_called_once_with(mock_conn, 1, 123, "ingested", "applied")
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


def test_post_job_feedback_ajax_returns_json_without_redirect(client: TestClient) -> None:
    mock_conn = MagicMock()
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        with patch("career_copilot.routers.recommendations.set_job_feedback") as mock_set:
            response = client.post(
                "/recommendations/ingested/123/feedback/applied?page=2&page_size=10",
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)

    assert response.status_code == 200
    assert response.json() == {"ok": True, "feedback": "applied"}
    mock_set.assert_called_once_with(mock_conn, 1, 123, "ingested", "applied")
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


def test_apply_on_source_records_applied_and_redirects_to_job_url(client: TestClient) -> None:
    mock_conn = MagicMock()
    row = (
        123,
        "remoteok",
        "remoteok-123",
        "AI Engineer",
        "Acme",
        "Remote",
        None,
        None,
        "Description",
        ["SourceTag"],
        ["Python"],
        None,
        "https://example.com/apply",
    )
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        with (
            patch("career_copilot.routers.recommendations.get_job_by_id", return_value=row),
            patch("career_copilot.routers.recommendations.set_job_feedback") as mock_set,
            patch(
                "career_copilot.routers.recommendations.clear_recommendation_caches"
            ) as clear_caches,
        ):
            response = client.get(
                "/recommendations/ingested/123/apply-on-source",
                follow_redirects=False,
            )
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)

    assert response.status_code == 303
    assert response.headers.get("location") == "https://example.com/apply"
    mock_set.assert_called_once_with(mock_conn, 1, 123, "ingested", "applied")
    mock_conn.commit.assert_called_once()
    clear_caches.assert_called_once()
    mock_conn.close.assert_called_once()


def test_drop_hidden_interactions_keeps_likes_and_unseen_jobs() -> None:
    jobs = [
        {"job_id": 1, "feedback": "liked"},
        {"job_id": 2, "feedback": "disliked"},
        {"job_id": 3, "feedback": "liked", "applied": True},
        {"job_id": 4},
        {"job_id": 5, "deleted": True},
    ]

    assert _drop_hidden_interactions(jobs) == [
        {"job_id": 1, "feedback": "liked"},
        {"job_id": 4},
    ]


def test_attach_feedback_keeps_applied_independent_from_liked() -> None:
    jobs = [{"job_id": 1}, {"job_id": 2}, {"job_id": 3}]

    _attach_feedback(jobs, {1: {"liked", "applied"}, 2: {"disliked"}})

    assert jobs == [
        {"job_id": 1, "feedback": "liked", "applied": True, "deleted": False},
        {"job_id": 2, "feedback": "disliked", "applied": False, "deleted": False},
        {"job_id": 3, "feedback": None, "applied": False, "deleted": False},
    ]


def test_post_job_feedback_rejects_invalid_feedback(client: TestClient) -> None:
    mock_conn = MagicMock()
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        response = client.post(
            "/recommendations/ingested/123/feedback/skip",
            follow_redirects=False,
        )
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)

    assert response.status_code == 400
    mock_conn.commit.assert_not_called()
    mock_conn.close.assert_called_once()
