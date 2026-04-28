"""Unit tests for recommendations routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from career_copilot.database import deps as db_deps
from career_copilot.web_app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_post_job_feedback_records_choice_and_redirects(client: TestClient) -> None:
    mock_conn = MagicMock()
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        with patch("career_copilot.routers.recommendations.set_job_feedback") as mock_set:
            response = client.post(
                "/recommendations/ingested/123/feedback/dislike?page=2&page_size=10",
                follow_redirects=False,
            )
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)

    assert response.status_code == 303
    assert response.headers.get("location") == "/recommendations?page=2&page_size=10"
    mock_set.assert_called_once_with(mock_conn, 1, 123, "ingested", "dislike")
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


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
