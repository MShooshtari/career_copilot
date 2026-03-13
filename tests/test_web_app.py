"""Tests for FastAPI web app and routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from career_copilot.web_app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_home_redirects_to_profile(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/profile"


def test_openapi_json_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data


def test_docs_available(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200


def test_profile_get_returns_html(client: TestClient) -> None:
    # Uses real get_db by default; may fail if DB not available.
    # Override get_db to avoid DB for unit test.
    with patch("career_copilot.routers.profile.get_db") as mock_get_db:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = lambda self: mock_cur
        mock_conn.cursor.return_value.__exit__ = lambda *a: None
        mock_get_db.return_value = mock_conn

        response = client.get("/profile")
        # 200 with profile page even when no profile row
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


def test_jobs_detail_not_found_redirects(client: TestClient) -> None:
    with patch("career_copilot.routers.jobs.get_db") as mock_get_db:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = lambda self: mock_cur
        mock_conn.cursor.return_value.__exit__ = lambda *a: None
        mock_get_db.return_value = mock_conn

        response = client.get("/jobs/99999", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/recommendations"


def test_recommendations_get_returns_html(client: TestClient) -> None:
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__ = lambda self: mock_cur
    mock_conn.cursor.return_value.__exit__ = lambda *a: None

    with (
        patch("career_copilot.routers.recommendations.get_db", return_value=mock_conn),
        patch(
            "career_copilot.routers.recommendations.get_recommended_job_results", return_value=[]
        ),
    ):
        response = client.get("/recommendations")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
