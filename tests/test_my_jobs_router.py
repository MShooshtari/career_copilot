"""Unit tests for my_jobs router: delete job, get detail, etc."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from career_copilot.web_app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_db():
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = lambda self: cur
    conn.cursor.return_value.__exit__ = lambda *a: None
    cur.fetchone.return_value = None
    return conn


def test_post_my_job_delete_redirects_to_recommendations(client: TestClient, mock_db: MagicMock) -> None:
    with patch("career_copilot.routers.my_jobs.get_db", return_value=mock_db):
        with patch("career_copilot.routers.my_jobs.delete_user_job") as mock_delete:
            response = client.post("/my-jobs/123/delete", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers.get("location") == "/recommendations"
    mock_delete.assert_called_once()
    assert mock_delete.call_args[0][1] == 1
    assert mock_delete.call_args[0][2] == 123


def test_get_my_job_detail_not_found_redirects(client: TestClient, mock_db: MagicMock) -> None:
    # mock_db fixture: cursor().fetchone() returns None, so job is not found
    with patch("career_copilot.routers.my_jobs.get_db", return_value=mock_db):
        response = client.get("/my-jobs/99999", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers.get("location") == "/recommendations"


def test_get_my_job_detail_found_returns_html(client: TestClient) -> None:
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda self: mock_cur
    mock_conn.cursor.return_value.__exit__ = lambda *a: None
    # user_jobs row: id, user_id, title, company, location, salary_min, salary_max, description, skills, url
    row = (
        1,
        1,
        "Software Engineer",
        "Acme",
        "Remote",
        None,
        None,
        "Description here.",
        ["Python"],
        "https://example.com/job",
    )
    mock_cur.fetchone.return_value = row

    with patch("career_copilot.routers.my_jobs.get_db", return_value=mock_conn):
        response = client.get("/my-jobs/1")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert b"Software Engineer" in response.content or b"Acme" in response.content
