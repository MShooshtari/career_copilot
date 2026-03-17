"""Unit tests for add_job router: GET/POST add-job."""

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


def test_get_add_job_returns_html(client: TestClient) -> None:
    with patch("career_copilot.routers.add_job.get_db") as mock_get_db:
        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn
        response = client.get("/add-job")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert b"add" in response.content.lower() or b"job" in response.content.lower()


def test_post_add_job_no_data_returns_error(client: TestClient, mock_db: MagicMock) -> None:
    with patch("career_copilot.routers.add_job.get_db", return_value=mock_db):
        response = client.post(
            "/add-job",
            data={"mode": "manual", "job_text": ""},
            follow_redirects=False,
        )
    assert response.status_code == 200
    assert b"error" in response.content.lower() or b"provide" in response.content.lower()


def test_post_add_job_agent_raises_returns_error(client: TestClient) -> None:
    with patch("career_copilot.routers.add_job.run_add_job_agent") as mock_agent:
        mock_agent.side_effect = Exception("fetch failed")
        response = client.post(
            "/add-job",
            data={"mode": "url", "job_url": "https://invalid.example.com/nonexistent"},
            follow_redirects=False,
        )
    assert response.status_code == 200
    assert b"error" in response.content.lower() or b"extract" in response.content.lower()


def test_post_add_job_manual_mode_returns_confirm_page(client: TestClient) -> None:
    with patch("career_copilot.routers.add_job.run_add_job_agent") as mock_agent:
        mock_agent.return_value = {
            "title": "Test Engineer",
            "company": "TestCo",
            "location": "Remote",
            "salary_min": None,
            "salary_max": None,
            "description": "Test job.",
            "skills": [],
            "url": None,
        }
        response = client.post(
            "/add-job",
            data={"mode": "manual", "job_text": "We need a Test Engineer at TestCo. Remote."},
            follow_redirects=False,
        )
    assert response.status_code == 200
    assert b"Confirm" in response.content or b"confirm" in response.content
    assert b"Test Engineer" in response.content
    assert b"TestCo" in response.content


def test_post_add_job_url_mode_returns_confirm_page(client: TestClient) -> None:
    with patch("career_copilot.routers.add_job.run_add_job_agent") as mock_agent:
        mock_agent.return_value = {
            "title": "Data Engineer",
            "company": "DataCo",
            "location": "NYC",
            "salary_min": 120000,
            "salary_max": 180000,
            "description": "Build pipelines.",
            "skills": ["Python", "SQL"],
            "url": "https://example.com/job/1",
        }
        response = client.post(
            "/add-job",
            data={"mode": "url", "job_url": "https://example.com/job/1"},
            follow_redirects=False,
        )
    assert response.status_code == 200
    assert b"Data Engineer" in response.content
    assert b"add-job/confirm" in response.content


def test_post_add_job_agent_returns_empty_shows_error(client: TestClient) -> None:
    with patch("career_copilot.routers.add_job.run_add_job_agent") as mock_agent:
        mock_agent.return_value = None
        response = client.post(
            "/add-job",
            data={"mode": "manual", "job_text": "Some text"},
            follow_redirects=False,
        )
    assert response.status_code == 200
    assert b"couldn't" in response.content.lower() or b"error" in response.content.lower()


def test_post_add_job_confirm_saves_and_redirects(client: TestClient, mock_db: MagicMock) -> None:
    with patch("career_copilot.routers.add_job.get_db", return_value=mock_db):
        with patch(
            "career_copilot.routers.add_job.insert_user_job", return_value=42
        ) as mock_insert:
            response = client.post(
                "/add-job/confirm",
                data={
                    "title": "Software Engineer",
                    "company": "Acme",
                    "location": "Remote",
                    "salary_min": "100000",
                    "salary_max": "150000",
                    "description": "Build things.",
                    "skills": "Python, AWS",
                    "url": "https://example.com/job",
                },
                follow_redirects=False,
            )
    assert response.status_code == 303
    assert "/recommendations" in response.headers.get("location", "")
    assert "added=42" in response.headers.get("location", "")
    mock_insert.assert_called_once()
    call_kw = mock_insert.call_args[1]
    assert call_kw["title"] == "Software Engineer"
    assert call_kw["company"] == "Acme"
    assert call_kw["skills"] == ["Python", "AWS"]
