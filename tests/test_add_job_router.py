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


def test_post_add_job_url_mode_invalid_url_returns_error(client: TestClient, mock_db: MagicMock) -> None:
    with patch("career_copilot.routers.add_job.get_db", return_value=mock_db):
        with patch("career_copilot.routers.add_job.extract_job_from_url") as mock_extract:
            mock_extract.side_effect = Exception("fetch failed")
            response = client.post(
                "/add-job",
                data={"mode": "url", "job_url": "https://invalid.example.com/nonexistent"},
                follow_redirects=False,
            )
    assert response.status_code == 200
    assert b"Could not fetch" in response.content or b"error" in response.content.lower()


def test_post_add_job_manual_mode_success_redirects(client: TestClient, mock_db: MagicMock) -> None:
    with patch("career_copilot.routers.add_job.get_db", return_value=mock_db):
        with patch("career_copilot.routers.add_job.extract_job_from_text") as mock_extract:
            mock_extract.return_value = {
                "title": "Test Engineer",
                "company": "TestCo",
                "location": "Remote",
                "salary_min": None,
                "salary_max": None,
                "description": "Test job.",
                "skills": [],
                "url": None,
            }
            with patch("career_copilot.routers.add_job.insert_user_job", return_value=42) as mock_insert:
                response = client.post(
                    "/add-job",
                    data={"mode": "manual", "job_text": "We need a Test Engineer at TestCo. Remote."},
                    follow_redirects=False,
                )
    assert response.status_code == 303
    assert "/recommendations" in response.headers.get("location", "")
    assert "added=42" in response.headers.get("location", "")
    mock_insert.assert_called_once()
    call_kw = mock_insert.call_args[1]
    assert call_kw["title"] == "Test Engineer"
    assert call_kw["company"] == "TestCo"


def test_post_add_job_url_mode_success_redirects(client: TestClient, mock_db: MagicMock) -> None:
    with patch("career_copilot.routers.add_job.get_db", return_value=mock_db):
        with patch("career_copilot.routers.add_job.extract_job_from_url") as mock_extract:
            mock_extract.return_value = {
                "title": "Data Engineer",
                "company": "DataCo",
                "location": "NYC",
                "salary_min": 120000,
                "salary_max": 180000,
                "description": "Build pipelines.",
                "skills": ["Python", "SQL"],
                "url": "https://example.com/job/1",
            }
            with patch("career_copilot.routers.add_job.insert_user_job", return_value=99) as mock_insert:
                response = client.post(
                    "/add-job",
                    data={"mode": "url", "job_url": "https://example.com/job/1"},
                    follow_redirects=False,
                )
    assert response.status_code == 303
    assert "added=99" in response.headers.get("location", "")
    mock_insert.assert_called_once()
    call_kw = mock_insert.call_args[1]
    assert call_kw["title"] == "Data Engineer"
    assert call_kw["url"] == "https://example.com/job/1"
    assert call_kw["skills"] == ["Python", "SQL"]


def test_post_add_job_generic_title_replaced_by_url(client: TestClient, mock_db: MagicMock) -> None:
    with patch("career_copilot.routers.add_job.get_db", return_value=mock_db):
        with patch("career_copilot.routers.add_job.extract_job_from_url") as mock_extract:
            mock_extract.return_value = {
                "title": "Job from Indeed",
                "company": None,
                "location": None,
                "description": "Some description.",
                "skills": [],
                "url": "https://ca.indeed.com/viewjob?jk=1&q=machine+learning",
            }
            with patch("career_copilot.routers.add_job.insert_user_job", return_value=1) as mock_insert:
                response = client.post(
                    "/add-job",
                    data={"mode": "url", "job_url": "https://ca.indeed.com/viewjob?jk=1&q=machine+learning"},
                    follow_redirects=False,
                )
    assert response.status_code == 303
    call_kw = mock_insert.call_args[1]
    assert call_kw["title"] == "Machine Learning"
