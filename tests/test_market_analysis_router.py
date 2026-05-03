"""Tests for market analysis routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from career_copilot.database import deps as db_deps
from career_copilot.web_app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_market_analysis_page_returns_html(client: TestClient) -> None:
    mock_conn = MagicMock()
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        response = client.get("/market-analysis")
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    mock_conn.close.assert_called_once()


def test_api_market_analysis_returns_json(client: TestClient) -> None:
    mock_conn = MagicMock()
    sample = {
        "cohort": {"size": 0, "job_ids_sample": [], "filtered_count": 0},
        "filters": {},
        "weekly_posted": [],
        "top_skills": [],
        "salary": {},
        "top_locations": [],
        "fit": {},
        "rag": {"available": False, "narrative": "", "error": "x"},
    }
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        with patch(
            "career_copilot.routers.market_analysis.build_market_analysis_report",
            return_value=sample,
        ) as build_report:
            response = client.get("/api/market-analysis")
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)
    assert response.status_code == 200
    assert response.json()["cohort"]["size"] == 0
    assert build_report.call_args.kwargs["include_rag"] is False
    mock_conn.close.assert_called_once()


def test_api_market_analysis_allows_rag_opt_in(client: TestClient) -> None:
    mock_conn = MagicMock()
    sample = {
        "cohort": {"size": 1, "job_ids_sample": [123], "filtered_count": 1},
        "filters": {},
        "weekly_posted": [],
        "top_skills": [],
        "salary": {},
        "top_locations": [],
        "fit": {},
        "rag": {"available": True, "narrative": "x", "error": None},
    }
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        with patch(
            "career_copilot.routers.market_analysis.build_market_analysis_report",
            return_value=sample,
        ) as build_report:
            response = client.get("/api/market-analysis?include_rag=true")
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)
    assert response.status_code == 200
    assert response.json()["rag"]["available"] is True
    assert build_report.call_args.kwargs["include_rag"] is True
    mock_conn.close.assert_called_once()
