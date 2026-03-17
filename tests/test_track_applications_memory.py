from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from career_copilot.database import deps as db_deps
from career_copilot.web_app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_applications_context_endpoint_returns_memory_and_history(client: TestClient) -> None:
    now = datetime.now(UTC)
    # Row shape:
    # (id, user_id, job_id, job_source, stage, status, history, application_memory, last_resume_text, created_at, updated_at)
    row = (
        10,
        1,
        123,
        "ingested",
        "resume_improvement",
        "active",
        [{"role": "assistant", "content": "hi"}],
        {"summary": "- Address: Los Angeles"},
        "UPDATED RESUME TEXT",
        now,
        now,
    )

    mock_conn = MagicMock()
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        with patch(
            "career_copilot.routers.track_applications.get_application_by_key",
            return_value=row,
        ):
            r = client.get(
                "/applications/context?job_id=123&job_source=ingested&stage=resume_improvement"
            )
        assert r.status_code == 200
        data = r.json()
        assert data["found"] is True
        assert data["history"] == [{"role": "assistant", "content": "hi"}]
        assert data["application_memory"]["summary"].startswith("- Address")
        assert data["last_resume_text"] == "UPDATED RESUME TEXT"
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)


def test_resume_improvement_chat_truncates_stored_history_to_last_20(client: TestClient) -> None:
    # Build a stored history longer than MAX_STORED_MESSAGES (20)
    stored_history = []
    for i in range(30):
        stored_history.append({"role": "user", "content": f"u{i}"})
        stored_history.append({"role": "assistant", "content": f"a{i}"})

    now = datetime.now(UTC)
    app_row = (
        1,
        1,
        7,
        "ingested",
        "resume_improvement",
        "active",
        stored_history,
        {"summary": "old"},
        "LAST RESUME",
        now,
        now,
    )

    conn1 = MagicMock()
    conn2 = MagicMock()

    # Router uses dependency injection for conn1, but also calls get_db() internally for conn2.
    # We patch the router.get_db to return conn2 for the persistence phase.
    app.dependency_overrides[db_deps.get_db] = lambda: conn1
    try:
        with (
            patch(
                "career_copilot.routers.resume_improvement.get_db",
                return_value=conn2,
            ),
            patch(
                "career_copilot.routers.resume_improvement.build_resume_improvement_context",
                return_value={
                    "resume_text": "ORIGINAL RESUME",
                    "job": {"id": 7, "company": "Acme", "title": "Eng", "description": "desc"},
                    "similar_jobs": [],
                    "similar_resumes": [],
                },
            ),
            patch(
                "career_copilot.routers.resume_improvement.get_application_by_key",
                side_effect=[app_row, app_row, app_row],
            ),
            patch(
                "career_copilot.routers.resume_improvement.chat_resume_improvement",
                return_value="reply",
            ),
            patch(
                "career_copilot.routers.resume_improvement.generate_full_resume",
                return_value="UPDATED RESUME",
            ),
            patch(
                "career_copilot.routers.resume_improvement.set_application_history"
            ) as mock_set_hist,
            patch("career_copilot.routers.resume_improvement.set_application_last_resume_text"),
            patch("career_copilot.routers.resume_improvement.set_application_memory"),
        ):
            r = client.post(
                "/jobs/7/improve-resume/chat",
                json={"message": "change address to LA", "history": []},
            )
        assert r.status_code == 200

        # We may call set_application_history twice (once full, then truncated). Validate last call is <= 20.
        assert mock_set_hist.call_count >= 1
        last_history = mock_set_hist.call_args_list[-1].args[3]
        assert isinstance(last_history, list)
        assert len(last_history) <= 20
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)


def test_delete_application_endpoint_removes_and_returns_updated_list(client: TestClient) -> None:
    mock_conn = MagicMock()
    app.dependency_overrides[db_deps.get_db] = lambda: mock_conn
    try:
        with (
            patch(
                "career_copilot.routers.track_applications.remove_application", return_value=True
            ),
            patch("career_copilot.routers.track_applications.list_applications", return_value=[]),
            patch(
                "career_copilot.routers.track_applications.enrich_applications_with_job_info",
                return_value=[],
            ),
        ):
            r = client.post("/applications/123/delete")
        assert r.status_code == 200
        data = r.json()
        assert data["removed"] is True
        assert data["applications"] == []
    finally:
        app.dependency_overrides.pop(db_deps.get_db, None)
