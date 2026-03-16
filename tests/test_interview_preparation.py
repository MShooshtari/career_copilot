from __future__ import annotations

from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from career_copilot.agents import interview_preparation as agent_mod
from career_copilot.web_app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_get_initial_interview_message_contains_prompt() -> None:
    message = agent_mod.get_initial_interview_message()
    assert isinstance(message, str)
    assert "What type of interview" in message


def test_build_interview_prep_context_uses_resume_improvement_context() -> None:
    with patch(
        "career_copilot.agents.interview_preparation.build_resume_improvement_context",
        return_value={"resume_text": "resume", "job": {"id": 1}},
    ) as mock_build:
        ctx = agent_mod.build_interview_prep_context(job_id=1, user_id=2, conn=MagicMock())
    mock_build.assert_called_once_with(1, 2, ANY)
    assert ctx == {"resume_text": "resume", "job": {"id": 1}}


@pytest.mark.parametrize(
    "user_message,expected",
    [
        ("", "Please tell me what type of interview"),
        ("   ", "Please tell me what type of interview"),
    ],
)
def test_chat_interview_preparation_empty_message_prompts_user(
    user_message: str,
    expected: str,
) -> None:
    reply = agent_mod.chat_interview_preparation(
        user_message=user_message,
        conversation_history=[],
        resume_text="resume",
        job={"company": "Acme", "title": "Engineer"},
    )
    assert expected in reply


def test_chat_interview_preparation_without_job_returns_error() -> None:
    reply = agent_mod.chat_interview_preparation(
        user_message="Technical interview",
        conversation_history=[],
        resume_text="resume",
        job={},
    )
    assert "don't have the job context" in reply


def test_chat_interview_preparation_first_reply_uses_tool_and_openai() -> None:
    class _FakeToolCall:
        def __init__(self) -> None:
            self.id = "tool-1"

            class _Func:
                def __init__(self) -> None:
                    self.name = "search_company_interview_context"

            self.function = _Func()

        def model_dump(self) -> dict[str, Any]:  # type: ignore[override]
            return {"id": self.id, "function": {"name": self.function.name}}

    class _FakeMessage:
        def __init__(self, content: str, tool_calls: list[_FakeToolCall] | None) -> None:
            self.content = content
            if tool_calls is not None:
                self.tool_calls = tool_calls

    class _FakeChoice:
        def __init__(self, msg: _FakeMessage) -> None:
            self.message = msg

    class _FakeResponse:
        def __init__(self, msg: _FakeMessage) -> None:
            self.choices = [_FakeChoice(msg)]

    class _FakeChat:
        def __init__(self) -> None:
            self.call_index = 0

        def create(self, **_: Any) -> _FakeResponse:
            self.call_index += 1
            if self.call_index == 1:
                return _FakeResponse(_FakeMessage("", [_FakeToolCall()]))
            return _FakeResponse(_FakeMessage("prep plan", None))

    class _FakeClient:
        def __init__(self) -> None:
            self.chat = MagicMock()
            self.chat.completions = _FakeChat()

    job = {"company": "Acme", "title": "Engineer"}
    history = [{"role": "assistant", "content": "initial prompt"}]

    with (
        patch.object(agent_mod, "_get_openai_client", return_value=_FakeClient()) as mock_client,
        patch.object(
            agent_mod,
            "search_web_for_company",
            return_value={"context": "ctx", "found_glassdoor": True, "found_reddit": False},
        ) as mock_search,
    ):
        reply = agent_mod.chat_interview_preparation(
            user_message="Technical",
            conversation_history=history,
            resume_text="resume text",
            job=job,
        )

    assert reply == "prep plan"
    mock_client.assert_called_once_with()
    mock_search.assert_called_once()
    called_kwargs = mock_search.call_args.kwargs
    assert called_kwargs["company_name"] == "Acme"
    assert called_kwargs["job_title"] == "Engineer"


def test_prepare_interview_initial_returns_prompt(client: TestClient) -> None:
    mock_conn = MagicMock()
    with (
        patch("career_copilot.routers.interview_preparation.get_db", return_value=mock_conn),
        patch(
            "career_copilot.routers.interview_preparation.build_interview_prep_context",
            return_value={"resume_text": "resume", "job": {"id": 1}},
        ),
        patch(
            "career_copilot.routers.interview_preparation.get_initial_interview_message",
            return_value="initial prompt",
        ),
    ):
        response = client.post(
            "/jobs/1/prepare-interview/chat",
            json={"message": "initial", "history": []},
        )
    data = response.json()
    assert response.status_code == 200
    assert data["reply"] == "initial prompt"


def test_prepare_interview_chat_calls_agent(client: TestClient) -> None:
    mock_conn = MagicMock()
    with (
        patch("career_copilot.routers.interview_preparation.get_db", return_value=mock_conn),
        patch(
            "career_copilot.routers.interview_preparation.build_interview_prep_context",
            return_value={"resume_text": "resume", "job": {"id": 1}},
        ),
        patch(
            "career_copilot.routers.interview_preparation.chat_interview_preparation",
            return_value="reply from agent",
        ) as mock_chat,
    ):
        body = {
            "message": "Technical",
            "history": [{"role": "assistant", "content": "initial"}],
        }
        response = client.post("/jobs/1/prepare-interview/chat", json=body)

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "reply from agent"
    mock_chat.assert_called_once()
