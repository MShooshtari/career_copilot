from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from career_copilot.agents import resume_improvement as agent_mod


def _build_fake_tool_call(name: str) -> Any:
    class _Func:
        def __init__(self, tool_name: str) -> None:
            self.name = tool_name

    class _ToolCall:
        def __init__(self, tool_name: str) -> None:
            self.id = "tool-call-1"
            self.function = _Func(tool_name)

        def model_dump(self) -> dict[str, Any]:  # type: ignore[override]
            return {"id": self.id, "function": {"name": self.function.name}}

    return _ToolCall(name)


def _build_fake_openai_response(content: str, tool_calls: list[Any] | None = None) -> Any:
    class _FakeMessage:
        def __init__(self, body: str, calls: list[Any] | None) -> None:
            self.content = body
            if calls is not None:
                self.tool_calls = calls

    class _FakeChoice:
        def __init__(self, msg: _FakeMessage) -> None:
            self.message = msg

    class _FakeResponse:
        def __init__(self, msg: _FakeMessage) -> None:
            self.choices = [_FakeChoice(msg)]

    return _FakeResponse(_FakeMessage(content, tool_calls))


def test_get_initial_resume_analysis_returns_text_without_tools() -> None:
    job = {"company": "Acme", "title": "Engineer", "description": "Build things."}
    similar_jobs: list[dict[str, Any]] = []
    similar_resumes: list[dict[str, Any]] = []

    class _FakeChat:
        def __init__(self) -> None:
            self.last_args: dict[str, Any] | None = None

        def create(self, **kwargs: Any) -> Any:
            self.last_args = kwargs
            return _build_fake_openai_response("analysis reply", None)

    class _FakeClient:
        def __init__(self) -> None:
            self.chat = MagicMock()
            self.chat.completions = _FakeChat()

    with patch.object(agent_mod, "_get_openai_client", return_value=_FakeClient()):
        reply = agent_mod.get_initial_resume_analysis(
            resume_text="resume text",
            job=job,
            similar_jobs=similar_jobs,
            similar_resumes=similar_resumes,
        )

    assert reply == "analysis reply"


def test_chat_resume_improvement_handles_tool_call() -> None:
    job = {"company": "Acme", "title": "Engineer", "description": "Build things."}

    calls: list[str] = []

    class _FakeChat:
        def __init__(self) -> None:
            self.call_index = 0

        def create(self, **_: Any) -> Any:
            self.call_index += 1
            if self.call_index == 1:
                # First response asks to call get_more_similar_jobs
                return _build_fake_openai_response(
                    "",
                    [_build_fake_tool_call("get_more_similar_jobs")],
                )
            # Second response returns final text
            return _build_fake_openai_response("final reply", None)

    class _FakeClient:
        def __init__(self) -> None:
            self.chat = MagicMock()
            self.chat.completions = _FakeChat()

    def _fake_get_similar_jobs(
        conn: object, job_document: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        calls.append(job_document)
        return [{"id": "job1", "document": "doc", "metadata": {}, "distance": 0.1}]

    with (
        patch.object(agent_mod, "_get_openai_client", return_value=_FakeClient()),
        patch.object(
            agent_mod,
            "get_similar_jobs_for_resume_improvement",
            side_effect=_fake_get_similar_jobs,
        ),
    ):
        reply = agent_mod.chat_resume_improvement(
            user_message="shorten bullet 2",
            conversation_history=[],
            resume_text="resume",
            job=job,
            similar_jobs=[],
            similar_resumes=[],
        )

    assert reply == "final reply"
    assert calls, "expected RAG tool to be called at least once"
