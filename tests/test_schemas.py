"""Tests for career_copilot.schemas (Pydantic API models)."""

from __future__ import annotations

from career_copilot.schemas import ResumeChatRequest, ResumePdfRequest


def test_resume_chat_request_defaults() -> None:
    r = ResumeChatRequest()
    assert r.message == ""
    assert r.history == []


def test_resume_chat_request_with_values() -> None:
    r = ResumeChatRequest(message="Improve my summary", history=[{"role": "user", "content": "Hi"}])
    assert r.message == "Improve my summary"
    assert len(r.history) == 1
    assert r.history[0]["role"] == "user"


def test_resume_pdf_request_defaults() -> None:
    r = ResumePdfRequest()
    assert r.history is None


def test_resume_pdf_request_with_history() -> None:
    r = ResumePdfRequest(history=[{"role": "assistant", "content": "Done"}])
    assert r.history is not None
    assert len(r.history) == 1
