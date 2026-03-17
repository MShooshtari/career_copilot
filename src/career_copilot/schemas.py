"""Pydantic request/response models for the web API."""

from __future__ import annotations

from pydantic import BaseModel


class ResumeChatRequest(BaseModel):
    """Request body for resume improvement chat."""

    message: str = ""
    history: list[dict[str, str]] = []


class ResumePdfRequest(BaseModel):
    """Request body for generating a PDF of the updated resume."""

    history: list[dict[str, str]] | None = None


class InterviewChatRequest(BaseModel):
    """Request body for interview preparation chat."""

    message: str = ""
    history: list[dict[str, str]] = []


class TrackApplicationsChatRequest(BaseModel):
    """Request body for track applications agent chat."""

    message: str = ""
    history: list[dict[str, str]] = []
