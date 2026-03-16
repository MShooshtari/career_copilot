"""Interview preparation chat API."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from career_copilot.agents.interview_preparation import (
    build_interview_prep_context,
    chat_interview_preparation,
    get_initial_interview_message,
)
from career_copilot.database.deps import get_db
from career_copilot.schemas import InterviewChatRequest

router = APIRouter(tags=["interview_preparation"])

USER_ID = 1


@router.post("/jobs/{job_id:int}/prepare-interview/chat")
async def post_prepare_interview_chat(
    job_id: int,
    body: InterviewChatRequest,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> JSONResponse:
    """
    Chat endpoint for interview preparation. Send message and optional history.
    If history is empty and message is 'initial', returns initial prompt asking interview type.
    Once the user replies with interview type, the backend searches the web (Glassdoor, Reddit,
    etc.) and returns a tailored preparation plan using job description and user resume.
    """
    ctx = build_interview_prep_context(job_id, USER_ID, conn)
    conn.close()
    resume_text = ctx["resume_text"]
    job = ctx["job"]

    if not job:
        return JSONResponse(
            status_code=404,
            content={"reply": "Job not found. Please go back to recommendations."},
        )

    is_initial = not (body.history or []) and (
        not (body.message or "").strip() or (body.message or "").strip().lower() == "initial"
    )
    if is_initial:
        try:
            reply = get_initial_interview_message()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"reply": f"Sorry, something went wrong. ({e!s})"},
            )
    else:
        try:
            reply = chat_interview_preparation(
                body.message or "",
                body.history or [],
                resume_text,
                job,
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"reply": f"Sorry, something went wrong. ({e!s})"},
            )
    return JSONResponse(content={"reply": reply})
