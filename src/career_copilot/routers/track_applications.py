"""Track applications: resume improvement and interview preparation for jobs."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

from career_copilot.agents.track_applications import chat_track_applications
from career_copilot.app_config import templates
from career_copilot.database.applications import (
    enrich_applications_with_job_info,
    list_applications,
)
from career_copilot.database.deps import get_db
from career_copilot.schemas import TrackApplicationsChatRequest

router = APIRouter(prefix="/applications", tags=["track_applications"])

USER_ID = 1


@router.get("", response_class=HTMLResponse)
async def get_applications_page(
    request: Request,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> HTMLResponse:
    """Track applications page: list of applications + agent chat."""
    rows = list_applications(conn, USER_ID)
    applications = enrich_applications_with_job_info(conn, rows)
    conn.close()
    return templates.TemplateResponse(
        "track_applications.html",
        {
            "request": request,
            "applications": applications,
            "user_id": USER_ID,
        },
    )


@router.post("/chat")
async def post_applications_chat(
    body: TrackApplicationsChatRequest,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> JSONResponse:
    """
    Chat with the track-applications agent (tool calling). Returns reply and
    updated applications list so the frontend can refresh.
    """
    try:
        reply = chat_track_applications(
            body.message or "",
            body.history or [],
            conn,
            USER_ID,
        )
    except Exception as e:
        conn.close()
        return JSONResponse(
            status_code=500,
            content={"reply": f"Sorry, something went wrong. ({e!s})", "applications": []},
        )
    rows = list_applications(conn, USER_ID)
    applications = enrich_applications_with_job_info(conn, rows)
    conn.close()
    return JSONResponse(
        content={
            "reply": reply,
            "applications": applications,
        },
    )
