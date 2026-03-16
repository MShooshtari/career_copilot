"""Job recommendations (RAG similarity to user profile)."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from career_copilot.app_config import templates
from career_copilot.database.deps import get_db
from career_copilot.database.jobs import (
    format_recommendation_jobs,
    format_user_jobs_for_recommendations,
    list_user_jobs,
    resolve_job_ids,
)
from career_copilot.rag.chroma_store import get_recommended_job_results

router = APIRouter(tags=["recommendations"])

USER_ID = 1


@router.get("/recommendations", response_class=HTMLResponse)
async def get_recommendations(
    request: Request,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> HTMLResponse:
    """Candidate retrieval: user-added jobs plus top 100 jobs by similarity to profile."""
    user_rows = list_user_jobs(conn, USER_ID)
    jobs_added = format_user_jobs_for_recommendations(user_rows)
    raw = get_recommended_job_results(user_id=USER_ID, n_results=100)
    id_map = resolve_job_ids(conn, raw)
    jobs_online = format_recommendation_jobs(raw, id_map)
    conn.close()
    return templates.TemplateResponse(
        "recommendations.html",
        {
            "request": request,
            "jobs_added": jobs_added,
            "jobs_online": jobs_online,
            "user_id": USER_ID,
        },
    )
