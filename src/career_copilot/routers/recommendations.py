"""Job recommendations (RAG similarity to user profile)."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from career_copilot.app_config import templates
from career_copilot.database.deps import get_db
from career_copilot.database.jobs import format_recommendation_jobs, resolve_job_ids
from career_copilot.rag.chroma_store import get_recommended_job_results

router = APIRouter(tags=["recommendations"])

USER_ID = 1


@router.get("/recommendations", response_class=HTMLResponse)
async def get_recommendations(
    request: Request,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> HTMLResponse:
    """Candidate retrieval: top 100 jobs by similarity to current user's profile embedding."""
    raw = get_recommended_job_results(user_id=USER_ID, n_results=100)
    id_map = resolve_job_ids(conn, raw)
    jobs_for_template = format_recommendation_jobs(raw, id_map)
    return templates.TemplateResponse(
        "recommendations.html",
        {"request": request, "jobs": jobs_for_template, "user_id": USER_ID},
    )
