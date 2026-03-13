"""Job detail and improve-resume page routes."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from career_copilot.app_config import templates
from career_copilot.database.deps import get_db
from career_copilot.database.jobs import get_job_by_id, row_to_job_dict, row_to_job_dict_snippet

router = APIRouter(prefix="/jobs", tags=["jobs"])

USER_ID = 1


@router.get("/{job_id:int}", response_class=HTMLResponse, response_model=None)
async def get_job_detail(
    request: Request,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> HTMLResponse | RedirectResponse:
    """Full job details for a single job (by Postgres id)."""
    row = get_job_by_id(conn, job_id)
    conn.close()
    if not row:
        return RedirectResponse(url="/recommendations", status_code=303)
    job = row_to_job_dict(row)
    return templates.TemplateResponse(
        "job_detail.html",
        {"request": request, "job": job, "user_id": USER_ID},
    )


@router.get("/{job_id:int}/improve-resume", response_class=HTMLResponse, response_model=None)
async def get_improve_resume(
    request: Request,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> HTMLResponse | RedirectResponse:
    """Resume improvement chatbot: user selects a job, gets RAG-backed suggestions and can chat."""
    row = get_job_by_id(conn, job_id)
    conn.close()
    if not row:
        return RedirectResponse(url="/recommendations", status_code=303)
    job = row_to_job_dict_snippet(row, description_max_chars=500)
    return templates.TemplateResponse(
        "improve_resume.html",
        {"request": request, "job": job, "job_id": job_id, "user_id": USER_ID},
    )
