"""Job detail and improve-resume page routes."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from career_copilot.app_config import templates
from career_copilot.constants import DEFAULT_USER_ID, JOB_DESCRIPTION_SNIPPET_MAX_CHARS
from career_copilot.database.deps import get_db
from career_copilot.database.jobs import get_job_by_id, row_to_job_dict, row_to_job_dict_snippet
from career_copilot.ingestion.common import html_to_plain_text

router = APIRouter(prefix="/jobs", tags=["jobs"])

USER_ID = DEFAULT_USER_ID


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
    if job.get("description"):
        job = {**job, "description": html_to_plain_text(job["description"]) or job["description"]}
    return templates.TemplateResponse(
        request,
        "job_detail.html",
        {"job": job, "user_id": USER_ID},
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
    job = row_to_job_dict_snippet(row, description_max_chars=JOB_DESCRIPTION_SNIPPET_MAX_CHARS)
    return templates.TemplateResponse(
        request,
        "improve_resume.html",
        {"job": job, "job_id": job_id, "user_id": USER_ID},
    )


@router.get("/{job_id:int}/prepare-interview", response_class=HTMLResponse, response_model=None)
async def get_prepare_interview(
    request: Request,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> HTMLResponse | RedirectResponse:
    """Interview preparation chatbot: user picks interview type, gets tailored prep using job, resume, and web research."""
    row = get_job_by_id(conn, job_id)
    conn.close()
    if not row:
        return RedirectResponse(url="/recommendations", status_code=303)
    job = row_to_job_dict_snippet(row, description_max_chars=JOB_DESCRIPTION_SNIPPET_MAX_CHARS)
    return templates.TemplateResponse(
        request,
        "prepare_interview.html",
        {"job": job, "job_id": job_id, "user_id": USER_ID},
    )
