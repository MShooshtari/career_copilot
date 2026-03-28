"""Add Job: user can add a job via manual text, file upload, or URL. Uses an agentic flow with tool calling and a confirmation step."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from career_copilot.agents.add_job import run_add_job_agent
from career_copilot.app_config import templates
from career_copilot.constants import DEFAULT_USER_ID
from career_copilot.database.deps import get_db
from career_copilot.database.jobs import insert_user_job
from career_copilot.ingestion.common import html_to_plain_text

router = APIRouter(tags=["add_job"])

USER_ID = DEFAULT_USER_ID


@router.get("/add-job", response_class=HTMLResponse)
async def get_add_job(request: Request) -> HTMLResponse:
    """Show Add Job form: manual entry, file upload, or URL."""
    error = request.query_params.get("error")
    msg = "Failed to save the job. Please try again." if error == "save_failed" else None
    return templates.TemplateResponse(
        request,
        "add_job.html",
        {"user_id": USER_ID, "error": msg},
    )


@router.post("/add-job", response_class=HTMLResponse, response_model=None)
async def post_add_job(
    request: Request,
    mode: str = Form("manual"),
    job_text: str = Form(""),
    location: str = Form(""),
    salary_min: str = Form(""),
    salary_max: str = Form(""),
    job_url: str = Form(""),
    manual_url: str = Form(""),
    file: UploadFile | None = None,
) -> HTMLResponse:
    """
    Run the agentic add-job flow (tool calling). On success, show the confirmation page
    with extracted fields so the user can review and edit before adding.
    """
    url_clean = (job_url or manual_url or "").strip() or None
    salary_min_int = int(salary_min) if salary_min.strip().isdigit() else None
    salary_max_int = int(salary_max) if salary_max.strip().isdigit() else None

    if mode == "url" and not url_clean:
        return templates.TemplateResponse(
            request,
            "add_job.html",
            {"user_id": USER_ID, "error": "Please enter a job URL."},
        )
    if mode == "file" and (not file or not file.filename):
        return templates.TemplateResponse(
            request,
            "add_job.html",
            {"user_id": USER_ID, "error": "Please select a file to upload."},
        )
    if mode == "manual" and not (job_text or "").strip():
        return templates.TemplateResponse(
            request,
            "add_job.html",
            {
                "user_id": USER_ID,
                "error": "Please provide job description text, a file, or a URL.",
            },
        )

    file_content: bytes | None = None
    filename: str | None = None
    if mode == "file" and file and file.filename:
        file_content = await file.read()
        filename = file.filename

    try:
        proposal = run_add_job_agent(
            mode,
            url=url_clean,
            text=(job_text or "").strip() or None,
            file_content=file_content,
            filename=filename,
            location=(location or "").strip() or None,
            salary_min=salary_min_int,
            salary_max=salary_max_int,
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "add_job.html",
            {
                "user_id": USER_ID,
                "error": f"Could not extract job details: {e!s}. Try pasting the description manually.",
            },
        )

    if not proposal or (not proposal.get("title") and not proposal.get("description")):
        return templates.TemplateResponse(
            request,
            "add_job.html",
            {
                "user_id": USER_ID,
                "error": "We couldn't extract enough job details. Try pasting the job description manually or a different URL.",
            },
        )

    # Ensure URL is set when user submitted a URL
    if mode == "url" and url_clean and not proposal.get("url"):
        proposal = {**proposal, "url": url_clean}

    return templates.TemplateResponse(
        request,
        "confirm_job.html",
        {"user_id": USER_ID, "proposal": proposal},
    )


@router.post("/add-job/confirm", response_class=RedirectResponse)
async def post_add_job_confirm(
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    title: str = Form(""),
    company: str = Form(""),
    location: str = Form(""),
    salary_min: str = Form(""),
    salary_max: str = Form(""),
    description: str = Form(""),
    skills: str = Form(""),
    url: str = Form(""),
) -> RedirectResponse:
    """Save the user-confirmed job and redirect to recommendations."""
    title_clean = (title or "").strip() or "Untitled Job"
    company_clean = (company or "").strip() or None
    location_clean = (location or "").strip() or None
    salary_min_int = int(salary_min) if (salary_min or "").strip().isdigit() else None
    salary_max_int = int(salary_max) if (salary_max or "").strip().isdigit() else None
    description_clean = (description or "").strip() or None
    skills_list = [s.strip() for s in (skills or "").split(",") if s.strip()]
    url_clean = (url or "").strip() or None

    description_clean = html_to_plain_text(description_clean) if description_clean else None

    try:
        job_id = insert_user_job(
            conn,
            USER_ID,
            title=title_clean,
            company=company_clean,
            location=location_clean,
            salary_min=salary_min_int,
            salary_max=salary_max_int,
            description=description_clean,
            skills=skills_list,
            url=url_clean,
            raw={"source": "confirm"},
        )
        conn.commit()
        conn.close()
    except Exception:
        conn.rollback()
        conn.close()
        return RedirectResponse(url="/add-job?error=save_failed", status_code=303)

    return RedirectResponse(url=f"/recommendations?added={job_id}", status_code=303)
