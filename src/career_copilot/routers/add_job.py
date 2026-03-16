"""Add Job: user can add a job via manual text, file upload, or URL."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from career_copilot.agents.add_job import (
    extract_job_from_file,
    extract_job_from_text,
    extract_job_from_url,
)
from career_copilot.app_config import templates
from career_copilot.database.deps import get_db
from career_copilot.database.jobs import insert_user_job

router = APIRouter(tags=["add_job"])

USER_ID = 1


@router.get("/add-job", response_class=HTMLResponse)
async def get_add_job(request: Request) -> HTMLResponse:
    """Show Add Job form: manual entry, file upload, or URL."""
    return templates.TemplateResponse(
        "add_job.html",
        {"request": request, "user_id": USER_ID},
    )


@router.post("/add-job", response_class=HTMLResponse)
async def post_add_job(
    request: Request,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    mode: str = Form("manual"),
    job_text: str = Form(""),
    location: str = Form(""),
    salary_min: str = Form(""),
    salary_max: str = Form(""),
    job_url: str = Form(""),
    manual_url: str = Form(""),
    file: UploadFile | None = None,
) -> HTMLResponse | RedirectResponse:
    """
    Add a job from manual text, uploaded file, or URL.
    Extracts details via the add_job agent and saves to user_jobs.
    """
    salary_min_int = int(salary_min) if salary_min.strip().isdigit() else None
    salary_max_int = int(salary_max) if salary_max.strip().isdigit() else None
    # For URL mode use job_url; for manual mode optional hint is manual_url
    url_clean = (job_url or manual_url or "").strip() or None

    extracted: dict = {}
    if mode == "url" and url_clean:
        try:
            extracted = extract_job_from_url(url_clean)
        except Exception:
            return templates.TemplateResponse(
                "add_job.html",
                {
                    "request": request,
                    "user_id": USER_ID,
                    "error": "Could not fetch or parse the URL. Try pasting the job description manually.",
                },
            )
    elif mode == "file" and file and file.filename:
        try:
            content = await file.read()
            extracted = extract_job_from_file(content, file.filename)
        except Exception:
            return templates.TemplateResponse(
                "add_job.html",
                {
                    "request": request,
                    "user_id": USER_ID,
                    "error": "Could not read or parse the file. Try PDF, TXT, or Word (.docx).",
                },
            )
    elif mode == "manual" and (job_text or "").strip():
        try:
            extracted = extract_job_from_text(
                job_text.strip(),
                location=location.strip() or None,
                salary_min=salary_min_int,
                salary_max=salary_max_int,
                url=url_clean,
            )
        except Exception:
            return templates.TemplateResponse(
                "add_job.html",
                {
                    "request": request,
                    "user_id": USER_ID,
                    "error": "Could not extract job details. Check your text and try again.",
                },
            )
    else:
        return templates.TemplateResponse(
            "add_job.html",
            {
                "request": request,
                "user_id": USER_ID,
                "error": "Please provide job description text, a file, or a URL.",
            },
        )

    title = extracted.get("title") or "Untitled Job"
    company = extracted.get("company") or ""
    location_out = extracted.get("location") or ""
    salary_min_out = extracted.get("salary_min")
    salary_max_out = extracted.get("salary_max")
    description = extracted.get("description") or ""
    skills = extracted.get("skills") or []
    url_out = extracted.get("url")

    try:
        job_id = insert_user_job(
            conn,
            USER_ID,
            title=title,
            company=company,
            location=location_out,
            salary_min=salary_min_out,
            salary_max=salary_max_out,
            description=description,
            skills=skills,
            url=url_out,
            raw=extracted,
        )
        conn.commit()
        conn.close()
    except Exception:
        conn.rollback()
        conn.close()
        return templates.TemplateResponse(
            "add_job.html",
            {
                "request": request,
                "user_id": USER_ID,
                "error": "Saved extracted details but failed to store the job. Please try again.",
            },
        )

    return RedirectResponse(
        url=f"/recommendations?added={job_id}",
        status_code=303,
    )
