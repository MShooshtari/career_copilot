"""My Jobs: user-added jobs (detail, improve resume, prepare interview)."""

from __future__ import annotations

from io import BytesIO
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from career_copilot.agents.interview_preparation import (
    build_interview_prep_context_from_job_dict,
    chat_interview_preparation,
    get_initial_interview_message,
)
from career_copilot.agents.resume_improvement import (
    build_resume_improvement_context_from_job_dict,
    chat_resume_improvement,
    generate_full_resume,
    get_initial_resume_analysis,
)
from career_copilot.app_config import templates
from career_copilot.database.deps import get_db
from career_copilot.database.jobs import delete_user_job, get_user_job_by_id, user_job_row_to_dict
from career_copilot.ingestion.common import html_to_plain_text
from career_copilot.resume_pdf import build_resume_pdf
from career_copilot.schemas import InterviewChatRequest, ResumeChatRequest, ResumePdfRequest

router = APIRouter(prefix="/my-jobs", tags=["my_jobs"])

USER_ID = 1


def _get_user_job_dict(conn: psycopg.Connection, job_id: int) -> dict | None:
    row = get_user_job_by_id(conn, USER_ID, job_id)
    if not row:
        return None
    return user_job_row_to_dict(row)


@router.post("/{job_id:int}/delete", response_class=RedirectResponse, response_model=None)
async def post_my_job_delete(
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> RedirectResponse:
    """Delete a user-added job and redirect to recommendations."""
    delete_user_job(conn, USER_ID, job_id)
    conn.commit()
    conn.close()
    return RedirectResponse(url="/recommendations", status_code=303)


@router.get("/{job_id:int}", response_class=HTMLResponse, response_model=None)
async def get_my_job_detail(
    request: Request,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> HTMLResponse | RedirectResponse:
    """Full job details for a user-added job."""
    job = _get_user_job_dict(conn, job_id)
    conn.close()
    if not job:
        return RedirectResponse(url="/recommendations", status_code=303)
    if job.get("description"):
        job = {**job, "description": html_to_plain_text(job["description"]) or job["description"]}
    return templates.TemplateResponse(
        "job_detail.html",
        {"request": request, "job": job, "user_id": USER_ID, "is_user_job": True},
    )


@router.get("/{job_id:int}/improve-resume", response_class=HTMLResponse, response_model=None)
async def get_my_job_improve_resume(
    request: Request,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> HTMLResponse | RedirectResponse:
    job = _get_user_job_dict(conn, job_id)
    conn.close()
    if not job:
        return RedirectResponse(url="/recommendations", status_code=303)
    desc = (job.get("description") or "")[:500]
    if len(job.get("description") or "") > 500:
        job = {**job, "description": desc.rstrip() + "…"}
    return templates.TemplateResponse(
        "improve_resume.html",
        {
            "request": request,
            "job": job,
            "job_id": job_id,
            "user_id": USER_ID,
            "is_user_job": True,
        },
    )


@router.get("/{job_id:int}/prepare-interview", response_class=HTMLResponse, response_model=None)
async def get_my_job_prepare_interview(
    request: Request,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> HTMLResponse | RedirectResponse:
    job = _get_user_job_dict(conn, job_id)
    conn.close()
    if not job:
        return RedirectResponse(url="/recommendations", status_code=303)
    desc = (job.get("description") or "")[:500]
    if len(job.get("description") or "") > 500:
        job = {**job, "description": desc.rstrip() + "…"}
    return templates.TemplateResponse(
        "prepare_interview.html",
        {
            "request": request,
            "job": job,
            "job_id": job_id,
            "user_id": USER_ID,
            "is_user_job": True,
        },
    )


@router.post("/{job_id:int}/improve-resume/chat")
async def post_my_job_improve_resume_chat(
    job_id: int,
    body: ResumeChatRequest,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> JSONResponse:
    job = _get_user_job_dict(conn, job_id)
    if not job:
        conn.close()
        return JSONResponse(
            status_code=404,
            content={"reply": "Job not found. Please go back to recommendations."},
        )
    ctx = build_resume_improvement_context_from_job_dict(job, USER_ID, conn)
    conn.close()
    resume_text = ctx["resume_text"]
    similar_jobs = ctx["similar_jobs"]
    similar_resumes = ctx["similar_resumes"]

    is_initial = not (body.history or []) and (
        not (body.message or "").strip() or (body.message or "").strip().lower() == "initial"
    )
    if is_initial:
        try:
            reply = get_initial_resume_analysis(
                resume_text, job, similar_jobs, similar_resumes
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"reply": f"Sorry, I couldn't run the analysis. ({e!s})"},
            )
    else:
        try:
            reply = chat_resume_improvement(
                body.message or "",
                body.history or [],
                resume_text,
                job,
                similar_jobs,
                similar_resumes,
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"reply": f"Sorry, something went wrong. ({e!s})"},
            )
    return JSONResponse(content={"reply": reply})


@router.post("/{job_id:int}/improve-resume/download")
async def post_my_job_improve_resume_download(
    job_id: int,
    body: ResumePdfRequest,
) -> StreamingResponse:
    conn = get_db()
    job = _get_user_job_dict(conn, job_id)
    if not job:
        conn.close()
        return StreamingResponse(
            BytesIO(b""),
            status_code=404,
            media_type="application/pdf",
        )
    try:
        ctx = build_resume_improvement_context_from_job_dict(job, USER_ID, conn)
    finally:
        conn.close()
    resume_text = ctx["resume_text"]
    similar_jobs = ctx["similar_jobs"]
    similar_resumes = ctx["similar_resumes"]
    history = body.history or []
    try:
        text = generate_full_resume(history, resume_text, job, similar_jobs, similar_resumes)
    except Exception:
        text = resume_text or ""
    pdf_bytes = build_resume_pdf(text or "")
    filename = f"improved_resume_my_job_{job_id}.pdf"
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{job_id:int}/prepare-interview/chat")
async def post_my_job_prepare_interview_chat(
    job_id: int,
    body: InterviewChatRequest,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> JSONResponse:
    job = _get_user_job_dict(conn, job_id)
    if not job:
        conn.close()
        return JSONResponse(
            status_code=404,
            content={"reply": "Job not found. Please go back to recommendations."},
        )
    ctx = build_interview_prep_context_from_job_dict(job, USER_ID, conn)
    conn.close()
    resume_text = ctx["resume_text"]

    is_initial = not (body.history or []) and (
        not (body.message or "").strip() or (body.message or "").strip().lower() == "initial"
    )
    if is_initial:
        try:
            reply = get_initial_interview_message()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"reply": f"Sorry, I couldn't run the preparation. ({e!s})"},
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
