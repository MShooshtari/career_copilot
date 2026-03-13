"""Resume improvement chat and PDF download API."""

from __future__ import annotations

from io import BytesIO
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from career_copilot.agents.resume_improvement import (
    build_resume_improvement_context,
    chat_resume_improvement,
    generate_full_resume,
    get_initial_resume_analysis,
)
from career_copilot.database.deps import get_db
from career_copilot.pdf_renderer import render_resume_pdf
from career_copilot.schemas import ResumeChatRequest, ResumePdfRequest

router = APIRouter(tags=["resume_improvement"])

USER_ID = 1


@router.post("/jobs/{job_id:int}/improve-resume/chat")
async def post_resume_improve_chat(
    job_id: int,
    body: ResumeChatRequest,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> JSONResponse:
    """
    Chat endpoint for resume improvement. Send message and optional history.
    If history is empty and message is empty or 'initial', returns initial analysis.
    """
    ctx = build_resume_improvement_context(job_id, USER_ID, conn)
    conn.close()
    resume_text = ctx["resume_text"]
    job = ctx["job"]
    similar_jobs = ctx["similar_jobs"]
    similar_resumes = ctx["similar_resumes"]

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
            reply = get_initial_resume_analysis(resume_text, job, similar_jobs, similar_resumes)
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


@router.post("/jobs/{job_id:int}/improve-resume/download")
async def post_resume_improve_download(
    job_id: int,
    body: ResumePdfRequest,
) -> StreamingResponse:
    """
    Generate a simple PDF from the latest improved resume text.

    The frontend sends the full chat history. We regenerate a clean, updated resume
    (no commentary) from the original resume + job + RAG context + conversation,
    then render that into a simple PDF.
    """
    conn = get_db()
    try:
        ctx = build_resume_improvement_context(job_id, USER_ID, conn)
    finally:
        conn.close()
    resume_text = ctx["resume_text"]
    job = ctx["job"]
    similar_jobs = ctx["similar_jobs"]
    similar_resumes = ctx["similar_resumes"]

    history = body.history or []
    try:
        text = generate_full_resume(history, resume_text, job, similar_jobs, similar_resumes)
    except Exception:
        text = resume_text or ""

    pdf_bytes = render_resume_pdf(text or "")
    filename = f"improved_resume_job_{job_id}.pdf"
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
