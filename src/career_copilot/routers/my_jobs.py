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
from career_copilot.auth.current_user import CurrentUserId
from career_copilot.app_config import templates
from career_copilot.constants import (
    APPLICATION_CHAT_MAX_STORED_MESSAGES,
    APPLICATION_MEMORY_SUMMARY_UPDATE_EVERY_N_MESSAGES,
    JOB_DESCRIPTION_SNIPPET_MAX_CHARS,
)
from career_copilot.database.applications import add_application as add_tracked_application
from career_copilot.database.applications import (
    get_application_by_key,
    set_application_history,
    set_application_last_resume_text,
    set_application_memory,
)
from career_copilot.database.deps import get_db
from career_copilot.database.jobs import delete_user_job, get_user_job_by_id, user_job_row_to_dict
from career_copilot.ingestion.common import html_to_plain_text
from career_copilot.resume_pdf import build_resume_pdf
from career_copilot.schemas import InterviewChatRequest, ResumeChatRequest, ResumePdfRequest

router = APIRouter(prefix="/my-jobs", tags=["my_jobs"])
MAX_STORED_MESSAGES = APPLICATION_CHAT_MAX_STORED_MESSAGES


def _get_user_job_dict(conn: psycopg.Connection, user_id: int, job_id: int) -> dict | None:
    row = get_user_job_by_id(conn, user_id, job_id)
    if not row:
        return None
    return user_job_row_to_dict(row)


@router.post("/{job_id:int}/delete", response_class=RedirectResponse, response_model=None)
async def post_my_job_delete(
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> RedirectResponse:
    """Delete a user-added job and redirect to recommendations."""
    delete_user_job(conn, user_id, job_id)
    conn.commit()
    conn.close()
    return RedirectResponse(url="/recommendations", status_code=303)


@router.get("/{job_id:int}", response_class=HTMLResponse, response_model=None)
async def get_my_job_detail(
    request: Request,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> HTMLResponse | RedirectResponse:
    """Full job details for a user-added job."""
    job = _get_user_job_dict(conn, user_id, job_id)
    conn.close()
    if not job:
        return RedirectResponse(url="/recommendations", status_code=303)
    if job.get("description"):
        job = {**job, "description": html_to_plain_text(job["description"]) or job["description"]}
    return templates.TemplateResponse(
        request,
        "job_detail.html",
        {"job": job, "user_id": user_id, "is_user_job": True},
    )


@router.get("/{job_id:int}/improve-resume", response_class=HTMLResponse, response_model=None)
async def get_my_job_improve_resume(
    request: Request,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> HTMLResponse | RedirectResponse:
    job = _get_user_job_dict(conn, user_id, job_id)
    conn.close()
    if not job:
        return RedirectResponse(url="/recommendations", status_code=303)
    desc = (job.get("description") or "")[:JOB_DESCRIPTION_SNIPPET_MAX_CHARS]
    if len(job.get("description") or "") > JOB_DESCRIPTION_SNIPPET_MAX_CHARS:
        job = {**job, "description": desc.rstrip() + "…"}
    return templates.TemplateResponse(
        request,
        "improve_resume.html",
        {
            "job": job,
            "job_id": job_id,
            "user_id": user_id,
            "is_user_job": True,
        },
    )


@router.get("/{job_id:int}/prepare-interview", response_class=HTMLResponse, response_model=None)
async def get_my_job_prepare_interview(
    request: Request,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> HTMLResponse | RedirectResponse:
    job = _get_user_job_dict(conn, user_id, job_id)
    conn.close()
    if not job:
        return RedirectResponse(url="/recommendations", status_code=303)
    desc = (job.get("description") or "")[:JOB_DESCRIPTION_SNIPPET_MAX_CHARS]
    if len(job.get("description") or "") > JOB_DESCRIPTION_SNIPPET_MAX_CHARS:
        job = {**job, "description": desc.rstrip() + "…"}
    return templates.TemplateResponse(
        request,
        "prepare_interview.html",
        {
            "job": job,
            "job_id": job_id,
            "user_id": user_id,
            "is_user_job": True,
        },
    )


@router.post("/{job_id:int}/improve-resume/chat")
async def post_my_job_improve_resume_chat(
    job_id: int,
    body: ResumeChatRequest,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> JSONResponse:
    job = _get_user_job_dict(conn, user_id, job_id)
    if not job:
        conn.close()
        return JSONResponse(
            status_code=404,
            content={"reply": "Job not found. Please go back to recommendations."},
        )

    # Ensure this stage shows up in Track applications
    add_tracked_application(
        conn,
        user_id,
        job_id,
        "user",
        "resume_improvement",
        status="active",
    )
    conn.commit()

    app_row = get_application_by_key(conn, user_id, job_id, "user", "resume_improvement")
    stored_history = (app_row[6] if app_row else None) or []
    last_resume_text = app_row[8] if app_row else None

    ctx = build_resume_improvement_context_from_job_dict(job, user_id, conn)
    conn.close()
    resume_text = (last_resume_text or "") or ctx["resume_text"]
    similar_jobs = ctx["similar_jobs"]
    similar_resumes = ctx["similar_resumes"]

    is_initial = not (body.history or []) and (
        not (body.message or "").strip() or (body.message or "").strip().lower() == "initial"
    )
    if is_initial:
        if stored_history:
            reply = "Loaded your previous resume-improvement session for this job."
        else:
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
                stored_history,
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

    # Persist history + last resume text per application
    try:
        conn2 = get_db()
        try:
            row2 = get_application_by_key(conn2, user_id, job_id, "user", "resume_improvement")
            if row2:
                app_id2 = int(row2[0])
                if is_initial and not stored_history:
                    history_now = [{"role": "assistant", "content": reply}]
                elif is_initial and stored_history:
                    history_now = list(stored_history or [])
                else:
                    history_now = list(stored_history or []) + [
                        {"role": "user", "content": (body.message or "").strip()},
                        {"role": "assistant", "content": reply},
                    ]
                set_application_history(conn2, user_id, app_id2, history_now)
                try:
                    updated_resume = generate_full_resume(
                        history_now, resume_text, job, similar_jobs, similar_resumes
                    )
                except Exception:
                    updated_resume = None
                set_application_last_resume_text(conn2, user_id, app_id2, updated_resume)

                if len(history_now) > MAX_STORED_MESSAGES:
                    history_now = history_now[-MAX_STORED_MESSAGES:]
                    set_application_history(conn2, user_id, app_id2, history_now)

                mem = (row2[7] or {}) if isinstance(row2[7], dict) else {}
                mem = {**mem}
                if updated_resume:
                    mem["current_resume_text"] = updated_resume
                if (mem.get("summary") in (None, "")) or (
                    len(history_now) >= APPLICATION_MEMORY_SUMMARY_UPDATE_EVERY_N_MESSAGES
                    and len(history_now) % APPLICATION_MEMORY_SUMMARY_UPDATE_EVERY_N_MESSAGES == 0
                ):
                    try:
                        from career_copilot.agents.application_memory import update_memory_summary

                        mem["summary"] = update_memory_summary(
                            prev_summary=str(mem.get("summary") or ""),
                            stage="resume_improvement",
                            recent_history=history_now,
                        )
                    except Exception:
                        pass
                set_application_memory(conn2, user_id, app_id2, mem)
                conn2.commit()
        finally:
            conn2.close()
    except Exception:
        pass
    return JSONResponse(content={"reply": reply})


@router.post("/{job_id:int}/improve-resume/download")
async def post_my_job_improve_resume_download(
    job_id: int,
    body: ResumePdfRequest,
    user_id: CurrentUserId,
) -> StreamingResponse:
    conn = get_db()
    job = _get_user_job_dict(conn, user_id, job_id)
    if not job:
        conn.close()
        return StreamingResponse(
            BytesIO(b""),
            status_code=404,
            media_type="application/pdf",
        )
    try:
        ctx = build_resume_improvement_context_from_job_dict(job, user_id, conn)
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
    # Prefer stored last resume for this application if present
    try:
        conn2 = get_db()
        try:
            row = get_application_by_key(conn2, user_id, job_id, "user", "resume_improvement")
            if row and row[8]:
                text = row[8]
        finally:
            conn2.close()
    except Exception:
        pass
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
    user_id: CurrentUserId,
) -> JSONResponse:
    job = _get_user_job_dict(conn, user_id, job_id)
    if not job:
        conn.close()
        return JSONResponse(
            status_code=404,
            content={"reply": "Job not found. Please go back to recommendations."},
        )

    # Ensure this stage shows up in Track applications
    add_tracked_application(
        conn,
        user_id,
        job_id,
        "user",
        "interview_preparation",
        status="active",
    )
    conn.commit()

    app_row = get_application_by_key(conn, user_id, job_id, "user", "interview_preparation")
    stored_history = (app_row[6] if app_row else None) or []

    ctx = build_interview_prep_context_from_job_dict(job, user_id, conn)
    conn.close()
    resume_text = ctx["resume_text"]

    is_initial = not (body.history or []) and (
        not (body.message or "").strip() or (body.message or "").strip().lower() == "initial"
    )
    if is_initial:
        if stored_history:
            reply = "Loaded your previous interview-prep session for this job."
        else:
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
                stored_history,
                resume_text,
                job,
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"reply": f"Sorry, something went wrong. ({e!s})"},
            )

    # Persist history per application
    try:
        conn2 = get_db()
        try:
            row2 = get_application_by_key(conn2, user_id, job_id, "user", "interview_preparation")
            if row2:
                app_id2 = int(row2[0])
                if is_initial and not stored_history:
                    history_now = [{"role": "assistant", "content": reply}]
                elif is_initial and stored_history:
                    history_now = list(stored_history or [])
                else:
                    history_now = list(stored_history or []) + [
                        {"role": "user", "content": (body.message or "").strip()},
                        {"role": "assistant", "content": reply},
                    ]
                set_application_history(conn2, user_id, app_id2, history_now)

                if len(history_now) > MAX_STORED_MESSAGES:
                    history_now = history_now[-MAX_STORED_MESSAGES:]
                    set_application_history(conn2, user_id, app_id2, history_now)

                mem = (row2[7] or {}) if isinstance(row2[7], dict) else {}
                mem = {**mem}
                if not mem.get("interview_type") and not is_initial:
                    try:
                        from career_copilot.agents.application_memory import (
                            extract_interview_type_guess,
                        )

                        guess = extract_interview_type_guess(body.message or "")
                        if guess:
                            mem["interview_type"] = guess
                    except Exception:
                        pass
                if (mem.get("summary") in (None, "")) or (
                    len(history_now) >= APPLICATION_MEMORY_SUMMARY_UPDATE_EVERY_N_MESSAGES
                    and len(history_now) % APPLICATION_MEMORY_SUMMARY_UPDATE_EVERY_N_MESSAGES == 0
                ):
                    try:
                        from career_copilot.agents.application_memory import update_memory_summary

                        mem["summary"] = update_memory_summary(
                            prev_summary=str(mem.get("summary") or ""),
                            stage="interview_preparation",
                            recent_history=history_now,
                        )
                    except Exception:
                        pass
                set_application_memory(conn2, user_id, app_id2, mem)
                conn2.commit()
        finally:
            conn2.close()
    except Exception:
        pass
    return JSONResponse(content={"reply": reply})
