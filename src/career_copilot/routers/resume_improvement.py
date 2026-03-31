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
    format_resume_via_mcp,
    generate_full_resume,
    get_initial_resume_analysis,
)
from career_copilot.constants import (
    APPLICATION_CHAT_MAX_STORED_MESSAGES,
    APPLICATION_MEMORY_SUMMARY_UPDATE_EVERY_N_MESSAGES,
    DEFAULT_USER_ID,
)
from career_copilot.database.applications import add_application as add_tracked_application
from career_copilot.database.applications import (
    get_application_by_key,
    set_application_history,
    set_application_last_resume_text,
    set_application_memory,
)
from career_copilot.database.deps import get_db
from career_copilot.database.profiles import get_resume_file_by_user_id
from career_copilot.resume_formatter.docx_builder import generate_formatted_docx
from career_copilot.resume_formatter.pdf_builder import generate_formatted_pdf
from career_copilot.resume_formatter.structure_parser import (
    StyleProfile,
    apply_original_bold,
    parse_resume_structure,
    parse_resume_structure_docx,
)
from career_copilot.resume_pdf import build_resume_pdf
from career_copilot.schemas import ResumeChatRequest, ResumePdfRequest

router = APIRouter(tags=["resume_improvement"])

USER_ID = DEFAULT_USER_ID
MAX_STORED_MESSAGES = APPLICATION_CHAT_MAX_STORED_MESSAGES


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
    # Ensure this stage shows up in Track applications
    add_tracked_application(
        conn,
        USER_ID,
        job_id,
        "ingested",
        "resume_improvement",
        status="active",
    )
    conn.commit()

    app_row = get_application_by_key(conn, USER_ID, job_id, "ingested", "resume_improvement")
    stored_history = (app_row[6] if app_row else None) or []
    last_resume_text = app_row[8] if app_row else None

    ctx = build_resume_improvement_context(job_id, USER_ID, conn)
    conn.close()
    # Use the last saved resume version as context (this is the "memory" that matters most)
    resume_text = (last_resume_text or "") or ctx["resume_text"]
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
        # If we already have history for this application, don't overwrite it.
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

    # Persist history + last resume version per application
    try:
        conn2 = get_db()
        try:
            # Re-read app row (it should exist) and update
            row2 = get_application_by_key(conn2, USER_ID, job_id, "ingested", "resume_improvement")
            if row2:
                app_id2 = int(row2[0])
                history_now = (
                    (row2[6] or []) if is_initial and stored_history else (stored_history or [])
                )
                if is_initial and not stored_history:
                    history_now = [{"role": "assistant", "content": reply}]
                elif not is_initial:
                    history_now = list(stored_history or []) + [
                        {"role": "user", "content": (body.message or "").strip()},
                        {"role": "assistant", "content": reply},
                    ]
                set_application_history(conn2, USER_ID, app_id2, history_now)

                # Only update last_resume_text after non-initial turns (or if we just created history)
                try:
                    updated_resume = generate_full_resume(
                        history_now, resume_text, job, similar_jobs, similar_resumes
                    )
                except Exception:
                    updated_resume = None
                set_application_last_resume_text(conn2, USER_ID, app_id2, updated_resume)

                # Keep history lightweight: store only last N messages
                if len(history_now) > MAX_STORED_MESSAGES:
                    history_now = history_now[-MAX_STORED_MESSAGES:]
                    set_application_history(conn2, USER_ID, app_id2, history_now)

                # Update compact memory (no full history)
                mem = (row2[7] or {}) if isinstance(row2[7], dict) else {}
                mem = {**mem}
                if updated_resume:
                    mem["current_resume_text"] = updated_resume
                # Refresh summary periodically (cheap, compact memory)
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
                set_application_memory(conn2, USER_ID, app_id2, mem)
                conn2.commit()
        finally:
            conn2.close()
    except Exception:
        # Don't fail the chat if persistence fails
        pass
    return JSONResponse(content={"reply": reply})


def _get_improved_text(job_id: int, body_history: list | None) -> str:
    """Resolve the latest improved resume text from DB or regenerate from history."""
    conn = get_db()
    try:
        app_row = get_application_by_key(conn, USER_ID, job_id, "ingested", "resume_improvement")
        last_resume_text = app_row[8] if app_row else None
        if last_resume_text:
            return last_resume_text

        ctx = build_resume_improvement_context(job_id, USER_ID, conn)
    finally:
        conn.close()

    resume_text = ctx["resume_text"]
    job = ctx["job"]
    similar_jobs = ctx["similar_jobs"]
    similar_resumes = ctx["similar_resumes"]

    history = body_history or []
    try:
        text = generate_full_resume(history, resume_text, job, similar_jobs, similar_resumes)
    except Exception:
        text = resume_text or ""

    return text or ""


def _get_mcp_server_url() -> str | None:
    import os

    return os.environ.get("MCP_SERVER_URL") or None


def _get_style_profile() -> StyleProfile:
    """Load the user's original resume and extract its StyleProfile."""
    conn = get_db()
    try:
        resume_bytes, filename = get_resume_file_by_user_id(conn, USER_ID)
    finally:
        conn.close()

    if resume_bytes:
        if (filename or "").lower().endswith((".docx", ".doc")):
            return parse_resume_structure_docx(resume_bytes)
        return parse_resume_structure(resume_bytes)
    return StyleProfile.default()


@router.post("/jobs/{job_id:int}/improve-resume/download")
async def post_resume_improve_download(
    job_id: int,
    body: ResumePdfRequest,
) -> StreamingResponse:
    """Generate a formatted PDF of the improved resume, cloning the original's style."""
    text = _get_improved_text(job_id, body.history)
    profile = _get_style_profile()
    text = apply_original_bold(text, profile.bold_phrases)
    mcp_url = _get_mcp_server_url()
    if mcp_url:
        try:
            import dataclasses

            profile_for_mcp = dataclasses.replace(profile, bold_phrases=[])
            pdf_bytes = format_resume_via_mcp(text, profile_for_mcp.to_json(), "pdf", mcp_url)
        except Exception as _mcp_err:
            print(f"[MCP] format failed: {_mcp_err!r}, falling back to direct")
            mcp_url = None
    if not mcp_url:
        try:
            pdf_bytes = generate_formatted_pdf(text, profile)
        except Exception as _pdf_err:
            print(f"[PDF] generate_formatted_pdf failed: {_pdf_err!r}")
            pdf_bytes = build_resume_pdf(text)

    filename = f"improved_resume_job_{job_id}.pdf"
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/jobs/{job_id:int}/improve-resume/download-docx")
async def post_resume_improve_download_docx(
    job_id: int,
    body: ResumePdfRequest,
) -> StreamingResponse:
    """Generate a formatted Word document of the improved resume, cloning the original's style."""
    text = _get_improved_text(job_id, body.history)
    profile = _get_style_profile()
    text = apply_original_bold(text, profile.bold_phrases)
    mcp_url = _get_mcp_server_url()
    if mcp_url:
        try:
            import dataclasses

            profile_for_mcp = dataclasses.replace(profile, bold_phrases=[])
            docx_bytes = format_resume_via_mcp(text, profile_for_mcp.to_json(), "docx", mcp_url)
        except Exception as _mcp_err:
            print(f"[MCP] format failed: {_mcp_err!r}, falling back to direct")
            mcp_url = None
    if not mcp_url:
        docx_bytes = generate_formatted_docx(text, profile)

    filename = f"improved_resume_job_{job_id}.docx"
    return StreamingResponse(
        BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
