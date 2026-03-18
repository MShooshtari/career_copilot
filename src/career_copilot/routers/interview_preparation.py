"""Interview preparation chat API."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from career_copilot.agents.interview_preparation import (
    build_interview_prep_context,
    chat_interview_preparation,
    get_initial_interview_message,
)
from career_copilot.constants import (
    DEFAULT_USER_ID,
    INTERVIEW_PREP_MAX_STORED_MESSAGES,
    INTERVIEW_PREP_SUMMARY_UPDATE_EVERY_N_MESSAGES,
)
from career_copilot.database.applications import add_application as add_tracked_application
from career_copilot.database.applications import (
    get_application_by_key,
    set_application_history,
    set_application_memory,
)
from career_copilot.database.deps import get_db
from career_copilot.schemas import InterviewChatRequest

router = APIRouter(tags=["interview_preparation"])

USER_ID = DEFAULT_USER_ID
MAX_STORED_MESSAGES = INTERVIEW_PREP_MAX_STORED_MESSAGES


@router.post("/jobs/{job_id:int}/prepare-interview/chat")
async def post_prepare_interview_chat(
    job_id: int,
    body: InterviewChatRequest,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> JSONResponse:
    """
    Chat endpoint for interview preparation. Send message and optional history.
    If history is empty and message is 'initial', returns initial prompt asking interview type.
    Once the user replies with interview type, the backend searches the web (Glassdoor, Reddit,
    etc.) and returns a tailored preparation plan using job description and user resume.
    """
    # Ensure this stage shows up in Track applications
    add_tracked_application(
        conn,
        USER_ID,
        job_id,
        "ingested",
        "interview_preparation",
        status="active",
    )
    conn.commit()

    app_row = get_application_by_key(conn, USER_ID, job_id, "ingested", "interview_preparation")
    stored_history = (app_row[6] if app_row else None) or []

    ctx = build_interview_prep_context(job_id, USER_ID, conn)
    conn.close()
    resume_text = ctx["resume_text"]
    job = ctx["job"]

    if not job:
        return JSONResponse(
            status_code=404,
            content={"reply": "Job not found. Please go back to recommendations."},
        )

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
                    content={"reply": f"Sorry, something went wrong. ({e!s})"},
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
            row2 = get_application_by_key(
                conn2, USER_ID, job_id, "ingested", "interview_preparation"
            )
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
                set_application_history(conn2, USER_ID, app_id2, history_now)

                # Keep history lightweight
                if len(history_now) > MAX_STORED_MESSAGES:
                    history_now = history_now[-MAX_STORED_MESSAGES:]
                    set_application_history(conn2, USER_ID, app_id2, history_now)

                # Update compact memory (interview_type + summary)
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
                    len(history_now) >= INTERVIEW_PREP_SUMMARY_UPDATE_EVERY_N_MESSAGES
                    and len(history_now) % INTERVIEW_PREP_SUMMARY_UPDATE_EVERY_N_MESSAGES == 0
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
                set_application_memory(conn2, USER_ID, app_id2, mem)
                conn2.commit()
        finally:
            conn2.close()
    except Exception:
        pass
    return JSONResponse(content={"reply": reply})
