"""Track applications: resume improvement and interview preparation for jobs."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from career_copilot.agents.track_applications import chat_track_applications
from career_copilot.app_config import templates
from career_copilot.auth.current_user import CurrentUserId
from career_copilot.database.applications import (
    enrich_applications_with_job_info,
    get_application_by_key,
    list_applications,
    remove_application,
)
from career_copilot.database.deps import get_db
from career_copilot.database.jobs import (
    list_jobs_with_feedback,
    remove_job_feedback,
    set_job_feedback,
)
from career_copilot.routers.recommendations import clear_recommendation_caches
from career_copilot.schemas import TrackApplicationsChatRequest

router = APIRouter(prefix="/applications", tags=["track_applications"])
FEEDBACK_PAGE_SIZE = 5


def _paginate_feedback_jobs(jobs: list[dict], page: int) -> tuple[list[dict], int]:
    total_pages = max(1, (len(jobs) + FEEDBACK_PAGE_SIZE - 1) // FEEDBACK_PAGE_SIZE)
    page = min(max(page, 1), total_pages)
    start = (page - 1) * FEEDBACK_PAGE_SIZE
    return jobs[start : start + FEEDBACK_PAGE_SIZE], page


@router.get("", response_class=HTMLResponse)
async def get_applications_page(
    request: Request,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
    applied_page: int = Query(1, ge=1),
    disliked_page: int = Query(1, ge=1),
) -> HTMLResponse:
    """Track applications page: list of applications + agent chat."""
    rows = list_applications(conn, user_id)
    applications = enrich_applications_with_job_info(conn, rows)
    all_applied_jobs = list_jobs_with_feedback(conn, user_id, "applied")
    all_disliked_jobs = list_jobs_with_feedback(conn, user_id, "disliked")
    applied_jobs, applied_page = _paginate_feedback_jobs(all_applied_jobs, applied_page)
    disliked_jobs, disliked_page = _paginate_feedback_jobs(all_disliked_jobs, disliked_page)
    conn.close()
    return templates.TemplateResponse(
        request,
        "track_applications.html",
        {
            "applications": applications,
            "applied_jobs": applied_jobs,
            "applied_total": len(all_applied_jobs),
            "applied_page": applied_page,
            "applied_total_pages": max(
                1, (len(all_applied_jobs) + FEEDBACK_PAGE_SIZE - 1) // FEEDBACK_PAGE_SIZE
            ),
            "disliked_jobs": disliked_jobs,
            "disliked_total": len(all_disliked_jobs),
            "disliked_page": disliked_page,
            "disliked_total_pages": max(
                1, (len(all_disliked_jobs) + FEEDBACK_PAGE_SIZE - 1) // FEEDBACK_PAGE_SIZE
            ),
            "user_id": user_id,
        },
    )


@router.get("/context")
async def get_application_context(
    job_id: int,
    job_source: str,
    stage: str,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> JSONResponse:
    """
    Return persisted per-application context (chat history + last resume text)
    for the given (job_id, job_source, stage).
    """
    if job_source not in ("ingested", "user") or stage not in (
        "resume_improvement",
        "interview_preparation",
    ):
        conn.close()
        return JSONResponse(status_code=400, content={"found": False, "history": []})
    row = get_application_by_key(conn, user_id, job_id, job_source, stage)
    conn.close()
    if not row:
        return JSONResponse(content={"found": False, "history": [], "last_resume_text": None})
    # Row shape includes history + last_resume_text per applications.py
    history = row[6] or []
    application_memory = row[7] or {}
    last_resume_text = row[8]
    return JSONResponse(
        content={
            "found": True,
            "history": history,
            "application_memory": application_memory,
            "last_resume_text": last_resume_text,
        }
    )


@router.post("/chat")
async def post_applications_chat(
    body: TrackApplicationsChatRequest,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> JSONResponse:
    """
    Chat with the track-applications agent (tool calling). Returns reply and
    updated applications list so the frontend can refresh.
    """
    try:
        reply = chat_track_applications(
            body.message or "",
            body.history or [],
            conn,
            user_id,
        )
    except Exception as e:
        conn.close()
        return JSONResponse(
            status_code=500,
            content={"reply": f"Sorry, something went wrong. ({e!s})", "applications": []},
        )
    rows = list_applications(conn, user_id)
    applications = enrich_applications_with_job_info(conn, rows)
    conn.close()
    return JSONResponse(
        content={
            "reply": reply,
            "applications": applications,
        },
    )


@router.post("/{application_id:int}/delete")
async def post_delete_application(
    application_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> JSONResponse:
    """Delete a tracked application by id and return refreshed list."""
    removed = remove_application(conn, user_id, application_id)
    if removed:
        conn.commit()
    rows = list_applications(conn, user_id)
    applications = enrich_applications_with_job_info(conn, rows)
    conn.close()
    return JSONResponse(content={"removed": bool(removed), "applications": applications})


@router.post("/jobs/{job_source}/{job_id:int}/delete", response_class=RedirectResponse)
async def post_delete_feedback_job(
    job_source: str,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
    applied_page: int = Query(1, ge=1),
    disliked_page: int = Query(1, ge=1),
) -> RedirectResponse:
    """Soft-hide a job from recommendations and feedback recovery lists."""
    if job_source not in {"ingested", "user"}:
        conn.close()
        raise HTTPException(status_code=400, detail="Unsupported job source")

    set_job_feedback(conn, user_id, job_id, job_source, "deleted")
    conn.commit()
    clear_recommendation_caches()
    conn.close()
    return RedirectResponse(
        url=f"/applications?applied_page={applied_page}&disliked_page={disliked_page}",
        status_code=303,
    )


@router.post(
    "/jobs/{job_source}/{job_id:int}/feedback/{feedback}/restore",
    response_class=RedirectResponse,
)
async def post_restore_feedback_job(
    job_source: str,
    job_id: int,
    feedback: str,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
    applied_page: int = Query(1, ge=1),
    disliked_page: int = Query(1, ge=1),
) -> RedirectResponse:
    """Remove an accidental disliked/applied interaction so the job can be recommended again."""
    if job_source not in {"ingested", "user"}:
        conn.close()
        raise HTTPException(status_code=400, detail="Unsupported job source")
    if feedback not in {"disliked", "applied"}:
        conn.close()
        raise HTTPException(status_code=400, detail="Unsupported feedback")

    remove_job_feedback(conn, user_id, job_id, job_source, feedback)
    conn.commit()
    clear_recommendation_caches()
    conn.close()
    return RedirectResponse(
        url=f"/applications?applied_page={applied_page}&disliked_page={disliked_page}",
        status_code=303,
    )
