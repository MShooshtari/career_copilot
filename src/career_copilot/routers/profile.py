"""User profile and resume upload routes."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from career_copilot.app_config import templates
from career_copilot.auth.current_user import CurrentUserId
from career_copilot.database.deps import get_db
from career_copilot.database.profiles import (
    get_profile_by_user_id,
    get_resume_file_by_user_id,
    upsert_user_profile,
)
from career_copilot.rag.user_embedding import index_user_embedding
from career_copilot.resume_io import extract_resume_text
from career_copilot.routers.recommendations import clear_recommendation_caches
from career_copilot.storage.resumes import put_resume, resume_storage_mode
from career_copilot.utils import strip_nul

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def get_profile(
    request: Request,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> HTMLResponse:
    row = get_profile_by_user_id(conn, user_id)
    conn.close()
    context = {
        "profile": row,
        "user_id": user_id,
    }
    return templates.TemplateResponse(request, "profile.html", context)


@router.get("/resume")
async def get_resume(
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> Response:
    """Download the current user's stored resume file."""
    data, filename = get_resume_file_by_user_id(conn, user_id)
    conn.close()
    if data is None:
        return Response(status_code=404)
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("", response_class=HTMLResponse)
@router.post("/", response_class=HTMLResponse)
async def post_profile(
    request: Request,
    user_id: CurrentUserId,
    skill_tags: str = Form(...),
    years_experience: int | None = Form(None),
    current_location: str = Form(""),
    preferred_roles: str = Form(""),
    industries: str = Form(""),
    work_mode: str = Form(""),
    employment_type: str = Form(""),
    preferred_locations: str = Form(""),
    salary_min: int | None = Form(None),
    salary_max: int | None = Form(None),
    resume_file: UploadFile | None = File(None),
) -> HTMLResponse:
    content_bytes: bytes | None = None
    resume_filename: str | None = None
    if resume_file is not None and resume_file.filename:
        content_bytes = await resume_file.read()
        resume_filename = resume_file.filename.strip() or None
        if not content_bytes:
            content_bytes = None
            resume_filename = None

    skill_tags = strip_nul(skill_tags or "")
    current_location = strip_nul(current_location or "")
    preferred_roles = strip_nul(preferred_roles or "")
    industries = strip_nul(industries or "")
    work_mode = strip_nul(work_mode or "")
    employment_type = strip_nul(employment_type or "")
    preferred_locations = strip_nul(preferred_locations or "")

    resume_text_for_embedding = ""
    if content_bytes:
        resume_text_for_embedding = strip_nul(extract_resume_text(content_bytes, resume_filename))

    conn = get_db()
    try:
        if content_bytes is None:
            existing_data, existing_name = get_resume_file_by_user_id(conn, user_id)
            if existing_data is not None:
                content_bytes = existing_data
                resume_filename = existing_name

        resume_blob_container = None
        resume_blob_name = None
        if content_bytes and resume_storage_mode() == "blob":
            resume_blob_container, resume_blob_name = put_resume(
                user_id=user_id,
                filename=resume_filename,
                content=content_bytes,
            )
            # Keep DB small; store bytes only when not using blob mode.
            content_bytes = None

        upsert_user_profile(
            conn,
            user_id=user_id,
            skill_tags=skill_tags,
            years_experience=years_experience,
            current_location=current_location,
            preferred_roles=preferred_roles,
            industries=industries,
            work_mode=work_mode,
            employment_type=employment_type,
            preferred_locations=preferred_locations,
            salary_min=salary_min,
            salary_max=salary_max,
            resume_file=content_bytes,
            resume_filename=resume_filename,
            resume_blob_container=resume_blob_container,
            resume_blob_name=resume_blob_name,
        )

        if content_bytes and not resume_text_for_embedding:
            resume_text_for_embedding = strip_nul(
                extract_resume_text(content_bytes, resume_filename)
            )
        index_user_embedding(
            conn,
            user_id=user_id,
            resume_text=resume_text_for_embedding,
            skill_tags=skill_tags,
            preferred_roles=preferred_roles,
            industries=industries,
            work_mode=work_mode,
            employment_type=employment_type,
            preferred_locations=preferred_locations,
        )
        conn.commit()
        clear_recommendation_caches()
    finally:
        conn.close()

    return RedirectResponse(url="/profile", status_code=303)
