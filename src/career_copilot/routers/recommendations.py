"""Job recommendations (RAG similarity to user profile)."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse

from career_copilot.app_config import templates
from career_copilot.auth.current_user import CurrentUserId
from career_copilot.constants import (
    RAG_DEFAULT_RECOMMENDATION_N_RESULTS,
    RECOMMENDATIONS_CANDIDATE_POOL_SIZE,
    RECOMMENDATIONS_DEFAULT_PAGE_SIZE,
    RECOMMENDATIONS_MAX_PAGE_SIZE,
    RECOMMENDATIONS_PAGE_SIZE_OPTIONS,
    RECOMMENDATIONS_RERANK_WINDOW_SIZE,
)
from career_copilot.database.deps import get_db
from career_copilot.database.jobs import (
    format_recommendation_jobs,
    format_user_jobs_for_recommendations,
    list_user_jobs,
    resolve_job_ids,
)
from career_copilot.ml.inference import score_candidates_by_distance
from career_copilot.rag.pgvector_rag import get_recommended_job_results

router = APIRouter(tags=["recommendations"])


@router.get("/recommendations", response_class=HTMLResponse)
async def get_recommendations(
    request: Request,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
    page: int = Query(1, ge=1),
    page_size: int = Query(
        RECOMMENDATIONS_DEFAULT_PAGE_SIZE, ge=1, le=RECOMMENDATIONS_MAX_PAGE_SIZE
    ),
) -> HTMLResponse:
    """Candidate retrieval: user-added jobs plus top 100 jobs by similarity to profile."""
    user_rows = list_user_jobs(conn, user_id)
    jobs_added = format_user_jobs_for_recommendations(user_rows)
    raw = get_recommended_job_results(
        conn,
        user_id=user_id,
        n_results=min(RECOMMENDATIONS_CANDIDATE_POOL_SIZE, RAG_DEFAULT_RECOMMENDATION_N_RESULTS),
    )
    raw = score_candidates_by_distance(raw)
    id_map = resolve_job_ids(conn, raw)
    # Only keep the top window after ranking (pagination is within this window).
    raw = raw[:RECOMMENDATIONS_RERANK_WINDOW_SIZE]
    jobs_online = format_recommendation_jobs(raw, id_map)
    conn.close()
    total_online = len(jobs_online)

    # Clamp page to the available window.
    if total_online <= 0:
        page = 1
    else:
        total_pages = (total_online // page_size) + (1 if total_online % page_size else 0)
        if page > total_pages:
            page = total_pages

    start = (page - 1) * page_size
    end = start + page_size
    jobs_online_page = jobs_online[start:end]

    return templates.TemplateResponse(
        request,
        "recommendations.html",
        {
            "jobs_added": jobs_added,
            "jobs_online": jobs_online_page,
            "total_online": total_online,
            "page": page,
            "page_size": page_size,
            "page_size_options": RECOMMENDATIONS_PAGE_SIZE_OPTIONS,
            "user_id": user_id,
        },
    )
