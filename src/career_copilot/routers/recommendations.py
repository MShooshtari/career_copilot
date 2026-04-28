"""Job recommendations (RAG similarity to user profile)."""

from __future__ import annotations

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from career_copilot.app_config import templates
from career_copilot.auth.current_user import CurrentUserId
from career_copilot.constants import (
    RAG_DEFAULT_RECOMMENDATION_N_RESULTS,
    RECOMMENDATIONS_CANDIDATE_POOL_SIZE,
    RECOMMENDATIONS_DEFAULT_PAGE_SIZE,
    RECOMMENDATIONS_DIVERSITY_CATEGORY_PENALTY,
    RECOMMENDATIONS_DIVERSITY_SIMILARITY_PENALTY,
    RECOMMENDATIONS_EXPLORATION_RATE,
    RECOMMENDATIONS_MAX_PAGE_SIZE,
    RECOMMENDATIONS_PAGE_SIZE_OPTIONS,
    RECOMMENDATIONS_RERANK_WINDOW_SIZE,
)
from career_copilot.database.deps import get_db
from career_copilot.database.jobs import (
    format_recommendation_jobs,
    format_user_jobs_for_recommendations,
    get_job_interactions_map,
    list_user_jobs,
    resolve_job_ids,
    set_job_feedback,
)
from career_copilot.ml.inference import score_candidates_by_distance
from career_copilot.ml.reranking import rerank_with_diversity_and_exploration
from career_copilot.rag.pgvector_rag import get_recommended_job_results

router = APIRouter(tags=["recommendations"])


def _attach_feedback(
    jobs: list[dict],
    interactions_by_job_id: dict[int, set[str]],
) -> list[dict]:
    for job in jobs:
        job_id = job.get("job_id")
        if job_id is not None:
            interactions = interactions_by_job_id.get(int(job_id), set())
            job["feedback"] = "dislike" if "dislike" in interactions else None
            if "like" in interactions:
                job["feedback"] = "like"
            job["applied"] = "applied" in interactions
    return jobs


def _drop_hidden_interactions(jobs: list[dict]) -> list[dict]:
    """Hide jobs after refresh once the user has dismissed or applied to them."""
    return [job for job in jobs if job.get("feedback") != "dislike" and not job.get("applied")]


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
    user_interactions = get_job_interactions_map(
        conn,
        user_id,
        "user",
        [int(job["job_id"]) for job in jobs_added if job.get("job_id") is not None],
    )
    _attach_feedback(jobs_added, user_interactions)
    jobs_added = _drop_hidden_interactions(jobs_added)

    raw = get_recommended_job_results(
        conn,
        user_id=user_id,
        n_results=min(RECOMMENDATIONS_CANDIDATE_POOL_SIZE, RAG_DEFAULT_RECOMMENDATION_N_RESULTS),
    )
    raw = score_candidates_by_distance(raw)
    raw = rerank_with_diversity_and_exploration(
        raw,
        window_size=RECOMMENDATIONS_RERANK_WINDOW_SIZE,
        user_id=user_id,
        diversity_penalty=RECOMMENDATIONS_DIVERSITY_SIMILARITY_PENALTY,
        category_penalty=RECOMMENDATIONS_DIVERSITY_CATEGORY_PENALTY,
        exploration_rate=RECOMMENDATIONS_EXPLORATION_RATE,
    )
    id_map = resolve_job_ids(conn, raw)
    jobs_online = format_recommendation_jobs(raw, id_map)
    online_interactions = get_job_interactions_map(
        conn,
        user_id,
        "ingested",
        [int(job["job_id"]) for job in jobs_online if job.get("job_id") is not None],
    )
    _attach_feedback(jobs_online, online_interactions)
    jobs_online = _drop_hidden_interactions(jobs_online)
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


@router.post(
    "/recommendations/{job_source}/{job_id:int}/feedback/{feedback}",
    response_class=RedirectResponse,
    response_model=None,
)
async def post_job_feedback(
    job_source: str,
    job_id: int,
    feedback: str,
    request: Request,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
    page: int = Query(1, ge=1),
    page_size: int = Query(
        RECOMMENDATIONS_DEFAULT_PAGE_SIZE, ge=1, le=RECOMMENDATIONS_MAX_PAGE_SIZE
    ),
) -> RedirectResponse | JSONResponse:
    """Record feedback or applied state for a shown recommendation card."""
    if job_source not in {"ingested", "user"}:
        conn.close()
        raise HTTPException(status_code=400, detail="Unsupported job source")
    if feedback not in {"like", "dislike", "applied"}:
        conn.close()
        raise HTTPException(status_code=400, detail="Unsupported feedback")

    set_job_feedback(conn, user_id, job_id, job_source, feedback)
    conn.commit()
    conn.close()
    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JSONResponse(content={"ok": True, "feedback": feedback})
    return RedirectResponse(
        url=f"/recommendations?page={page}&page_size={page_size}",
        status_code=303,
    )
