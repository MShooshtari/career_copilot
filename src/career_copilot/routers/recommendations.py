"""Job recommendations (RAG similarity to user profile)."""

from __future__ import annotations

import copy
import threading
import time
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from career_copilot.app_config import templates
from career_copilot.auth.current_user import CurrentUserId
from career_copilot.constants import (
    RAG_DEFAULT_RECOMMENDATION_N_RESULTS,
    RECOMMENDATIONS_CACHE_TTL_SECONDS,
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
    get_job_by_id,
    get_job_interactions_map,
    get_user_job_by_id,
    list_user_jobs,
    resolve_job_ids,
    row_to_job_dict,
    set_job_feedback,
    user_job_row_to_dict,
)
from career_copilot.ml.inference import score_candidates_by_distance
from career_copilot.ml.reranking import rerank_with_diversity_and_exploration
from career_copilot.rag.pgvector_rag import get_recommended_job_results

router = APIRouter(tags=["recommendations"])


class _TTLCache:
    def __init__(self, *, ttl_seconds: int, max_entries: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._items: dict[tuple[int], tuple[float, list[dict]]] = {}
        self._lock = threading.Lock()

    def get(self, key: tuple[int]) -> list[dict] | None:
        now = time.time()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if now >= expires_at:
                self._items.pop(key, None)
                return None
            return copy.deepcopy(value)

    def set(self, key: tuple[int], value: list[dict]) -> None:
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            if len(self._items) >= self._max_entries and key not in self._items:
                oldest_key = min(self._items, key=lambda k: self._items[k][0])
                self._items.pop(oldest_key, None)
            self._items[key] = (now + self._ttl_seconds, copy.deepcopy(value))

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _purge_expired(self, now: float) -> None:
        expired = [key for key, (expires_at, _value) in self._items.items() if now >= expires_at]
        for key in expired:
            self._items.pop(key, None)


_online_recommendations_cache = _TTLCache(
    ttl_seconds=RECOMMENDATIONS_CACHE_TTL_SECONDS,
    max_entries=128,
)


def clear_recommendation_caches() -> None:
    _online_recommendations_cache.clear()


def _attach_feedback(
    jobs: list[dict],
    interactions_by_job_id: dict[int, set[str]],
) -> list[dict]:
    for job in jobs:
        job_id = job.get("job_id")
        if job_id is not None:
            interactions = interactions_by_job_id.get(int(job_id), set())
            job["feedback"] = "disliked" if "disliked" in interactions else None
            if "liked" in interactions:
                job["feedback"] = "liked"
            job["applied"] = "applied" in interactions
            job["deleted"] = "deleted" in interactions
    return jobs


def _drop_hidden_interactions(jobs: list[dict]) -> list[dict]:
    """Hide jobs after refresh once the user has dismissed or applied to them."""
    return [
        job
        for job in jobs
        if job.get("feedback") != "disliked" and not job.get("applied") and not job.get("deleted")
    ]


def _online_recommendation_jobs(conn: psycopg.Connection, user_id: int) -> list[dict]:
    cache_key = (int(user_id),)
    cached = _online_recommendations_cache.get(cache_key)
    if cached is not None:
        return cached

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
    _online_recommendations_cache.set(cache_key, jobs_online)
    return jobs_online


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

    jobs_online = _online_recommendation_jobs(conn, user_id)
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
    if feedback not in {"liked", "disliked", "applied"}:
        conn.close()
        raise HTTPException(status_code=400, detail="Unsupported feedback")

    set_job_feedback(conn, user_id, job_id, job_source, feedback)
    conn.commit()
    clear_recommendation_caches()
    conn.close()
    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JSONResponse(content={"ok": True, "feedback": feedback})
    return RedirectResponse(
        url=f"/recommendations?page={page}&page_size={page_size}",
        status_code=303,
    )


@router.get(
    "/recommendations/{job_source}/{job_id:int}/apply-on-source",
    response_class=RedirectResponse,
    response_model=None,
)
async def get_apply_on_source(
    job_source: str,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> RedirectResponse:
    """Record source apply clicks as applied, then redirect to the external job URL."""
    if job_source == "ingested":
        row = get_job_by_id(conn, job_id)
        job = row_to_job_dict(row) if row else None
    elif job_source == "user":
        row = get_user_job_by_id(conn, user_id, job_id)
        job = user_job_row_to_dict(row) if row else None
    else:
        conn.close()
        raise HTTPException(status_code=400, detail="Unsupported job source")

    if not job:
        conn.close()
        return RedirectResponse(url="/recommendations", status_code=303)

    set_job_feedback(conn, user_id, job_id, job_source, "applied")
    conn.commit()
    clear_recommendation_caches()
    conn.close()

    target_url = job.get("url") or "/applications"
    return RedirectResponse(url=target_url, status_code=303)
