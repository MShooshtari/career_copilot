"""Market analysis: cohort (SQL + vector), charts data, RAG narrative."""

from __future__ import annotations

from typing import Annotated, Literal

import psycopg
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from career_copilot.app_config import templates
from career_copilot.auth.current_user import CurrentUserId
from career_copilot.constants import (
    MARKET_ANALYSIS_DEFAULT_COHORT_LIMIT,
    MARKET_ANALYSIS_DEFAULT_POSTED_WITHIN_DAYS,
)
from career_copilot.database.deps import get_db
from career_copilot.market_analysis_service import (
    MarketCohortFilters,
    build_market_analysis_report,
)

router = APIRouter(tags=["market_analysis"])


@router.get("/market-analysis", response_class=HTMLResponse)
async def market_analysis_page(
    request: Request,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
) -> HTMLResponse:
    conn.close()
    return templates.TemplateResponse(
        request,
        "market_analysis.html",
        {"user_id": user_id},
    )


@router.get("/api/market-analysis")
async def api_market_analysis(
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    user_id: CurrentUserId,
    posted_within_days: int = Query(MARKET_ANALYSIS_DEFAULT_POSTED_WITHIN_DAYS, ge=1, le=730),
    location_contains: str | None = Query(None, max_length=200),
    title_contains: str | None = Query(None, max_length=200),
    source_equals: str | None = Query(None, max_length=80),
    remote_only: bool = Query(False),
    remote_mode: Literal["both", "remote_only", "no_remote"] = Query("both"),
    salary_at_least: int | None = Query(None, ge=0, le=10_000_000),
    cohort_limit: int = Query(MARKET_ANALYSIS_DEFAULT_COHORT_LIMIT, ge=10, le=5_000),
    include_rag: bool = Query(False),
) -> JSONResponse:
    resolved_remote_mode = "remote_only" if remote_only else remote_mode
    filters = MarketCohortFilters(
        posted_within_days=posted_within_days,
        location_contains=location_contains,
        title_contains=title_contains,
        source_equals=source_equals,
        remote_mode=resolved_remote_mode,
        salary_at_least=salary_at_least,
    )
    try:
        data = build_market_analysis_report(
            conn,
            user_id=user_id,
            filters=filters,
            cohort_limit=cohort_limit,
            include_rag=include_rag,
        )
    finally:
        conn.close()
    return JSONResponse(data)
