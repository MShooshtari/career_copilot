"""FastAPI application: Career Copilot user profile and job recommendations."""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response

from career_copilot.auth.config import auth_enabled, session_secret_key
from career_copilot.database.db import connect
from career_copilot.database.schema import init_schema
from career_copilot.routers import (
    add_job,
    auth,
    home,
    interview_preparation,
    jobs,
    market_analysis,
    my_jobs,
    profile,
    recommendations,
    resume_improvement,
    track_applications,
)

app = FastAPI(title="Career Copilot - User Profile")

if auth_enabled():
    secret = session_secret_key()
    if not secret:
        raise RuntimeError("AUTH_ENABLED=1 requires SESSION_SECRET_KEY")
    app.add_middleware(SessionMiddleware, secret_key=secret)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> Response:
    """
    Allow dependencies to return HTML pages for browser flows by setting
    `HTTPException(detail=<starlette Response>)`.

    FastAPI's default handler always JSON-encodes `detail`, which breaks for Response objects.
    """
    if isinstance(exc.detail, Response):
        return exc.detail
    # Defer to FastAPI defaults for normal string/dict details.
    from fastapi.exception_handlers import http_exception_handler

    return await http_exception_handler(request, exc)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


app.include_router(home.router)
app.include_router(auth.router)
app.include_router(recommendations.router)
app.include_router(market_analysis.router)
app.include_router(add_job.router)
app.include_router(my_jobs.router)
app.include_router(jobs.router)
app.include_router(profile.router)
app.include_router(resume_improvement.router)
app.include_router(interview_preparation.router)
app.include_router(track_applications.router)


def get_db():
    """Return a DB connection (caller should close).

    Used by startup; routers use `career_copilot.database.deps.get_db`.
    """
    return connect()


@app.on_event("startup")
def _startup() -> None:
    if os.environ.get("TESTING") == "1":
        return
    try:
        conn = get_db()
        try:
            init_schema(conn)
        finally:
            conn.close()
    except Exception:
        # No DB available or connection failed (e.g. CI); continue without schema init
        pass
