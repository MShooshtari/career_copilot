"""FastAPI application: Career Copilot user profile and job recommendations."""

from __future__ import annotations

import os

from fastapi import FastAPI

from career_copilot.database.db import connect
from career_copilot.database.schema import init_schema
from career_copilot.routers import (
    home,
    interview_preparation,
    jobs,
    profile,
    recommendations,
    resume_improvement,
)

app = FastAPI(title="Career Copilot - User Profile")

app.include_router(home.router)
app.include_router(recommendations.router)
app.include_router(jobs.router)
app.include_router(profile.router)
app.include_router(resume_improvement.router)
app.include_router(interview_preparation.router)


def get_db():
    """Return a DB connection (caller should close). Used by startup; routers use database.deps.get_db."""
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
