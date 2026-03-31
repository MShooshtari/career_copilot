"""Root and redirect routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from career_copilot.auth.config import auth_enabled

router = APIRouter(tags=["home"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> RedirectResponse:
    if auth_enabled():
        sess = request.session if "session" in request.scope else None
        if not (isinstance(sess, dict) and sess.get("ext_identity")):
            return RedirectResponse(url="/auth/login", status_code=303)
    return RedirectResponse(url="/profile", status_code=303)
