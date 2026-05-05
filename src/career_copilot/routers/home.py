"""Root and redirect routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from career_copilot.auth.config import auth_enabled
from career_copilot.auth.guest import get_guest_subject

router = APIRouter(tags=["home"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> RedirectResponse:
    if auth_enabled():
        sess = request.session if "session" in request.scope else None
        has_external_identity = isinstance(sess, dict) and bool(sess.get("ext_identity"))
        if not (has_external_identity or get_guest_subject(sess)):
            return RedirectResponse(url="/auth/sign-in", status_code=303)
    return RedirectResponse(url="/profile", status_code=303)
