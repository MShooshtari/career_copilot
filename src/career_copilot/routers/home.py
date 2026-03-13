"""Root and redirect routes."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(tags=["home"])


@router.get("/", response_class=HTMLResponse)
async def home() -> RedirectResponse:
    return RedirectResponse(url="/profile", status_code=303)
