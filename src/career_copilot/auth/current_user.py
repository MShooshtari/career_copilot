from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from career_copilot.app_config import templates
from career_copilot.auth.config import auth_enabled
from career_copilot.auth.deps import get_external_identity
from career_copilot.constants import DEFAULT_USER_ID
from career_copilot.database.deps import get_db
from career_copilot.database.users import get_or_create_user


def _wants_html_response(request: Request) -> bool:
    accept = (request.headers.get("accept") or "").lower()
    parts = [p.strip() for p in accept.split(",") if p.strip()]
    if not parts:
        return False
    # Browsers typically send text/html early in Accept; fetch() often sends */*.
    return parts[0].startswith("text/html")


async def get_current_user_id(
    request: Request,
    ext=Depends(get_external_identity),
) -> int:
    """
    Resolve the current request's user_id.

    - When AUTH is enabled: requires an authenticated external identity and maps it to an internal user id.
    - When AUTH is disabled: returns DEFAULT_USER_ID (local demo mode), and ensures a local user row exists.
    """
    # Tests patch out DB access; keep request handlers working without a real DB.
    if os.environ.get("TESTING") == "1":
        return DEFAULT_USER_ID

    conn = get_db()
    try:
        if not auth_enabled():
            user_id = get_or_create_user(
                conn,
                external_provider="local",
                external_subject="demo",
                email="demo@example.com",
            )
            conn.commit()
            return user_id
        if ext is None:
            if _wants_html_response(request):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=templates.TemplateResponse(
                        request,
                        "sign_in_required.html",
                        {},
                        status_code=status.HTTP_401_UNAUTHORIZED,
                    ),
                )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
            )
        user_id = get_or_create_user(
            conn,
            external_provider=ext.provider,
            external_subject=ext.subject,
            email=ext.email,
        )
        conn.commit()
        return user_id
    finally:
        conn.close()


CurrentUserId = Annotated[int, Depends(get_current_user_id)]
