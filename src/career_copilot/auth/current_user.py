from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from career_copilot.auth.config import auth_enabled
from career_copilot.auth.deps import get_external_identity
from career_copilot.database.deps import get_db
from career_copilot.database.users import get_or_create_user


async def get_current_user_id(
    request: Request,
    ext=Depends(get_external_identity),
) -> int:
    """
    Resolve the current request's user_id.

    - When AUTH is enabled: requires an authenticated external identity and maps it to an internal user id.
    - When AUTH is disabled: returns DEFAULT_USER_ID (local demo mode), and ensures a local user row exists.
    """
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
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
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

