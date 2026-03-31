from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from career_copilot.auth.config import auth_enabled
from career_copilot.auth.entra import ExternalIdentity, validate_bearer_jwt


def _extract_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    if not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip() or None


async def get_external_identity(request: Request) -> ExternalIdentity | None:
    """
    Returns an authenticated external identity if present.

    Sources:
    - Bearer JWT (Authorization header)
    - Session (set by /auth/login flow)
    """
    # Session-based (interactive login)
    sess = request.session if "session" in request.scope else None
    if isinstance(sess, dict) and sess.get("ext_identity"):
        ext = sess.get("ext_identity")
        if isinstance(ext, dict) and ext.get("provider") and ext.get("subject"):
            return ExternalIdentity(
                provider=str(ext["provider"]),
                subject=str(ext["subject"]),
                email=str(ext["email"]) if ext.get("email") else None,
                claims=dict(ext.get("claims") or {}),
            )

    # Bearer token (API client)
    token = _extract_bearer_token(request)
    if token:
        try:
            return validate_bearer_jwt(token)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e)
            ) from e

    return None


async def require_external_identity(
    ext: Annotated[ExternalIdentity | None, Depends(get_external_identity)],
) -> ExternalIdentity:
    if not auth_enabled():
        # When auth is disabled, routes should rely on DEFAULT_USER_ID via get_current_user_id.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth disabled")
    if ext is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return ext

