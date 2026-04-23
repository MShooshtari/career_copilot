from __future__ import annotations

from typing import Any

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from career_copilot.auth.config import (
    auth_enabled,
    entra_authority,
    entra_client_id,
    entra_client_secret,
    entra_metadata_url,
    entra_redirect_uri,
    entra_tenant_domain,
    entra_tenant_id,
)
from career_copilot.auth.entra import ExternalIdentity

router = APIRouter(prefix="/auth", tags=["auth"])


def _build_default_metadata_url() -> str:
    # External ID tenants expose standard OIDC discovery endpoints.
    # If you need a non-default endpoint, set ENTRA_METADATA_URL explicitly.
    tenant = entra_tenant_domain()
    if not tenant:
        raise RuntimeError("ENTRA_TENANT_DOMAIN is required for interactive login")

    authority = entra_authority()
    if authority:
        base = authority.strip().rstrip("/")
        if ".well-known/openid-configuration" in base:
            return base
    else:
        # For CIAM, tenant domains are often provided as either:
        # - <tenant>.ciamlogin.com
        # - <tenant>.onmicrosoft.com (directory domain; not the login host)
        #
        # We don't try to infer the correct host from onmicrosoft.com because it's ambiguous.
        # Prefer setting ENTRA_METADATA_URL explicitly, or set ENTRA_TENANT_DOMAIN to <tenant>.ciamlogin.com.
        base = tenant.strip().rstrip("/")
        if "://" not in base:
            base = f"https://{base}"

    tid = entra_tenant_id()
    if tid and ".well-known/openid-configuration" not in base:
        base = f"{base}/{tid}".rstrip("/")

    return f"{base}/v2.0/.well-known/openid-configuration"


def _oauth() -> OAuth:
    oauth = OAuth()
    metadata_url = entra_metadata_url() or _build_default_metadata_url()
    oauth.register(
        name="entra",
        client_id=entra_client_id(),
        client_secret=entra_client_secret(),
        server_metadata_url=metadata_url,
        client_kwargs={"scope": "openid profile email"},
    )
    return oauth


@router.get("/login")
async def login(request: Request) -> RedirectResponse:
    if not auth_enabled():
        return RedirectResponse(url="/profile", status_code=303)

    oauth = _oauth()
    redirect_uri = entra_redirect_uri()
    if not redirect_uri:
        raise RuntimeError("ENTRA_REDIRECT_URI is required for interactive login")
    return await oauth.entra.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def callback(request: Request) -> RedirectResponse:
    if not auth_enabled():
        return RedirectResponse(url="/profile", status_code=303)

    oauth = _oauth()
    token = await oauth.entra.authorize_access_token(request)
    userinfo: dict[str, Any] = {}
    try:
        userinfo = await oauth.entra.userinfo(token=token)
    except Exception:
        # Some configurations don't expose a userinfo endpoint; fall back to id_token claims.
        pass

    claims: dict[str, Any] = {}
    if "id_token" in token and isinstance(token["id_token"], str):
        try:
            # Authlib's OpenID helpers can be async depending on client type/config.
            parsed = await oauth.entra.parse_id_token(request, token)
            claims = dict(parsed or {})
        except Exception:
            claims = {}
    if userinfo:
        claims = {**claims, **dict(userinfo)}

    subject = str(claims.get("sub") or claims.get("oid") or "").strip()
    if not subject:
        raise RuntimeError("Login did not return a subject claim (sub/oid)")

    email = None
    for k in ("email", "preferred_username"):
        v = claims.get(k)
        if isinstance(v, str) and v:
            email = v
            break

    ext = ExternalIdentity(
        provider="entra_external_id",
        subject=subject,
        email=email,
        claims=claims,
    )
    request.session["ext_identity"] = {
        "provider": ext.provider,
        "subject": ext.subject,
        "email": ext.email,
        "claims": ext.claims,
    }
    return RedirectResponse(url="/profile", status_code=303)


@router.get("/logout")
async def logout(request: Request) -> RedirectResponse:
    if getattr(request, "session", None) is not None:
        request.session.pop("ext_identity", None)
    return RedirectResponse(url="/profile", status_code=303)
