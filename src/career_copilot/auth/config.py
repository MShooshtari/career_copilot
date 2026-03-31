from __future__ import annotations

import os


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def auth_enabled() -> bool:
    return (_env("AUTH_ENABLED", "0") or "0").strip() in (
        "1",
        "true",
        "True",
        "yes",
        "YES",
    )


def session_secret_key() -> str:
    # Required when AUTH_ENABLED=1 (SessionMiddleware uses it).
    return (_env("SESSION_SECRET_KEY", "") or "").strip()


def entra_client_id() -> str:
    return (_env("ENTRA_CLIENT_ID", "") or "").strip()


def entra_client_secret() -> str:
    return (_env("ENTRA_CLIENT_SECRET", "") or "").strip()


def entra_redirect_uri() -> str:
    return (_env("ENTRA_REDIRECT_URI", "") or "").strip()


def entra_metadata_url() -> str | None:
    v = (_env("ENTRA_METADATA_URL") or "").strip()
    return v or None


def entra_authority() -> str | None:
    v = (_env("ENTRA_AUTHORITY") or "").strip()
    return v or None


def entra_tenant_domain() -> str:
    return (_env("ENTRA_TENANT_DOMAIN", "") or "").strip()


def entra_provider_name() -> str:
    return "entra_external_id"

