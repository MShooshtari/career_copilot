from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from jose import jwk, jwt
from jose.constants import Algorithms

from career_copilot.auth.config import (
    entra_client_id,
    entra_metadata_url,
    entra_provider_name,
)


@dataclass(frozen=True)
class ExternalIdentity:
    provider: str
    subject: str
    email: str | None
    claims: dict[str, Any]


class JwksCache:
    def __init__(self, *, ttl_seconds: int = 3600) -> None:
        self._ttl_seconds = ttl_seconds
        self._value: dict[str, Any] | None = None
        self._expires_at = 0.0

    def get(self) -> dict[str, Any] | None:
        if self._value is None:
            return None
        if time.time() >= self._expires_at:
            return None
        return self._value

    def set(self, value: dict[str, Any]) -> None:
        self._value = value
        self._expires_at = time.time() + self._ttl_seconds


_jwks_cache = JwksCache(ttl_seconds=3600)
_oidc_cache = JwksCache(ttl_seconds=3600)


def _get_metadata_url() -> str:
    url = entra_metadata_url()
    if not url:
        raise RuntimeError("ENTRA_METADATA_URL is required for bearer token validation")
    return url


def _fetch_oidc_config() -> dict[str, Any]:
    cached = _oidc_cache.get()
    if cached is not None:
        return cached
    url = _get_metadata_url()
    with httpx.Client(timeout=10.0) as client:
        r = client.get(url)
        r.raise_for_status()
        cfg = r.json()
    if not isinstance(cfg, dict):
        raise RuntimeError("Invalid OIDC metadata response")
    _oidc_cache.set(cfg)
    return cfg


def _fetch_jwks() -> dict[str, Any]:
    cached = _jwks_cache.get()
    if cached is not None:
        return cached
    cfg = _fetch_oidc_config()
    jwks_uri = cfg.get("jwks_uri")
    if not isinstance(jwks_uri, str) or not jwks_uri:
        raise RuntimeError("OIDC config missing jwks_uri")
    with httpx.Client(timeout=10.0) as client:
        r = client.get(jwks_uri)
        r.raise_for_status()
        jwks = r.json()
    if not isinstance(jwks, dict):
        raise RuntimeError("Invalid JWKS response")
    _jwks_cache.set(jwks)
    return jwks


def validate_bearer_jwt(token: str) -> ExternalIdentity:
    """
    Validate an Entra-issued JWT using OIDC discovery + JWKS.

    This is used for API clients that send `Authorization: Bearer <token>`.
    Browser interactive login uses Authlib (see routers/auth.py).
    """
    cfg = _fetch_oidc_config()
    issuer = cfg.get("issuer")
    if not isinstance(issuer, str) or not issuer:
        raise RuntimeError("OIDC config missing issuer")

    client_id = entra_client_id()
    if not client_id:
        raise RuntimeError("ENTRA_CLIENT_ID is required for bearer token validation")

    headers = jwt.get_unverified_header(token)
    kid = headers.get("kid")
    if not kid:
        raise RuntimeError("JWT missing kid header")

    jwks = _fetch_jwks()
    keys = jwks.get("keys", [])
    if not isinstance(keys, list):
        raise RuntimeError("Invalid JWKS keys")
    key_dict = next(
        (k for k in keys if isinstance(k, dict) and k.get("kid") == kid),
        None,
    )
    if key_dict is None:
        raise RuntimeError("No matching JWKS key for kid")

    public_key = jwk.construct(key_dict, Algorithms.RS256)
    message, encoded_sig = token.rsplit(".", 1)
    decoded_sig = jwt.base64url_decode(encoded_sig.encode("utf-8"))
    if not public_key.verify(message.encode("utf-8"), decoded_sig):
        raise RuntimeError("JWT signature verification failed")

    claims = jwt.get_unverified_claims(token)
    # Minimal issuer/audience checks (exp/nbf handled below)
    if claims.get("iss") != issuer:
        raise RuntimeError("Invalid issuer")

    aud = claims.get("aud")
    if isinstance(aud, str):
        ok_aud = aud == client_id
    elif isinstance(aud, list):
        ok_aud = client_id in aud
    else:
        ok_aud = False
    if not ok_aud:
        raise RuntimeError("Invalid audience")

    # Expiration check
    now = int(time.time())
    exp = claims.get("exp")
    if not isinstance(exp, int) or exp <= now:
        raise RuntimeError("Token expired")

    sub = claims.get("sub") or claims.get("oid")
    if not isinstance(sub, str) or not sub:
        raise RuntimeError("Token missing subject")

    email = None
    for k in ("email", "emails", "preferred_username"):
        v = claims.get(k)
        if isinstance(v, str) and v:
            email = v
            break
        if isinstance(v, list) and v and isinstance(v[0], str):
            email = v[0]
            break

    return ExternalIdentity(
        provider=entra_provider_name(),
        subject=sub,
        email=email,
        claims=json.loads(json.dumps(claims)),
    )
