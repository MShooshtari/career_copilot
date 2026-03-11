from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests


REMOTEOK_API_URL = "https://remoteok.com/api"


@dataclass(frozen=True)
class NormalizedJob:
    source: str
    source_id: str | None
    title: str | None
    company: str | None
    location: str | None
    salary_min: int | None
    salary_max: int | None
    description: str | None
    skills: list[str] | None
    posted_at: datetime | None
    url: str | None
    raw: dict[str, Any]


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        # Some APIs send unix seconds.
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        # RemoteOK commonly sends ISO timestamps (e.g. "2024-01-01T00:00:00+00:00").
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_remoteok_job(raw: dict[str, Any]) -> NormalizedJob:
    return NormalizedJob(
        source="remoteok",
        source_id=str(raw.get("id")) if raw.get("id") is not None else None,
        title=raw.get("position") or raw.get("title"),
        company=raw.get("company"),
        location=raw.get("location"),
        salary_min=_coerce_int(raw.get("salary_min")),
        salary_max=_coerce_int(raw.get("salary_max")),
        description=raw.get("description"),
        skills=list(raw.get("tags") or []) if raw.get("tags") is not None else None,
        posted_at=_parse_datetime(raw.get("date") or raw.get("posted_at")),
        url=raw.get("url"),
        raw=raw,
    )


def fetch_remoteok_jobs(timeout_s: int = 30) -> list[dict[str, Any]]:
    resp = requests.get(
        REMOTEOK_API_URL,
        headers={"User-Agent": "career_copilot/0.1 (local ingestion)"},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()

    # RemoteOK returns a metadata object at index 0; jobs start at index 1.
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[1:]

    return [x for x in data if isinstance(x, dict)]

