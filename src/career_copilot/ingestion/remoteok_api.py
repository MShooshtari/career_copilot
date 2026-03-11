from __future__ import annotations

from typing import Any

import requests

from career_copilot.ingestion.common import (
    NormalizedJob,
    coerce_int,
    html_to_plain_text,
    parse_datetime,
)

REMOTEOK_API_URL = "https://remoteok.com/api"


def normalize_remoteok_job(raw: dict[str, Any]) -> NormalizedJob:
    return NormalizedJob(
        source="remoteok",
        source_id=str(raw.get("id")) if raw.get("id") is not None else None,
        title=raw.get("position") or raw.get("title"),
        company=raw.get("company"),
        location=raw.get("location"),
        salary_min=coerce_int(raw.get("salary_min")),
        salary_max=coerce_int(raw.get("salary_max")),
        description=html_to_plain_text(raw.get("description")),
        skills=list(raw.get("tags") or []) if raw.get("tags") is not None else None,
        posted_at=parse_datetime(raw.get("date") or raw.get("posted_at")),
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

