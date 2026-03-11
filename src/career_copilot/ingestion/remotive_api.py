"""Remotive remote jobs API. No API key required. Rate limit: max ~4 requests/day recommended."""
from __future__ import annotations

from typing import Any

import requests

from career_copilot.ingestion.common import NormalizedJob, parse_datetime

REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"


def normalize_remotive_job(raw: dict[str, Any]) -> NormalizedJob:
    return NormalizedJob(
        source="remotive",
        source_id=str(raw["id"]) if raw.get("id") is not None else None,
        title=raw.get("title"),
        company=raw.get("company_name"),
        location=raw.get("candidate_required_location"),
        salary_min=None,
        salary_max=None,
        description=raw.get("description"),
        skills=raw.get("tags") if isinstance(raw.get("tags"), list) else None,
        posted_at=parse_datetime(raw.get("publication_date")),
        url=raw.get("url"),
        raw=raw,
    )


def fetch_remotive_jobs(timeout_s: int = 30) -> list[dict[str, Any]]:
    resp = requests.get(
        REMOTIVE_API_URL,
        headers={"User-Agent": "career_copilot/0.1 (local ingestion)"},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    jobs = data.get("jobs") if isinstance(data, dict) else []
    return [j for j in jobs if isinstance(j, dict)]
