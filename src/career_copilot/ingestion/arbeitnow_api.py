"""Arbeitnow job board API (Europe, ATS-backed). No API key required."""
from __future__ import annotations

from typing import Any

import requests

from career_copilot.ingestion.common import NormalizedJob, parse_datetime

ARBEITNOW_API_URL = "https://www.arbeitnow.com/api/job-board-api"


def normalize_arbeitnow_job(raw: dict[str, Any]) -> NormalizedJob:
    slug = raw.get("slug")
    return NormalizedJob(
        source="arbeitnow",
        source_id=slug if isinstance(slug, str) else None,
        title=raw.get("title"),
        company=raw.get("company_name"),
        location=raw.get("location"),
        salary_min=None,
        salary_max=None,
        description=raw.get("description"),
        skills=raw.get("tags") if isinstance(raw.get("tags"), list) else None,
        posted_at=parse_datetime(raw.get("published_at") or raw.get("posted") or raw.get("created_at")),
        url=raw.get("url"),
        raw=raw,
    )


def fetch_arbeitnow_jobs(timeout_s: int = 60) -> list[dict[str, Any]]:
    resp = requests.get(
        ARBEITNOW_API_URL,
        headers={"User-Agent": "career_copilot/0.1 (local ingestion)"},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    items = data.get("data") if isinstance(data, dict) else []
    return [j for j in items if isinstance(j, dict)]
