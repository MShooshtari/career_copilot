"""Adzuna job search API. Requires free app_id and app_key from https://developer.adzuna.com/signup."""
from __future__ import annotations

import os
from typing import Any

import requests

from career_copilot.ingestion.common import NormalizedJob, coerce_int, parse_datetime

ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs"
# Countries: gb, us, au, de, fr, etc. Paginate with results_per_page=50.
DEFAULT_COUNTRIES = ("gb", "us")
RESULTS_PER_PAGE = 50
MAX_PAGES_PER_COUNTRY = 5


def _get_credentials() -> tuple[str, str] | None:
    app_id = (os.environ.get("ADZUNA_APP_ID") or os.environ.get("adzuna_app_id") or "").strip()
    app_key = (os.environ.get("ADZUNA_APP_KEY") or os.environ.get("adzuna_app_key") or "").strip()
    if app_id and app_key:
        return (app_id, app_key)
    return None


def normalize_adzuna_job(raw: dict[str, Any]) -> NormalizedJob:
    country = raw.get("_country", "")
    job_id = raw.get("id")
    company = raw.get("company")
    if isinstance(company, dict):
        company = company.get("display_name")
    loc = raw.get("location")
    if isinstance(loc, dict):
        loc = loc.get("display_name")
    return NormalizedJob(
        source="adzuna",
        source_id=f"{country}_{job_id}" if job_id is not None else None,
        title=raw.get("title"),
        company=company,
        location=loc,
        salary_min=coerce_int(raw.get("salary_min")),
        salary_max=coerce_int(raw.get("salary_max")),
        description=raw.get("description"),
        skills=None,
        posted_at=parse_datetime(raw.get("created")),
        url=raw.get("redirect_url"),
        raw={**raw, "_country": country},
    )


def fetch_adzuna_jobs(
    countries: tuple[str, ...] = DEFAULT_COUNTRIES,
    results_per_page: int = RESULTS_PER_PAGE,
    max_pages_per_country: int = MAX_PAGES_PER_COUNTRY,
    timeout_s: int = 30,
) -> list[dict[str, Any]]:
    creds = _get_credentials()
    if not creds:
        return []
    app_id, app_key = creds
    all_jobs: list[dict[str, Any]] = []
    for country in countries:
        for page in range(1, max_pages_per_country + 1):
            url = f"{ADZUNA_BASE}/{country}/search/{page}"
            params = {
                "app_id": app_id,
                "app_key": app_key,
                "results_per_page": results_per_page,
                "content-type": "application/json",
            }
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": "career_copilot/0.1 (local ingestion)"},
                timeout=timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results") if isinstance(data, dict) else []
            if not results:
                break
            for r in results:
                if isinstance(r, dict):
                    r["_country"] = country
                    all_jobs.append(r)
    return all_jobs
