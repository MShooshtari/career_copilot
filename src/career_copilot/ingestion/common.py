"""Shared types and helpers for job ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


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


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
