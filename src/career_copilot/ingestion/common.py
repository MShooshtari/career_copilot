"""Shared types and helpers for job ingestion."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _fix_mojibake(s: str) -> str:
    """
    Repair UTF-8 text that was mis-decoded as Latin-1/Windows-1252 (e.g. Weâre -> We're).
    Uses a best-effort round-trip; if it fails, returns the original string.
    """
    # Common mojibake comes from UTF-8 bytes interpreted as single-byte encodings.
    # We try a latin1 round-trip but ignore characters that can't be encoded there
    # so the fix still applies to the rest of the string.
    try:
        repaired = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        # Only use the repaired version if it actually improved things (heuristic: fewer "â" artifacts).
        if "â" in s and s.count("â") > repaired.count("â"):
            return repaired
        return repaired if repaired else s
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


def _strip_html_definitive(s: str) -> str:
    """Remove all HTML tags and decode entities. Result is never left with raw tags."""
    # Replace block end tags with newline so we keep paragraph/list structure
    s = re.sub(r"</(?:p|div|br|li|h[1-6]|tr|ul|ol)\s*>", "\n", s, flags=re.IGNORECASE)
    # Remove all remaining tags (and anything that looks like a tag)
    s = re.sub(r"<[^>]+>", " ", s)
    # Decode HTML entities (&#x26; -> &, &amp; -> &, etc.)
    s = html.unescape(s)
    return s


def html_to_plain_text(value: str | None) -> str | None:
    """
    Convert HTML description to human-readable plain text for storage.
    Strips all tags, unescapes entities, fixes common mojibake, and normalizes whitespace.
    Returns None for None or empty input.
    """
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    # If it doesn't look like HTML, only fix mojibake and return
    if "<" not in s and ">" not in s:
        s = _fix_mojibake(s)
        return s
    # Definitive strip: no tags can remain in output
    text = _strip_html_definitive(s)
    # Fix mojibake (e.g. Weâre -> We're)
    text = _fix_mojibake(text)
    # Normalize whitespace: collapse runs, strip each line, collapse blank lines
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text if text else None


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
