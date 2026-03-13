"""Shared utilities used across the application."""

from __future__ import annotations


def strip_nul(s: str) -> str:
    """Remove NUL bytes; PostgreSQL TEXT cannot store them."""
    return s.replace("\x00", "")
