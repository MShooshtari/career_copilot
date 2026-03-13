"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src is on path so career_copilot is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config: pytest.Config) -> None:
    """Mark test runs so app startup skips DB init (no Postgres in CI / optional locally)."""
    import os

    os.environ["TESTING"] = "1"
