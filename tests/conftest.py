"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is on path so career_copilot is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _mock_db_at_startup():
    """Avoid real DB connection during app startup in tests (CI has no Postgres)."""
    mock_conn = MagicMock()
    # Patch get_db so startup never calls connect() — reliable across import order / CI.
    with patch("career_copilot.web_app.get_db", return_value=mock_conn):
        yield
