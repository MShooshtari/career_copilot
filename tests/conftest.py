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
def _mock_db_connect_at_startup():
    """Avoid real DB connection and getpass during app startup in tests (no TTY in CI)."""
    mock_conn = MagicMock()
    # Patch where connect is used (web_app.get_db calls this), not where it's defined.
    with patch("career_copilot.web_app.connect", return_value=mock_conn):
        yield
