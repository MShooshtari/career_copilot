"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Skip app DB init when running tests (no Postgres in CI). Set before any app import.
os.environ["TESTING"] = "1"

# Ensure src is on path so career_copilot is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Patch DB connect so app startup and route handlers never touch a real DB.
# Mock connection: cursor().fetchone() returns None so "not found" flows work in tests.
_mock_conn = MagicMock()
_mock_cur = MagicMock()
_mock_cur.fetchone.return_value = None
_mock_conn.cursor.return_value.__enter__ = lambda self: _mock_cur
_mock_conn.cursor.return_value.__exit__ = lambda *a: None
_connect_patch = patch(
    "career_copilot.database.db.connect",
    return_value=_mock_conn,
)
_connect_patch.start()
