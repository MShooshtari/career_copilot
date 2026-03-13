"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Skip app DB init when running tests (no Postgres in CI). Set before any app import.
os.environ["TESTING"] = "1"

# Ensure src is on path so career_copilot is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
