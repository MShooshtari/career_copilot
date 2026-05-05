"""
Run Career Copilot AI evaluation benchmarks.

Usage (from project root):
  python scripts/run_evals.py --suite all

Ensures the `src` directory is on PYTHONPATH so `career_copilot` is importable
without requiring an editable install.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    os.chdir(ROOT)
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR)] + ([existing] if existing else []))
    runpy.run_module("career_copilot.evals.run", run_name="__main__")

