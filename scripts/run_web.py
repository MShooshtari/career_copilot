"""
Run the Career Copilot web app (user profile page).

Usage (from project root):
  python scripts/run_web.py

Ensures the `src` directory is on PYTHONPATH so that `career_copilot` is
importable both in the main process and in the uvicorn reload subprocess.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# So the uvicorn reload subprocess can find career_copilot
existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR)] + ([existing] if existing else []))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    os.chdir(ROOT)
    uvicorn.run(
        "career_copilot.web_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
