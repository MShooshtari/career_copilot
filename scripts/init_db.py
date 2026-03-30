from __future__ import annotations

import sys
from pathlib import Path

# Ensure `career_copilot` package is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from career_copilot.database.db import connect  # noqa: E402
from career_copilot.database.schema import init_schema  # noqa: E402

if __name__ == "__main__":
    try:
        conn = connect()
        try:
            init_schema(conn)
        finally:
            conn.close()
    except Exception:
        # No DB available or connection failed (e.g. CI); continue without schema init
        pass