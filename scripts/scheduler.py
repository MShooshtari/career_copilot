"""
Run ingestion on a schedule (e.g. every 5 minutes or every hour).
Duplicates are avoided: the ingestion script upserts by (source, source_id).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import after path setup so run_ingestion can load career_copilot
import run_ingestion  # noqa: E402
from apscheduler.schedulers.blocking import BlockingScheduler  # noqa: E402

# Run every 5 minutes; change to hours=1 for hourly
scheduler = BlockingScheduler()
scheduler.add_job(run_ingestion.main, "interval", minutes=5)
# scheduler.add_job(run_ingestion.main, "interval", hours=1)

if __name__ == "__main__":
    print("Ingestion scheduler started (every 5 minutes). Ctrl+C to stop.")
    scheduler.start()
