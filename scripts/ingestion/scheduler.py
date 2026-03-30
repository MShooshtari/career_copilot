"""
Run ingestion on a schedule (e.g. every 5 minutes or every hour).
Duplicates are avoided: the ingestion script upserts by (source, source_id).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for d in (SCRIPTS_DIR, SRC_DIR):
    s = str(d)
    if s not in sys.path:
        sys.path.insert(0, s)

from apscheduler.schedulers.blocking import BlockingScheduler  # noqa: E402
from ingestion.run import main as run_ingestion  # noqa: E402

# Run every 5 minutes; change to hours=1 for hourly
scheduler = BlockingScheduler()
# scheduler.add_job(run_ingestion, "interval", minutes=5)
scheduler.add_job(run_ingestion, "interval", hours=6)

if __name__ == "__main__":
    print("Ingestion scheduler started (every 6 hours). Ctrl+C to stop.")
    scheduler.start()
