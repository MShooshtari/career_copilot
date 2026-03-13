"""Shared app configuration (paths, templates) for the web app and routers."""
from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

# Project root (app_config is in src/career_copilot/)
ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = ROOT / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
