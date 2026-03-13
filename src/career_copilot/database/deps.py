"""FastAPI dependencies for database access."""

from __future__ import annotations

import psycopg

from career_copilot.database.db import connect


def get_db() -> psycopg.Connection:
    """Yield a database connection for request scope (caller should close when done)."""
    return connect()
