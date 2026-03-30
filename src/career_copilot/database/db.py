from __future__ import annotations

import os
import sys
from getpass import getpass
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.conninfo import make_conninfo


def load_env() -> None:
    """
    Load environment variables from `.env` if present.
    """
    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env")


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def get_connection_kwargs(*, dbname: str | None = None) -> dict:
    """
    Connection configuration, supporting either:
    - POSTGRES_DSN, or
    - discrete POSTGRES_* env vars (with interactive password prompt if missing)
    """
    sslmode = _env("POSTGRES_SSLMODE")

    dsn = _env("POSTGRES_DSN")
    if dsn:
        if sslmode:
            return {"conninfo": make_conninfo(dsn, sslmode=sslmode)}
        return {"conninfo": dsn}

    host = _env("POSTGRES_HOST", "localhost")
    port = int(_env("POSTGRES_PORT", "5432") or "5432")
    user = _env("POSTGRES_USER", "postgres")
    db = dbname or _env("POSTGRES_DB", "career_copilot")
    password = _env("POSTGRES_PASSWORD")
    if password is None and sys.stdin.isatty():
        password = getpass(f"Password for PostgreSQL user {user}: ")
    elif password is None:
        password = ""

    kw: dict = {"host": host, "port": port, "user": user, "password": password, "dbname": db}
    if sslmode:
        kw["sslmode"] = sslmode
    return kw


def connect(*, dbname: str | None = None) -> psycopg.Connection:
    load_env()
    kwargs = get_connection_kwargs(dbname=dbname)
    if "conninfo" in kwargs:
        return psycopg.connect(kwargs["conninfo"])
    return psycopg.connect(**kwargs)
