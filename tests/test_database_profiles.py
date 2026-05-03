"""Tests for profile skill storage helpers."""

from __future__ import annotations

from career_copilot.database.profiles import replace_user_skills


class _Cursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...] | dict | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, sql: str, params=None) -> None:
        self.calls.append((sql, params))


class _Conn:
    def __init__(self) -> None:
        self.cur = _Cursor()

    def cursor(self):
        return self.cur


def test_replace_user_skills_stores_ai_skills_on_first_manual_skill_row() -> None:
    conn = _Conn()

    replace_user_skills(
        conn,
        user_id=42,
        skill_tags="Python, SQL, python",
        ai_extracted_skills=["FastAPI", "SQL", "fastapi"],
    )

    assert conn.cur.calls[0][1] == (42,)
    insert_params = [params for sql, params in conn.cur.calls if "INSERT INTO user_skills" in sql]
    assert insert_params == [
        (42, "Python", ["FastAPI", "SQL"]),
        (42, "SQL", None),
    ]


def test_replace_user_skills_keeps_ai_skills_when_manual_skills_empty() -> None:
    conn = _Conn()

    replace_user_skills(
        conn,
        user_id=42,
        skill_tags="",
        ai_extracted_skills=["Python", "SQL"],
    )

    insert_params = [params for sql, params in conn.cur.calls if "INSERT INTO user_skills" in sql]
    assert insert_params == [(42, "", ["Python", "SQL"])]
