from __future__ import annotations

from uuid import uuid4

GUEST_PROVIDER = "guest"
GUEST_SESSION_KEY = "guest_subject"


def new_guest_subject() -> str:
    return uuid4().hex


def get_guest_subject(session: object) -> str | None:
    if not isinstance(session, dict):
        return None
    subject = session.get(GUEST_SESSION_KEY)
    if not isinstance(subject, str):
        return None
    subject = subject.strip()
    return subject or None
