"""
Compact, per-application memory updates.

We keep only the last N chat messages for continuity, and persist a small memory object
that helps the app resume a session without replaying the full chat history.
"""

from __future__ import annotations


def _get_openai_client():
    import os

    from career_copilot.database.db import load_env

    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env to update memory summaries.")
    from openai import OpenAI

    return OpenAI()


SUMMARY_SYSTEM = """You update a compact application memory summary.

Goal: keep a short summary of the user's decisions and the current state.

Rules:
- Keep it short (3-8 bullets).
- Focus on stable facts and decisions (e.g. location/address, target role, resume changes agreed, interview type).
- Do not include private secrets.
- Do not quote long text. No more than ~800 characters total.
"""


def should_refresh_summary(history: list[dict[str, str]], every_user_turns: int = 4) -> bool:
    """Refresh periodically based on number of user messages."""
    user_turns = sum(1 for h in (history or []) if (h or {}).get("role") == "user")
    return user_turns > 0 and user_turns % max(1, every_user_turns) == 0


def update_memory_summary(
    *,
    prev_summary: str,
    stage: str,
    recent_history: list[dict[str, str]],
) -> str:
    """
    Use the LLM to update a compact summary given the previous summary and
    the most recent (already-trimmed) chat history.
    """
    client = _get_openai_client()
    stage_label = "Resume improvement" if stage == "resume_improvement" else "Interview preparation"
    history_text_parts: list[str] = []
    for h in recent_history[-20:]:
        role = (h or {}).get("role") or ""
        content = ((h or {}).get("content") or "").strip()
        if role in ("user", "assistant") and content:
            history_text_parts.append(f"{role.upper()}: {content}")
    history_text = "\n".join(history_text_parts)[:6000]

    user_prompt = f"""Stage: {stage_label}

Previous summary (may be empty):
{prev_summary or "(none)"}

Recent chat:
{history_text or "(none)"}

Return the updated summary as bullet points."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def extract_interview_type_guess(text: str) -> str | None:
    """Very lightweight heuristic for interview type (no LLM call)."""
    t = (text or "").lower()
    if not t.strip():
        return None
    mapping = {
        "recruiter": ["recruiter", "screen", "introduction"],
        "technical": ["technical"],
        "coding": ["coding", "leetcode", "algorithm", "data structures"],
        "system_design": ["system design", "architecture", "scalability"],
        "behavioral": ["behavioural", "behavioral", "hr", "star"],
        "final": ["final", "negotiation", "offer"],
    }
    for k, keys in mapping.items():
        if any(x in t for x in keys):
            return k
    return None

