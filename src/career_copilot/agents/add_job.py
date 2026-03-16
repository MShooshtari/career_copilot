"""
Add Job Agent: extract structured job details from manual text, uploaded file, or URL.

Supports:
- Manual: user-provided text + optional hints (location, salary, etc.)
- File: PDF, TXT, or Word (.docx) — extract text then LLM extraction
- URL: fetch page content (Indeed, LinkedIn, company site) then LLM extraction
"""

from __future__ import annotations

import json
import re
from typing import Any

EXTRACTION_SYSTEM = """You are a job listing parser. Extract structured fields from the given job description text.
Output a single JSON object with these keys (use null for missing):
- title: string (job title)
- company: string (company name)
- location: string (e.g. "Remote", "New York, NY")
- salary_min: integer or null (annual salary minimum if mentioned)
- salary_max: integer or null (annual salary maximum if mentioned)
- description: string (full job description, preserve key requirements and responsibilities)
- skills: array of strings (mentioned skills, technologies, requirements; dedupe and normalize)
- url: string or null (source URL if present in the text)

Rules:
- For salary, convert to annual if given hourly or monthly; use integers only.
- Keep description substantive but trim excessive boilerplate.
- Skills: list concrete items (e.g. "Python", "AWS", "5+ years experience").
- If the text is not a job description, do your best to extract any job-related fields."""


def _get_openai_client():
    import os

    from career_copilot.database.db import load_env

    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env for the add-job agent.")
    from openai import OpenAI

    return OpenAI()


def _extract_with_llm(text: str, extra_hint: str = "") -> dict[str, Any]:
    """Use LLM to extract job fields from plain text. Returns dict with title, company, etc."""
    if not (text or text.strip()):
        return {}
    client = _get_openai_client()
    user_content = text.strip()[:30000]  # cap length
    if extra_hint:
        user_content = f"Additional hints from the user:\n{extra_hint}\n\n---\n\nJob description text:\n{user_content}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Try to parse JSON (handle markdown code block)
    if "```" in raw:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    # Normalize types
    out: dict[str, Any] = {
        "title": (data.get("title") or "").strip() or None,
        "company": (data.get("company") or "").strip() or None,
        "location": (data.get("location") or "").strip() or None,
        "salary_min": _coerce_int(data.get("salary_min")),
        "salary_max": _coerce_int(data.get("salary_max")),
        "description": (data.get("description") or "").strip() or None,
        "skills": _coerce_skills(data.get("skills")),
        "url": (data.get("url") or "").strip() or None,
    }
    return out


def _coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_skills(v: Any) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()]


def extract_job_from_text(
    text: str,
    *,
    location: str | None = None,
    salary_min: int | None = None,
    salary_max: int | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """
    Extract job details from manually entered text. Optional hints are passed to the LLM.
    Returns dict with keys: title, company, location, salary_min, salary_max, description, skills, url.
    """
    hints = []
    if location:
        hints.append(f"Location: {location}")
    if salary_min is not None or salary_max is not None:
        s = f"Salary: {salary_min or '?'} - {salary_max or '?'}"
        hints.append(s)
    if url:
        hints.append(f"Source URL: {url}")
    extra = "\n".join(hints) if hints else ""
    return _extract_with_llm(text, extra)


def _extract_text_from_file(content: bytes, filename: str) -> str:
    """Extract plain text from job description file (PDF, TXT, or DOCX)."""
    if not content:
        return ""
    fn = (filename or "").lower()
    # PDF
    if fn.endswith(".pdf") or content.startswith(b"%PDF"):
        from career_copilot.resume_io import extract_resume_text

        return extract_resume_text(content, filename or "job.pdf")
    # DOCX
    if fn.endswith(".docx") or fn.endswith(".doc"):
        try:
            from io import BytesIO

            import docx

            doc = docx.Document(BytesIO(content))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(parts) if parts else ""
        except Exception:
            pass
    # TXT or fallback
    try:
        return content.decode("utf-8", errors="replace")
    except Exception:
        return ""


def extract_job_from_file(content: bytes, filename: str) -> dict[str, Any]:
    """
    Extract job details from an uploaded file (PDF, TXT, or Word).
    Returns dict with keys: title, company, location, salary_min, salary_max, description, skills, url.
    """
    text = _extract_text_from_file(content, filename)
    if not text.strip():
        return {"title": None, "company": None, "location": None, "salary_min": None,
                "salary_max": None, "description": None, "skills": [], "url": None}
    return _extract_with_llm(text)


def _fetch_page_text(url: str) -> str:
    """Fetch URL and return main text content (strip HTML)."""
    import httpx

    try:
        with httpx.Client(follow_redirects=True, timeout=15.0) as client:
            resp = client.get(url, headers={"User-Agent": "CareerCopilot/1.0 (Job parser)"})
            resp.raise_for_status()
            html = resp.text
    except Exception:
        return ""
    # Prefer BeautifulSoup if available for cleaner text
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except ImportError:
        # Fallback: crude tag strip
        text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
        text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def extract_job_from_url(url: str) -> dict[str, Any]:
    """
    Fetch job page (Indeed, LinkedIn, company site, etc.) and extract job details.
    Returns dict with keys: title, company, location, salary_min, salary_max, description, skills, url.
    """
    if not (url or "").strip():
        return {}
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    text = _fetch_page_text(url)
    if not text.strip():
        return {"title": None, "company": None, "location": None, "salary_min": None,
                "salary_max": None, "description": None, "skills": [], "url": url}
    # Pass URL so LLM can set it in output
    result = _extract_with_llm(text, f"Source URL: {url}")
    if not result.get("url"):
        result["url"] = url
    return result
