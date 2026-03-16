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


def _extract_jsonld_job(html: str) -> dict[str, Any] | None:
    """Extract job fields from JSON-LD JobPosting if present."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for script in soup.find_all("script", type="application/ld+json"):
            if not script.string:
                continue
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    continue
                for item in data:
                    if isinstance(item, dict):
                        t = item.get("@type") or item.get("type")
                        if t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t):
                            out: dict[str, Any] = {}
                            if item.get("title"):
                                out["title"] = item["title"]
                            if item.get("hiringOrganization", {}).get("name"):
                                out["company"] = item["hiringOrganization"]["name"]
                            if item.get("jobLocation"):
                                loc = item["jobLocation"]
                                if isinstance(loc, dict) and loc.get("address", {}).get("addressLocality"):
                                    out["location"] = loc["address"].get("addressLocality") or ""
                                elif isinstance(loc, str):
                                    out["location"] = loc
                            if item.get("description"):
                                out["description"] = item["description"]
                            if out:
                                return out
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    except ImportError:
        pass
    return None


def _extract_workday_like_json(html: str) -> dict[str, Any] | None:
    """Try to find job title/description in Workday-style or other embedded JSON."""
    # Workday and similar ATS often embed a large JSON blob in a script tag
    # Look for patterns like "title" : "..." or "jobTitle" or "headline"
    found: dict[str, Any] = {}
    # Match script content that might be JSON (avoid inline event handlers)
    for match in re.finditer(r"<script[^>]*>([\s\S]*?)</script>", html, re.IGNORECASE):
        blob = match.group(1)
        if len(blob) < 100 or "job" not in blob.lower():
            continue
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            # Try to find "title":"..." or "jobTitle":"..." in raw string
            for name in ("title", "jobTitle", "headline", "positionTitle"):
                m = re.search(rf'["\']?{name}["\']?\s*:\s*["\']([^"\']{{3,200}})["\']', blob, re.IGNORECASE)
                if m:
                    found["title"] = m.group(1).strip()
                    break
            if "description" in blob.lower():
                m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', blob)
                if m:
                    desc = m.group(1).encode().decode("unicode_escape")
                    if len(desc) > 50:
                        found["description"] = desc[:8000]
            if found:
                return found
            continue
        if not isinstance(data, dict):
            continue
        # Recurse into nested objects (e.g. initial state)
        def dig(d: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(d, dict):
                for k, v in d.items():
                    klo = k.lower()
                    if klo in ("title", "jobtitle", "headline") and isinstance(v, str) and len(v) > 2:
                        found.setdefault("title", v)
                    if klo == "description" and isinstance(v, str) and len(v) > 50:
                        found.setdefault("description", v[:8000])
                    if klo == "company" and isinstance(v, str):
                        found.setdefault("company", v)
                    dig(v, depth + 1)
            elif isinstance(d, list):
                for x in d[:3]:
                    dig(x, depth + 1)

        dig(data, 0)
        if found:
            return found
    return found if found else None


def _title_from_url_path(url: str) -> str:
    """Derive a readable job title from URL path (e.g. Senior-Data-Scientist-II -> Senior Data Scientist II)."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = (parsed.path or "").strip("/")
    if not path:
        return ""
    # Take last meaningful segment (often job slug)
    segments = [s for s in path.split("/") if s and not s.startswith("en-")]
    if not segments:
        return ""
    slug = segments[-1]
    # Remove common suffixes like _REQ-6008-1 or -REQ-6008
    slug = re.sub(r"_?REQ[-_]?\d+[-_]?\d*$", "", slug, flags=re.IGNORECASE)
    slug = re.sub(r"[-_]?\d+$", "", slug)
    # Replace hyphens/underscores with spaces and title-case
    slug = slug.replace("-", " ").replace("_", " ").strip()
    if not slug:
        return ""
    return slug.title()


def _html_to_text(html: str) -> str:
    """Extract main text from HTML (strip scripts, styles, nav, etc.)."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except ImportError:
        text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
        text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def _fetch_page_html(url: str) -> str:
    """Fetch URL and return raw HTML (for JSON-LD / embedded JSON extraction)."""
    import httpx

    try:
        with httpx.Client(follow_redirects=True, timeout=15.0) as client:
            resp = client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            resp.raise_for_status()
            return resp.text
    except Exception:
        return ""


def extract_job_from_url(url: str) -> dict[str, Any]:
    """
    Fetch job page (Indeed, LinkedIn, Workday, company site, etc.) and extract job details.
    Tries JSON-LD JobPosting, then embedded JSON (e.g. Workday), then HTML text + LLM.
    If the page is JS-heavy and empty, derives title from URL path so we don't save "Untitled Job".
    """
    if not (url or "").strip():
        return {}
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    html = _fetch_page_html(url)
    # 1) Try JSON-LD JobPosting
    jsonld = _extract_jsonld_job(html)
    if jsonld:
        result = {
            "title": jsonld.get("title"),
            "company": jsonld.get("company"),
            "location": jsonld.get("location"),
            "salary_min": None,
            "salary_max": None,
            "description": jsonld.get("description"),
            "skills": _coerce_skills(jsonld.get("skills")) if jsonld.get("skills") else [],
            "url": url,
        }
        if not result.get("title"):
            result["title"] = _title_from_url_path(url)
        return result

    # 2) Try Workday-style or other embedded JSON
    embedded = _extract_workday_like_json(html)
    if embedded:
        result = {
            "title": embedded.get("title"),
            "company": embedded.get("company"),
            "location": None,
            "salary_min": None,
            "salary_max": None,
            "description": embedded.get("description"),
            "skills": [],
            "url": url,
        }
        if result.get("title") or result.get("description"):
            if not result.get("title"):
                result["title"] = _title_from_url_path(url)
            return result

    # 3) Main text from same HTML + LLM
    text = _html_to_text(html)
    url_hint = f"Source URL: {url}"
    if not text or len(text.strip()) < 200:
        # JS-heavy or login wall: derive title from URL so we don't get "Untitled Job"
        title_from_url = _title_from_url_path(url)
        if title_from_url:
            url_hint += f"\nThe page content could not be read (e.g. JavaScript-rendered or login required). Use this title derived from the URL: {title_from_url}. Set company/location/description to null if unknown."
        text = text.strip() or title_from_url or "Job posting (content not available from URL)."
    result = _extract_with_llm(text, url_hint)
    if not result.get("url"):
        result["url"] = url
    if not (result.get("title") or "").strip():
        result["title"] = _title_from_url_path(url) or "Job from link"
    return result
