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


def _parse_jsonld_location(loc: Any) -> str:
    """Turn JSON-LD jobLocation (object or list) into a single location string."""
    if not loc:
        return ""
    if isinstance(loc, str):
        return loc.strip()
    if isinstance(loc, list):
        parts = []
        for x in loc:
            p = _parse_jsonld_location(x)
            if p:
                parts.append(p)
        return ", ".join(parts) if parts else ""
    if isinstance(loc, dict):
        addr = loc.get("address") or loc
        if isinstance(addr, dict):
            locality = (addr.get("addressLocality") or "").strip()
            region = (addr.get("addressRegion") or "").strip()
            country = (addr.get("addressCountry") or "").strip()
            if isinstance(country, dict):
                country = (country.get("name") or "").strip()
            return ", ".join(x for x in (locality, region, country) if x)
        if isinstance(addr, str):
            return addr
    return ""


def _parse_jsonld_salary(item: dict) -> tuple[int | None, int | None]:
    """Extract salary_min, salary_max from JSON-LD baseSalary if present."""
    base = item.get("baseSalary")
    if not isinstance(base, dict):
        return (None, None)
    val = base.get("value")
    if not isinstance(val, dict):
        return (None, None)
    min_v = val.get("minValue") or val.get("value")
    max_v = val.get("maxValue") or val.get("value")
    unit = (val.get("unitText") or "").upper() or "YEAR"
    try:
        mn = int(min_v) if min_v is not None else None
        mx = int(max_v) if max_v is not None else None
    except (TypeError, ValueError):
        return (None, None)
    # Convert to annual if needed
    if "HOUR" in unit and mn is not None:
        mn = mn * 2080
    if "HOUR" in unit and mx is not None:
        mx = mx * 2080
    if "MONTH" in unit and mn is not None:
        mn = mn * 12
    if "MONTH" in unit and mx is not None:
        mx = mx * 12
    return (mn, mx)


def _extract_jsonld_job(html: str) -> dict[str, Any] | None:
    """Extract job fields from JSON-LD JobPosting if present (Indeed, Rippling, etc.)."""
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
                    if not isinstance(item, dict):
                        continue
                    t = item.get("@type") or item.get("type")
                    if t != "JobPosting" and not (isinstance(t, list) and "JobPosting" in t):
                        continue
                    out: dict[str, Any] = {}
                    if item.get("title"):
                        out["title"] = item["title"]
                    org = item.get("hiringOrganization")
                    if isinstance(org, dict) and org.get("name"):
                        out["company"] = org["name"]
                    elif isinstance(org, str):
                        out["company"] = org
                    loc = item.get("jobLocation")
                    if loc:
                        out["location"] = _parse_jsonld_location(loc)
                    if item.get("description"):
                        out["description"] = item["description"]
                    salary_min, salary_max = _parse_jsonld_salary(item)
                    if salary_min is not None:
                        out["salary_min"] = salary_min
                    if salary_max is not None:
                        out["salary_max"] = salary_max
                    if out:
                        return out
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    except ImportError:
        pass
    return None


def _extract_indeed_job(html: str) -> dict[str, Any] | None:
    """Extract job fields from Indeed viewjob page (script/JSON or HTML)."""
    found: dict[str, Any] = {}
    # Multiple patterns: Indeed and ca.indeed.com use various key names
    patterns = [
        (r'"title"\s*:\s*"((?:[^"\\]|\\.){3,300})"', "title"),
        (r'"jobTitle"\s*:\s*"((?:[^"\\]|\\.){3,300})"', "title"),
        (r'"companyName"\s*:\s*"((?:[^"\\]|\\.){1,200})"', "company"),
        (r'"company"\s*:\s*"((?:[^"\\]|\\.){1,200})"', "company"),
        (r'"jobLocation(?:City|Display)?"\s*:\s*"((?:[^"\\]|\\.){1,150})"', "location"),
        (r'"location"\s*:\s*"((?:[^"\\]|\\.){1,150})"', "location"),
        (r'"description"\s*:\s*"((?:[^"\\]|\\.){100,12000})"', "description"),
        (r'"jobDescription"\s*:\s*"((?:[^"\\]|\\.){100,12000})"', "description"),
        (r'"snippet"\s*:\s*"((?:[^"\\]|\\.){50,8000})"', "description"),
        # Shorter description fallback (some pages use truncated snippet)
        (r'"description"\s*:\s*"((?:[^"\\]|\\.){50,12000})"', "description"),
    ]
    for pattern, key in patterns:
        for match in re.finditer(pattern, html):
            val = match.group(1)
            try:
                val = val.encode("utf-8").decode("unicode_escape")
            except Exception:
                pass
            val = re.sub(r"<[^>]+>", " ", val).strip()
            if key == "description" and len(val) > 50:
                if not found.get("description") or len(val) > len(found.get("description") or ""):
                    found["description"] = val[:12000]
            elif key != "description" and val:
                found.setdefault(key, val)

    # Also dig through script JSON blobs for job-like structures (Indeed sometimes nests data)
    if not found.get("title") or not found.get("description"):
        for script_match in re.finditer(r"<script[^>]*>([\s\S]*?)</script>", html, re.IGNORECASE):
            blob = script_match.group(1)
            if len(blob) < 200 or "job" not in blob.lower():
                continue
            try:
                # Find JSON-like "key":"value" for title/company/description
                for key_name, out_key in (
                    (r'"title"\s*:\s*"([^"]{3,300})"', "title"),
                    (r'"jobTitle"\s*:\s*"([^"]{3,300})"', "title"),
                    (r'"companyName"\s*:\s*"([^"]{1,200})"', "company"),
                    (r'"company"\s*:\s*"([^"]{1,200})"', "company"),
                    (r'"description"\s*:\s*"((?:[^"\\]|\\.){100,12000})"', "description"),
                    (r'"snippet"\s*:\s*"((?:[^"\\]|\\.){50,8000})"', "description"),
                ):
                    m = re.search(key_name, blob)
                    if m:
                        val = m.group(1).strip()
                        try:
                            val = val.encode("utf-8").decode("unicode_escape")
                        except Exception:
                            pass
                        val = re.sub(r"<[^>]+>", " ", val).strip()
                        if out_key == "description" and len(val) > 50:
                            if not found.get("description") or len(val) > len(found.get("description") or ""):
                                found["description"] = val[:12000]
                        elif out_key != "description" and val:
                            found.setdefault(out_key, val)
            except Exception:
                pass

    return found if found else None


def _extract_workday_like_json(html: str) -> dict[str, Any] | None:
    """Try to find job title/description in Workday, Rippling, Indeed, or other embedded JSON."""
    found: dict[str, Any] = {}

    def dig(d: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(d, dict):
            for k, v in d.items():
                klo = k.lower()
                if klo in ("title", "jobtitle", "headline", "positiontitle") and isinstance(v, str) and 2 < len(v) < 300:
                    found.setdefault("title", v)
                if klo in ("description", "snippet", "jobdescription") and isinstance(v, str) and len(v) > 50:
                    found.setdefault("description", v[:12000])
                if klo in ("company", "companyname", "organization", "employer") and isinstance(v, str):
                    found.setdefault("company", v)
                if klo in ("location", "joblocation", "city", "place") and isinstance(v, str) and len(v) < 200:
                    found.setdefault("location", v)
                if klo in ("salarymin", "salary_min", "minpay", "minimum") and isinstance(v, (int, float)):
                    found.setdefault("salary_min", int(v))
                if klo in ("salarymax", "salary_max", "maxpay", "maximum") and isinstance(v, (int, float)):
                    found.setdefault("salary_max", int(v))
                dig(v, depth + 1)
        elif isinstance(d, list):
            for x in d[:5]:
                dig(x, depth + 1)

    # Prefer __NEXT_DATA__ (Next.js / Rippling-style) and similar known payloads
    for script_id in ("__NEXT_DATA__", "__NUXT_DATA__", "window.__INITIAL_STATE__", "window.__data"):
        m = re.search(rf'<script[^>]*id=["\']?{re.escape(script_id)}["\']?[^>]*>([\s\S]*?)</script>', html, re.IGNORECASE)
        if not m:
            m = re.search(rf'{re.escape(script_id)}\s*=\s*(\{{[\s\S]*?\}});', html)
        if m:
            blob = m.group(1).strip()
            if blob.startswith("{"):
                try:
                    data = json.loads(blob)
                    dig(data, 0)
                    if found:
                        return found
                except json.JSONDecodeError:
                    pass

    for match in re.finditer(r"<script[^>]*>([\s\S]*?)</script>", html, re.IGNORECASE):
        blob = match.group(1)
        if len(blob) < 150:
            continue
        blob_lower = blob.lower()
        if "job" not in blob_lower and "title" not in blob_lower and "position" not in blob_lower:
            continue
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            for name in ("title", "jobTitle", "headline", "positionTitle"):
                m = re.search(rf'["\']?{name}["\']?\s*:\s*["\']([^"\']{{3,200}})["\']', blob, re.IGNORECASE)
                if m:
                    found["title"] = m.group(1).strip()
                    break
            if "description" in blob_lower:
                m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', blob)
                if m:
                    try:
                        desc = m.group(1).encode("utf-8").decode("unicode_escape")
                    except Exception:
                        desc = m.group(1)
                    if len(desc) > 50:
                        found["description"] = desc[:8000]
            if found:
                return found
            continue
        if not isinstance(data, dict):
            continue
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
    segments = [s for s in path.split("/") if s and not s.startswith("en-")]
    if not segments:
        return ""
    slug = segments[-1]
    # Skip generic path segments that are not job titles (Indeed, LinkedIn, etc.)
    if slug.lower() in ("viewjob", "jobs", "job", "careers", "career", "apply", "home", "view", "listing", "search"):
        return ""
    # Skip UUID-like segments (Rippling, Greenhouse, etc.)
    if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", slug, re.IGNORECASE):
        return ""
    # Remove common suffixes like _REQ-6008-1 or -REQ-6008
    slug = re.sub(r"_?REQ[-_]?\d+[-_]?\d*$", "", slug, flags=re.IGNORECASE)
    slug = re.sub(r"[-_]?\d+$", "", slug)
    slug = slug.replace("-", " ").replace("_", " ").strip()
    if not slug:
        return ""
    return slug.title()


def _indeed_title_from_query(url: str) -> str:
    """If URL is Indeed with q= (search query), return a readable title e.g. 'Machine Learning Engineer'."""
    if "indeed.com" not in url.lower():
        return ""
    from urllib.parse import parse_qs, unquote_plus, urlparse

    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    q = (qs.get("q") or [None])[0]
    if not q or not isinstance(q, str):
        return ""
    decoded = unquote_plus(q).strip()
    if len(decoded) < 2 or len(decoded) > 120:
        return ""
    return decoded.title()


def _indeed_company_from_path(url: str) -> str:
    """If URL is Indeed /cmp/Company-Name/jobs, return a readable company name e.g. 'Ind Technology Pty Ltd'."""
    if "indeed.com" not in url.lower():
        return ""
    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    path = (parsed.path or "").strip("/")
    if "/cmp/" not in path.lower() and not path.lower().startswith("cmp/"):
        return ""
    parts = path.split("/")
    try:
        cmp_idx = next(i for i, p in enumerate(parts) if p.lower() == "cmp")
        if cmp_idx + 1 < len(parts):
            slug = parts[cmp_idx + 1]
            name = unquote(slug).replace("-", " ").replace("_", " ").strip()
            if 2 <= len(name) <= 120:
                return name.title()
    except StopIteration:
        pass
    return ""


def _generic_title_from_url(url: str) -> str:
    """Return a short generic title when we can't derive one from path (e.g. Indeed viewjob)."""
    url_lower = url.lower()
    if "indeed.com" in url_lower:
        from_indeed_q = _indeed_title_from_query(url)
        if from_indeed_q:
            return from_indeed_q
        company = _indeed_company_from_path(url)
        if company:
            return f"Job at {company}"
        return "Job from Indeed"
    if "linkedin.com" in url_lower:
        return "Job from LinkedIn"
    if "bamboohr.com" in url_lower:
        company = _company_from_bamboohr_subdomain(url)
        if company:
            return f"Job at {company}"
        return "Job from BambooHR"
    if "rippling.com" in url_lower or "workday" in url_lower:
        return "Job from listing"
    return "Job from link"


def _company_from_bamboohr_subdomain(url: str) -> str:
    """Extract company name from BambooHR subdomain, e.g. tractionrec.bamboohr.com -> Traction Rec."""
    if "bamboohr.com" not in url.lower():
        return ""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if not host.endswith(".bamboohr.com") or host == "bamboohr.com":
        return ""
    sub = host.removesuffix(".bamboohr.com").strip()
    if not sub:
        return ""
    # Convert subdomain to title: tractionrec -> Tractionrec; my-company -> My Company
    name = sub.replace("-", " ").replace("_", " ").strip()
    if 2 <= len(name) <= 80:
        return name.title()
    return ""


def _try_bamboohr_job_api(url: str) -> dict[str, Any] | None:
    """
    For *.bamboohr.com/careers/<id>, try to fetch job data from a public list/API if available.
    Many BambooHR career sites expose /careers/list (JSON) or embed job data; we try list first.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if "bamboohr.com" not in (parsed.netloc or "").lower():
        return None
    path = (parsed.path or "").strip("/")
    if not path.startswith("careers/"):
        return None
    parts = path.split("/")
    if len(parts) < 2:
        return None
    job_id = parts[1]
    if not job_id or not job_id.isdigit():
        return None
    base = f"{parsed.scheme}://{parsed.netloc}"
    import httpx

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/html",
    }
    for list_path in ("/careers/list", "/careers/list.json", "/api/careers/list"):
        try:
            with httpx.Client(follow_redirects=True, timeout=12.0) as client:
                r = client.get(base + list_path, headers=headers)
                if r.status_code != 200 or len(r.content) < 20:
                    continue
                ct = (r.headers.get("content-type") or "").lower()
                if "json" not in ct and not r.text.strip().startswith("{"):
                    continue
                try:
                    data = r.json()
                except Exception:
                    continue
                # Find job by id in list (common shapes: { jobs: [...] } or [ { id: "135" }, ... ])
                jobs = None
                if isinstance(data, list):
                    jobs = data
                elif isinstance(data, dict):
                    jobs = data.get("jobs") or data.get("jobOpenings") or data.get("positions") or data.get("results")
                if not jobs or not isinstance(jobs, list):
                    continue
                for job in jobs:
                    if not isinstance(job, dict):
                        continue
                    jid = job.get("id") or job.get("jobId") or job.get("job_id")
                    if jid is None:
                        continue
                    if str(jid).strip() == str(job_id).strip():
                        out: dict[str, Any] = {"url": url}
                        out["title"] = (job.get("title") or job.get("jobTitle") or job.get("name") or "").strip() or None
                        out["company"] = (job.get("company") or job.get("companyName") or job.get("organization") or "").strip() or None
                        out["location"] = (job.get("location") or job.get("jobLocation") or job.get("city") or "").strip() or None
                        out["description"] = (job.get("description") or job.get("jobDescription") or job.get("content") or "").strip() or None
                        out["salary_min"] = _coerce_int(job.get("salaryMin") or job.get("salary_min") or job.get("minSalary"))
                        out["salary_max"] = _coerce_int(job.get("salaryMax") or job.get("salary_max") or job.get("maxSalary"))
                        if out.get("title") or out.get("description"):
                            if not out.get("company"):
                                out["company"] = _company_from_bamboohr_subdomain(url) or None
                            return out
                break
        except Exception:
            continue
    return None


def title_from_url_path(url: str) -> str:
    """Public helper: derive a readable job title from URL path."""
    return _title_from_url_path(url)


def _html_to_text(html: str) -> str:
    """Extract main text from HTML. Prefer job content regions (Indeed, Rippling, etc.)."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        # Indeed viewjob, BambooHR, etc.: primary description containers
        for selector in (
            "#jobDescriptionText",
            ".jobsearch-jobDescriptionText",
            ".jobsearch-JobComponent-description",
            ".job-details__content",
            ".job-details-content",
            "[data-qa='job-description']",
            ".jdp-description",
            "main",
            "[role='main']",
            "article",
            ".job-description",
            ".job-details",
            ".job-content",
            ".description",
            "#job-description",
        ):
            try:
                main = soup.select_one(selector)
                if main and len(main.get_text(strip=True)) > 300:
                    for tag in main(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()
                    text = main.get_text(separator="\n")
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    if lines:
                        return "\n".join(lines)
            except Exception:
                pass
        # Fallback: full body without script/style/nav/footer/header
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


def _is_search_results_url(url: str) -> bool:
    """True if URL looks like a job search/list page rather than a single job."""
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    path = (parsed.path or "").lower()
    qs = parse_qs(parsed.query)
    if "indeed.com" in (parsed.netloc or "").lower():
        if "/viewjob" in path or "jk=" in (parsed.query or ""):
            return False
        if "/jobs" in path or (path in ("/", "") and "q" in qs):
            return True
    if "linkedin.com" in (parsed.netloc or "").lower() and "/jobs/view" not in path:
        return True
    return False


def _build_indeed_embedded_url(url: str) -> str | None:
    """If url is an Indeed page with jk= (viewjob or company /cmp/.../jobs), return the embedded job view URL; else None."""
    if "indeed.com" not in url.lower() or "viewtype=embedded" in url.lower():
        return None
    from urllib.parse import parse_qs, urlparse, urlunparse

    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    jk = qs.get("jk", [None])[0]
    if not jk:
        return None
    # Canonical embedded job URL is always /viewjob?viewtype=embedded&jk=...
    embedded_query = f"viewtype=embedded&jk={jk}"
    return urlunparse((parsed.scheme, parsed.netloc, "/viewjob", "", embedded_query, ""))


def _fetch_page_html(url: str) -> str:
    """Fetch URL and return raw HTML. Retries on failure. For Indeed viewjob, tries embedded view if main page fails or is empty."""
    import time

    import httpx

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    max_attempts = 3

    # Indeed company page (/cmp/.../jobs?jk=) shows a listing; fetch embedded job URL to get single-job content
    fetch_url = url
    if "indeed.com" in url and "viewjob" not in url.lower():
        embedded_url = _build_indeed_embedded_url(url)
        if embedded_url:
            fetch_url = embedded_url

    for attempt in range(max_attempts):
        try:
            with httpx.Client(follow_redirects=True, timeout=20.0) as client:
                resp = client.get(fetch_url, headers=headers)
                resp.raise_for_status()
                html = resp.text
                # Indeed viewjob: main page is often JS-rendered; embedded view can have HTML with job body
                if "indeed.com" in url and "viewjob" in fetch_url.lower() and "viewtype=embedded" not in fetch_url:
                    text_from_main = _html_to_text(html)
                    if not text_from_main or len(text_from_main.strip()) < 400:
                        embedded_url = _build_indeed_embedded_url(url)
                        if embedded_url:
                            try:
                                emb = client.get(embedded_url, headers=headers, timeout=15.0)
                                if emb.status_code == 200 and len(emb.text) > 500:
                                    html = emb.text
                            except Exception:
                                pass
                return html
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(1.5)
            continue

    # Last resort for Indeed viewjob: try embedded URL when main URL failed (e.g. blocked or timeout)
    embedded_url = _build_indeed_embedded_url(url)
    if embedded_url:
        try:
            with httpx.Client(follow_redirects=True, timeout=15.0) as client:
                resp = client.get(embedded_url, headers=headers)
                if resp.status_code == 200 and len(resp.text) > 200:
                    return resp.text
        except Exception:
            pass

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
    # If main fetch failed (e.g. Indeed blocking or timeout), try Indeed embedded view as fallback
    if not html and "indeed.com" in url:
        embedded_url = _build_indeed_embedded_url(url)
        if embedded_url:
            html = _fetch_page_html(embedded_url)

    # 0b) BambooHR: try public job list API when URL is /careers/<id> (page is usually JS-rendered)
    if "bamboohr.com" in url:
        bamboohr = _try_bamboohr_job_api(url)
        if bamboohr and (bamboohr.get("title") or bamboohr.get("description")):
            result = {
                "title": bamboohr.get("title"),
                "company": bamboohr.get("company"),
                "location": bamboohr.get("location"),
                "salary_min": bamboohr.get("salary_min"),
                "salary_max": bamboohr.get("salary_max"),
                "description": bamboohr.get("description"),
                "skills": [],
                "url": url,
            }
            if not result.get("title"):
                result["title"] = _generic_title_from_url(url)
            return result

    # 0) Indeed viewjob: try Indeed-specific extraction (script/embedded data).
    # Only use it when we got a real title and (company or substantial description); otherwise prefer HTML+LLM.
    if "indeed.com" in url:
        indeed = _extract_indeed_job(html)
        desc = indeed.get("description") if indeed else ""
        has_substantial = desc and len((desc or "").strip()) > 300
        use_indeed = (
            indeed
            and (indeed.get("title") or indeed.get("company"))
            and (indeed.get("company") or has_substantial)
        )
        if use_indeed:
            result = {
                "title": indeed.get("title"),
                "company": indeed.get("company"),
                "location": indeed.get("location"),
                "salary_min": None,
                "salary_max": None,
                "description": indeed.get("description"),
                "skills": [],
                "url": url,
            }
            if not result.get("title"):
                result["title"] = _title_from_url_path(url) or _generic_title_from_url(url)
            return result
    # 1) Try JSON-LD JobPosting (Indeed, Rippling, many ATS)
    jsonld = _extract_jsonld_job(html)
    if jsonld:
        result = {
            "title": jsonld.get("title"),
            "company": jsonld.get("company"),
            "location": jsonld.get("location"),
            "salary_min": jsonld.get("salary_min"),
            "salary_max": jsonld.get("salary_max"),
            "description": jsonld.get("description"),
            "skills": _coerce_skills(jsonld.get("skills")) if jsonld.get("skills") else [],
            "url": url,
        }
        if not result.get("title"):
            result["title"] = _title_from_url_path(url) or _generic_title_from_url(url)
        return result

    # 2) Try Workday / Rippling / Indeed embedded JSON
    embedded = _extract_workday_like_json(html)
    if embedded:
        result = {
            "title": embedded.get("title"),
            "company": embedded.get("company"),
            "location": embedded.get("location"),
            "salary_min": embedded.get("salary_min"),
            "salary_max": embedded.get("salary_max"),
            "description": embedded.get("description"),
            "skills": [],
            "url": url,
        }
        if result.get("title") or result.get("description"):
            if not result.get("title"):
                result["title"] = _title_from_url_path(url) or _generic_title_from_url(url)
            return result

    # 3) Main text from same HTML + LLM
    text = _html_to_text(html)
    url_hint = f"Source URL: {url}"
    indeed_q = _indeed_title_from_query(url)
    if indeed_q:
        url_hint += f"\nIndeed search query from URL (use as job title if not clearly stated in the text): {indeed_q}."
    if _is_search_results_url(url):
        url_hint += "\nThis URL may be a job search results page (e.g. Indeed list). Extract any job listing snippets you find (title, company, location, salary). If multiple jobs appear, pick the first or most prominent one. If no clear single job is found, use a generic title from the page or query (e.g. 'Data Scientist jobs - Vancouver')."
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
        result["title"] = _title_from_url_path(url) or _generic_title_from_url(url)
    return result


# --- Agentic add-job: tool-calling loop and tools ---

ADD_JOB_AGENT_SYSTEM = """You are a job ingestion agent. Your goal is to extract a complete job record (title, company, location, salary_min, salary_max, description, skills, url) from the user's input (URL, pasted text, or file).

You have tools at your disposal. Use them when the initial extraction failed or returned empty/weak data:
1. **fetch_page_content** – Fetch a URL and get the main text from the page. Use when you have a URL and need to (re)fetch or try an alternative URL.
2. **extract_from_text** – Run the LLM extractor on raw text. Use after you get page content or when the user pasted text. Pass optional hints (e.g. "Source URL: ...", "Indeed search query: ...").
3. **try_indeed_embedded** – For Indeed URLs with job id (jk=), fetch the embedded job view which often has better content. Use when the main page returned little text.
4. **try_bamboohr_api** – For BambooHR career URLs (*.bamboohr.com/careers/123), try the list API to get job data. Use when the URL looks like BambooHR.
5. **web_search** – Search the web for job title or company + role. Use when you need to disambiguate or find missing fields (e.g. company name from a vague page).
6. **finalize_proposal** – When you have the best job record you can produce (even if some fields are null), call this with all fields. The user will then see the details and confirm before the job is added. Always call finalize_proposal exactly once when you are done; do not skip it.

Strategy: start with the input you were given. If it's a URL, try fetch_page_content first. If the result is empty or too short, try try_indeed_embedded (for Indeed) or try_bamboohr_api (for BambooHR). Then run extract_from_text on the text you have. If key fields are still missing, try web_search. Then call finalize_proposal with whatever you have (use null for unknown fields; the user can edit on the confirmation screen)."""


def _tool_fetch_page_content(url: str) -> dict[str, Any]:
    """Fetch URL and return main text content (for agent tool)."""
    if not (url or "").strip():
        return {"error": "URL is required", "text": ""}
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    html = _fetch_page_html(url)
    text = _html_to_text(html)
    return {"url": url, "text": (text or "")[:12000], "success": bool(text and len(text.strip()) > 100)}


def _tool_extract_from_text(text: str, hints: str = "") -> dict[str, Any]:
    """Run LLM extraction on text (for agent tool)."""
    if not (text or text.strip()):
        return {"error": "Text is required", "extracted": {}}
    extracted = _extract_with_llm(text.strip()[:30000], hints)
    return {"extracted": extracted, "has_title": bool(extracted.get("title")), "has_description": bool(extracted.get("description"))}


def _tool_try_indeed_embedded(url: str) -> dict[str, Any]:
    """Fetch Indeed embedded job view if URL has jk= (for agent tool)."""
    embedded_url = _build_indeed_embedded_url(url)
    if not embedded_url:
        return {"error": "Not an Indeed URL with job id (jk=)", "text": ""}
    html = _fetch_page_html(embedded_url)
    text = _html_to_text(html)
    return {"url": embedded_url, "text": (text or "")[:12000], "success": bool(text and len(text.strip()) > 100)}


def _tool_try_bamboohr_api(url: str) -> dict[str, Any]:
    """Try BambooHR careers list API (for agent tool)."""
    result = _try_bamboohr_job_api(url)
    if result and (result.get("title") or result.get("description")):
        return {"found": True, "job": result}
    return {"found": False, "job": None}


def _tool_web_search(query: str) -> dict[str, Any]:
    """Search the web for job/company info. Can be backed by MCP or search API (for agent tool)."""
    import os

    # Optional: Tavily, SerpAPI, or MCP – check env
    api_key = os.environ.get("TAVILY_API_KEY") or os.environ.get("SERPAPI_API_KEY")
    if api_key and os.environ.get("TAVILY_API_KEY"):
        try:
            import httpx
            r = httpx.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "search_depth": "basic", "max_results": 5},
                timeout=10.0,
            )
            if r.status_code == 200:
                data = r.json()
                results = data.get("results") or []
                snippets = [{"title": x.get("title"), "url": x.get("url"), "content": (x.get("content") or "")[:500]} for x in results[:5]]
                return {"success": True, "results": snippets}
        except Exception as e:
            return {"success": False, "error": str(e), "results": []}
    # Placeholder when no search API: return empty so agent can still finalize
    return {"success": False, "results": [], "hint": "Set TAVILY_API_KEY in .env for web search. You can still finalize with the data you have."}


def _tool_finalize_proposal(
    title: str | None = None,
    company: str | None = None,
    location: str | None = None,
    salary_min: int | None = None,
    salary_max: int | None = None,
    description: str | None = None,
    skills: list[str] | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """Submit the proposed job for user confirmation (for agent tool). Called by the model when done."""
    skills_list = skills if isinstance(skills, list) else ([str(skills)] if skills else [])
    return {
        "status": "proposal_ready",
        "proposal": {
            "title": (title or "").strip() or None,
            "company": (company or "").strip() or None,
            "location": (location or "").strip() or None,
            "salary_min": _coerce_int(salary_min),
            "salary_max": _coerce_int(salary_max),
            "description": (description or "").strip() or None,
            "skills": [str(s).strip() for s in skills_list if str(s).strip()],
            "url": (url or "").strip() or None,
        },
    }


ADD_JOB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_page_content",
            "description": "Fetch a URL and return the main text content of the page. Use when you need to load or retry loading a job page.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full URL to fetch (e.g. https://ca.indeed.com/viewjob?jk=...)"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_from_text",
            "description": "Extract job fields (title, company, location, salary, description, skills) from raw text using the LLM. Use after you have page content or pasted text. Pass optional hints (e.g. source URL, search query) to improve extraction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The job description or page text to extract from"},
                    "hints": {"type": "string", "description": "Optional hints, e.g. 'Source URL: ...', 'Indeed search query: ...'"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "try_indeed_embedded",
            "description": "For Indeed job URLs that contain jk= (job id), fetch the embedded job view which often has better content when the main page is JavaScript-heavy. Call this when the URL is Indeed and fetch_page_content returned little or no text.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The Indeed URL (e.g. viewjob or /cmp/.../jobs?jk=...)"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "try_bamboohr_api",
            "description": "For BambooHR career URLs (*.bamboohr.com/careers/123), try to get job data from the public list API. Use when the URL looks like a BambooHR careers page.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The BambooHR careers URL"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for job title, company name, or role information. Use when you need to fill in missing fields (e.g. company name from a vague page) or disambiguate. Requires TAVILY_API_KEY in .env to work.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query (e.g. 'Acme Corp Software Engineer job')"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_proposal",
            "description": "Call this when you have the best job record you can produce. Pass all fields you extracted (use null for unknown). The user will see the details and confirm before the job is added. You MUST call this exactly once when done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "location": {"type": "string"},
                    "salary_min": {"type": "integer"},
                    "salary_max": {"type": "integer"},
                    "description": {"type": "string"},
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "url": {"type": "string"},
                },
                "required": [],
            },
        },
    },
]


def run_add_job_agent(
    mode: str,
    *,
    url: str | None = None,
    text: str | None = None,
    file_content: bytes | None = None,
    filename: str | None = None,
    location: str | None = None,
    salary_min: int | None = None,
    salary_max: int | None = None,
) -> dict[str, Any] | None:
    """
    Run the agentic add-job flow with tool calling. Returns a proposal dict (title, company, ...)
    for user confirmation, or None if the agent gave up or errored.
    """
    user_parts = []
    if mode == "url" and url:
        url_clean = url.strip()
        if not url_clean.startswith("http"):
            url_clean = "https://" + url_clean
        user_parts.append(f"The user wants to add a job from this URL: {url_clean}")
        user_parts.append("Fetch the page, extract job details, and use try_indeed_embedded or try_bamboohr_api if the first fetch returns empty or poor content. Then call finalize_proposal with the best job record you can produce.")
    elif mode == "file" and file_content and filename:
        raw_text = _extract_text_from_file(file_content, filename or "")
        user_parts.append(f"The user uploaded a file: {filename}. Extracted text (first 8000 chars):\n{(raw_text or '')[:8000]}")
        user_parts.append("Run extract_from_text on this text, then call finalize_proposal with the result. If extraction is weak, you can try web_search to fill gaps.")
    elif mode == "manual" and text:
        hints = []
        if location:
            hints.append(f"Location: {location}")
        if salary_min is not None or salary_max is not None:
            hints.append(f"Salary: {salary_min or '?'} - {salary_max or '?'}")
        user_parts.append(f"The user pasted a job description:\n{(text or '')[:8000]}")
        user_parts.append("Run extract_from_text with the text and optional hints, then call finalize_proposal. If fields are missing, you can try web_search.")
    else:
        return None

    user_message = "\n\n".join(user_parts)

    client = _get_openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ADD_JOB_AGENT_SYSTEM},
        {"role": "user", "content": user_message},
    ]

    max_steps = 15
    for _ in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=ADD_JOB_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            }
        )

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "fetch_page_content":
                result = _tool_fetch_page_content(args.get("url") or "")
            elif name == "extract_from_text":
                result = _tool_extract_from_text(args.get("text") or "", args.get("hints") or "")
            elif name == "try_indeed_embedded":
                result = _tool_try_indeed_embedded(args.get("url") or "")
            elif name == "try_bamboohr_api":
                result = _tool_try_bamboohr_api(args.get("url") or "")
            elif name == "web_search":
                result = _tool_web_search(args.get("query") or "")
            elif name == "finalize_proposal":
                result = _tool_finalize_proposal(
                    title=args.get("title"),
                    company=args.get("company"),
                    location=args.get("location"),
                    salary_min=args.get("salary_min"),
                    salary_max=args.get("salary_max"),
                    description=args.get("description"),
                    skills=args.get("skills"),
                    url=args.get("url"),
                )
                if result.get("status") == "proposal_ready" and result.get("proposal"):
                    return result["proposal"]
            else:
                result = {"error": f"Unknown tool: {name}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result),
                }
            )

    return None
