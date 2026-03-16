"""
Interview Preparation Chatbot Agent.

Uses job description and user resume for context. Optionally searches the web
(Glassdoor, Reddit, company site, general search) for company culture, interview
rounds, and feedback to tailor preparation advice.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

# Reuse resume + job loading from resume_improvement
from career_copilot.agents.resume_improvement import build_resume_improvement_context

INTERVIEW_TYPE_PROMPT = """What type of interview are you preparing for?

You can pick one of these common types, or describe your own:

- **Recruiter / Introduction** — Initial screen, background, salary expectations
- **Technical** — Domain knowledge, tools, past projects
- **Coding** — Algorithms, data structures, live coding (if applicable to the role)
- **System Design** — Architecture, scalability (if applicable)
- **HR / Behavioural** — STAR stories, culture fit, strengths and weaknesses
- **Final / Negotiations** — Offer discussion, compensation, start date

Reply with the type (e.g. "Technical" or "Coding") or describe your interview in your own words. I'll then search for company-specific insights and tailor a preparation plan for you."""


def _resolve_result_url(href: str) -> str:
    """
    Resolve DuckDuckGo redirect/tracking URLs by extracting the real destination
    from the uddg (or url/u) query parameter. We do NOT follow HTTP redirects:
    sites like Glassdoor often redirect to a different company page when hit
    server-side, which would attach the wrong link to the snippet.
    """
    if not (href or "").strip():
        return href or ""
    href = href.strip()
    try:
        parsed = urlparse(href)
        if "duckduckgo.com" in (parsed.netloc or "").lower() and (
            (parsed.path or "").rstrip("/") in ("/l", "")
        ):
            qs = parse_qs(parsed.query)
            for key in ("uddg", "url", "u"):
                if key in qs and qs[key]:
                    raw = qs[key][0]
                    resolved = unquote(raw)
                    if resolved.startswith("http://") or resolved.startswith("https://"):
                        return resolved
        return href
    except Exception:
        return href


def _is_generic_or_low_value_url(url: str) -> bool:
    """
    Filter out homepage/generic URLs that are not useful as citations
    (e.g. glassdoor.com, reddit.com with no specific page).
    """
    if not url or not url.startswith("http"):
        return True
    try:
        parsed = urlparse(url)
        path = (parsed.path or "").strip().rstrip("/").lower()
        netloc = (parsed.netloc or "").lower()

        if "glassdoor" in netloc:
            # Filter only bare homepage or top-level nav (no specific company/page)
            if not path or path == "/":
                return True
            if path in ("/overview", "/employer-list", "/reviews", "/interviews"):
                return True
            # Single segment like /Overview (no company slug)
            if path.count("/") <= 1:
                return True
            return False

        if "reddit.com" in netloc:
            if not path or path == "/":
                return True
            # Must have at least /r/SubredditName or /r/.../comments/...
            parts = [p for p in path.split("/") if p]
            if len(parts) < 2:  # e.g. just ["r"]
                return True
            if parts[0] != "r":
                return True
            return False

        return False
    except Exception:
        return False


def _get_openai_client():
    import os

    from career_copilot.database.db import load_env

    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env for the interview preparation chatbot."
        )
    from openai import OpenAI

    return OpenAI()


def search_web_for_company(
    company_name: str,
    job_title: str,
    interview_type: str,
    max_results_per_query: int = 5,
) -> dict[str, Any]:
    """
    Search the web for company culture, interview process, reviews, and tips.
    Explicitly searches Glassdoor and Reddit; also Fishbowl, company culture, and role-specific questions.
    Returns dict with: context (str), found_glassdoor (bool), found_reddit (bool).
    """
    # Explicit Glassdoor and Reddit queries first so we can track if we got any results from each
    glassdoor_query = f"{company_name} interview review site:glassdoor.com"
    reddit_query = f"{company_name} interview experience site:reddit.com"
    other_queries = [
        f"{company_name} interview process Glassdoor",
        f"{company_name} interview experience Reddit",
        f"{company_name} interview Fishbowl",
        f"{company_name} company culture",
        f"{company_name} {job_title} interview questions",
    ]
    snippets: list[str] = []
    source_list: list[tuple[str, str]] = []  # (label, resolved_url) for exact citation
    seen_bodies: set[str] = set()
    found_glassdoor = False
    found_reddit = False

    def is_glassdoor_url(href: str) -> bool:
        return "glassdoor" in (href or "").lower()

    def is_reddit_url(href: str) -> bool:
        return "reddit.com" in (href or "").lower()

    def add_snippet(label: str, body: str, link: str, is_gd: bool, is_rd: bool) -> None:
        if not link or _is_generic_or_low_value_url(link):
            return
        nonlocal found_glassdoor, found_reddit
        if is_gd:
            found_glassdoor = True
        if is_rd:
            found_reddit = True
        seen_bodies.add(body)
        snippets.append(f"[{label}]\n{body[:600]}{'…' if len(body) > 600 else ''}\nSource: {link}")
        source_list.append((label, link))

    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            # Dedicated Glassdoor search
            try:
                for r in ddgs.text(glassdoor_query, max_results=max_results_per_query):
                    title = (r.get("title") or "").strip()
                    body = (r.get("body") or "").strip()
                    href = (r.get("href") or "").strip()
                    if not body or body in seen_bodies or not is_glassdoor_url(href):
                        continue
                    link = _resolve_result_url(href)
                    add_snippet(f"Glassdoor - {title}", body, link, True, False)
            except Exception:
                pass
            # Dedicated Reddit search
            try:
                for r in ddgs.text(reddit_query, max_results=max_results_per_query):
                    title = (r.get("title") or "").strip()
                    body = (r.get("body") or "").strip()
                    href = (r.get("href") or "").strip()
                    if not body or body in seen_bodies or not is_reddit_url(href):
                        continue
                    link = _resolve_result_url(href)
                    add_snippet(f"Reddit - {title}", body, link, False, True)
            except Exception:
                pass
            # Other queries (may also return Glassdoor/Reddit)
            for q in other_queries:
                try:
                    for r in ddgs.text(q, max_results=max_results_per_query):
                        title = (r.get("title") or "").strip()
                        body = (r.get("body") or "").strip()
                        href = (r.get("href") or "").strip()
                        if not body or body in seen_bodies:
                            continue
                        link = _resolve_result_url(href) if href else ""
                        if not link:
                            continue
                        add_snippet(
                            title,
                            body,
                            link,
                            is_glassdoor_url(href),
                            is_reddit_url(href),
                        )
                except Exception:
                    continue
    except ImportError:
        return {
            "context": (
                "Web search is not available (install duckduckgo-search: pip install duckduckgo-search). "
                "Preparation will be based on the job description and your resume only."
            ),
            "found_glassdoor": False,
            "found_reddit": False,
        }

    if not snippets:
        return {
            "context": "No web results found. Preparation will be based on the job description and your resume only.",
            "found_glassdoor": False,
            "found_reddit": False,
        }

    context_text = "\n\n---\n\n".join(snippets[:20])
    if source_list:
        exact_urls = "\n".join(
            f"{i}. {label}: {url}" for i, (label, url) in enumerate(source_list[:25], 1)
        )
        context_text += (
            "\n\n--- EXACT URLs to cite (copy these URLs exactly in your markdown links; "
            "do not use a generic homepage like glassdoor.com or reddit.com) ---\n" + exact_urls
        )

    return {
        "context": context_text,
        "found_glassdoor": found_glassdoor,
        "found_reddit": found_reddit,
    }


def get_initial_interview_message() -> str:
    """First message from the bot: ask the user what type of interview they are preparing for."""
    return INTERVIEW_TYPE_PROMPT


def _build_system_prompt(
    job: dict[str, Any],
    resume_text: str,
    web_search_context: str,
    found_glassdoor: bool,
    found_reddit: bool,
) -> str:
    company = job.get("company") or "the company"
    title = job.get("title") or "the role"
    job_desc = (job.get("description") or "")[:4500]
    skills_line = ""
    if job.get("skills"):
        skills_line = "Mentioned skills: " + ", ".join(job["skills"]) + "."

    search_note = ""
    if not found_glassdoor or not found_reddit:
        parts = []
        if not found_glassdoor:
            parts.append("Glassdoor")
        if not found_reddit:
            parts.append("Reddit")
        search_note = f" We explicitly searched Glassdoor and Reddit. We did NOT find any reviews or comments on: {', '.join(parts)}. In your 'From searching online' section you MUST state that you didn't find any reviews or comments on {', '.join(parts)} for this company."

    return f"""You are an interview preparation coach for Career Copilot. The user is preparing for an interview at a specific company for a specific role. Your job is to:

1. Use the interview type they shared (e.g. Technical, HR/Behavioural, Coding) to focus your advice.
2. Use the job description and the user's resume to tailor tips (e.g. which skills to emphasize, which projects to mention).
3. Use the web search results below (from Glassdoor, Reddit, Fishbowl, company site, etc.) to add company-specific advice when available.
4. When giving the main preparation plan (in response to their interview type), you MUST structure your response in three clearly labelled sections:
   - **From the job description:** List points you derived from reviewing the job description (requirements, keywords, focus areas).
   - **From your resume:** List points you derived from reviewing their resume (stories to highlight, skills to mention, projects to discuss).
   - **From searching online:** List points from the web search results. For each point use the matching link from the "EXACT URLs to cite" list (copy the URL exactly). If no URL in the list clearly matches that source, or you are unsure, do NOT invent or reuse a different link — instead say "I couldn't find a reliable link for this source." Do NOT use generic URLs like https://glassdoor.com or https://reddit.com.{search_note}
5. After these three sections, you may add a short "Suggested next steps" or "Questions to ask" if helpful.
6. In follow-up messages, answer their questions and deepen the preparation; you may use a simpler format unless they ask for the full structured breakdown again.

--- Company and role ---
Company: {company}. Role: {title}. {skills_line}

--- Job description (excerpt) ---
{job_desc}

--- User's resume (for tailoring stories and talking points) ---
{resume_text or "(No resume uploaded yet.)"}

--- Web search results and EXACT URLs to cite (in "From searching online" use only the numbered URLs from the list below; copy each URL exactly) ---
{web_search_context}"""


def chat_interview_preparation(
    user_message: str,
    conversation_history: list[dict[str, str]],
    resume_text: str,
    job: dict[str, Any],
) -> str:
    """
    One conversational turn. If this is the first user message (after the initial bot prompt),
    run web search for the company and interview type, then generate a preparation plan.
    Otherwise continue the conversation with existing context (no repeated search).
    """
    if not job:
        return "I don't have the job context. Please start from the job page again."
    if not (user_message or "").strip():
        return "Please tell me what type of interview you're preparing for (e.g. Technical, HR/Behavioural), or describe it in your own words."

    client = _get_openai_client()
    is_first_user_reply = (
        len(conversation_history) == 1 and conversation_history[0].get("role") == "assistant"
    )

    if is_first_user_reply:
        interview_type = user_message.strip()
        search_result = search_web_for_company(
            company_name=job.get("company") or "",
            job_title=job.get("title") or "",
            interview_type=interview_type,
        )
        web_context = search_result["context"]
        found_glassdoor = search_result.get("found_glassdoor", False)
        found_reddit = search_result.get("found_reddit", False)
    else:
        web_context = (
            "(Use any company/interview context already discussed in the conversation. "
            "No new web search was run for this follow-up.)"
        )
        found_glassdoor = True  # avoid prompting to "mention not found" in follow-ups
        found_reddit = True

    system = _build_system_prompt(job, resume_text, web_context, found_glassdoor, found_reddit)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for h in conversation_history:
        role = h.get("role")
        content = h.get("content") or ""
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message.strip()})

    if is_first_user_reply:
        messages[-1]["content"] = (
            "The user said they are preparing for this type of interview: "
            + user_message.strip()
            + "\n\n"
            "Provide a tailored preparation plan in this exact structure:\n\n"
            "## From the job description\n"
            "List 3–5 points you derived from reviewing the job description (requirements, keywords, what to emphasize). Use bullet points.\n\n"
            "## From your resume\n"
            "List 3–5 points you derived from reviewing their resume (which experiences to highlight, stories to tell, skills to mention). Use bullet points.\n\n"
            "## From searching online\n"
            "List points from the web search results. For each point use the matching URL from the 'EXACT URLs to cite' list (copy it exactly). If no URL in the list matches that source, say 'I couldn't find a reliable link for this source' — do not guess or reuse another link. Do NOT use https://glassdoor.com or https://reddit.com. Use bullet points. If we did not find any reviews on Glassdoor or Reddit for this company, say so.\n\n"
            "You may add a short ## Suggested next steps or ## Questions to ask the interviewer at the end. Keep everything actionable and specific to their background and the role."
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
    )
    msg = response.choices[0].message
    return (msg.content or "").strip()


def build_interview_prep_context(job_id: int, user_id: int, conn: Any) -> dict[str, Any]:
    """Load job and user resume for interview prep. Reuses resume_improvement context (job + resume only)."""
    ctx = build_resume_improvement_context(job_id, user_id, conn)
    return {
        "resume_text": ctx["resume_text"],
        "job": ctx["job"],
    }
