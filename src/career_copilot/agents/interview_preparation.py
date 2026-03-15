"""
Interview Preparation Chatbot Agent.

Uses job description and user resume for context. Optionally searches the web
(Glassdoor, Reddit, company site, general search) for company culture, interview
rounds, and feedback to tailor preparation advice.
"""

from __future__ import annotations

from typing import Any

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
) -> str:
    """
    Search the web for company culture, interview process, reviews, and tips.
    Uses DuckDuckGo (no API key). Queries target Glassdoor, Reddit, Fishbowl, company site, and general search.
    Returns a single string of concatenated snippets for the LLM context.
    """
    queries = [
        f"{company_name} interview process Glassdoor",
        f"{company_name} interview experience Reddit",
        f"{company_name} interview Fishbowl",
        f"{company_name} company culture",
        f"{company_name} {job_title} interview questions",
    ]
    snippets: list[str] = []
    seen_bodies: set[str] = set()

    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            for q in queries:
                try:
                    for r in ddgs.text(q, max_results=max_results_per_query):
                        title = (r.get("title") or "").strip()
                        body = (r.get("body") or "").strip()
                        href = (r.get("href") or "").strip()
                        if not body or body in seen_bodies:
                            continue
                        seen_bodies.add(body)
                        snippet = f"[{title}]\n{body[:600]}{'…' if len(body) > 600 else ''}"
                        if href:
                            snippet += f"\nSource: {href}"
                        snippets.append(snippet)
                except Exception:
                    continue
    except ImportError:
        return (
            "Web search is not available (install duckduckgo-search: pip install duckduckgo-search). "
            "Preparation will be based on the job description and your resume only."
        )

    if not snippets:
        return "No web results found. Preparation will be based on the job description and your resume only."

    return "\n\n---\n\n".join(snippets[:20])  # cap total context


def get_initial_interview_message() -> str:
    """First message from the bot: ask the user what type of interview they are preparing for."""
    return INTERVIEW_TYPE_PROMPT


def _build_system_prompt(
    job: dict[str, Any],
    resume_text: str,
    web_search_context: str,
) -> str:
    company = job.get("company") or "the company"
    title = job.get("title") or "the role"
    job_desc = (job.get("description") or "")[:4500]
    skills_line = ""
    if job.get("skills"):
        skills_line = "Mentioned skills: " + ", ".join(job["skills"]) + "."

    return f"""You are an interview preparation coach for Career Copilot. The user is preparing for an interview at a specific company for a specific role. Your job is to:

1. Use the interview type they shared (e.g. Technical, HR/Behavioural, Coding) to focus your advice.
2. Use the job description and the user's resume to tailor tips (e.g. which skills to emphasize, which projects to mention).
3. Use the web search results below (from Glassdoor, Reddit, Fishbowl, company site, Google) to add company-specific advice: typical interview rounds, culture, common questions, and candidate feedback when available.
4. When giving the main preparation plan (in response to their interview type), you MUST structure your response in three clearly labelled sections so the user knows where each point came from:
   - **From the job description:** List points you derived from reviewing the job description (requirements, keywords, focus areas).
   - **From your resume:** List points you derived from reviewing their resume (stories to highlight, skills to mention, projects to discuss).
   - **From searching online:** List points from the web search results (company culture, interview reviews, common questions, tips). For each point that comes from a specific source, include the link. The web search results below include "Source: <url>" — use that exact URL when citing (e.g. "Glassdoor review: ... [link](url)" or "As mentioned on Reddit: ... (source: url)").
5. After these three sections, you may add a short "Suggested next steps" or "Questions to ask" if helpful.
6. In follow-up messages, answer their questions and deepen the preparation (e.g. mock questions, deeper dives); you may use a simpler format in follow-ups unless they ask for the full structured breakdown again.

--- Company and role ---
Company: {company}. Role: {title}. {skills_line}

--- Job description (excerpt) ---
{job_desc}

--- User's resume (for tailoring stories and talking points) ---
{resume_text or "(No resume uploaded yet.)"}

--- Web search results (company culture, interview process, reviews, tips). Each block ends with "Source: <url>" when available — cite these URLs in your "From searching online" section. ---
{web_search_context}

Respond in clear, helpful markdown. Be specific to the company and role. When search results are missing, the "From searching online" section can say "No specific results found; consider checking Glassdoor and Reddit for [company] interviews." and still give strong advice from the job description and resume in the first two sections."""


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
        len(conversation_history) == 1
        and conversation_history[0].get("role") == "assistant"
    )

    if is_first_user_reply:
        interview_type = user_message.strip()
        web_context = search_web_for_company(
            company_name=job.get("company") or "",
            job_title=job.get("title") or "",
            interview_type=interview_type,
        )
    else:
        # For follow-ups we don't re-search; use a short note so the model doesn't expect fresh search data
        web_context = (
            "(Use any company/interview context already discussed in the conversation. "
            "No new web search was run for this follow-up.)"
        )

    system = _build_system_prompt(job, resume_text, web_context)
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
            "List points from the web search results (company culture, interview process, reviews, common questions). For each point that comes from a specific source, include the link (the search results have 'Source: <url>' — use that URL). Use bullet points; e.g. 'Glassdoor reviews mention X ([link](url)).' If no relevant results were found, say so and suggest they check Glassdoor/Reddit for the company name.\n\n"
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
