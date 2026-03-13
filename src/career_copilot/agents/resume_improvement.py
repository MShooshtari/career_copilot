"""
Resume Improvement Chatbot Agent (Chatbot Agent #1).

RAG pipeline: job description, similar jobs, similar resumes (user profiles).
LLM: improve resume to match job, suggest bullets, rewrites, ATS score; then conversational follow-ups.
"""
from __future__ import annotations

from typing import Any

from career_copilot.rag.chroma_store import (
    get_similar_jobs_for_resume_improvement,
    get_similar_resumes_for_resume_improvement,
)

# Reuse same doc size as Chroma job docs
JOB_DOC_MAX_CHARS = 6_000


def _job_dict_to_document(job: dict[str, Any], max_chars: int = JOB_DOC_MAX_CHARS) -> str:
    """Build searchable document string from job dict (same format as Chroma job docs)."""
    parts = []
    if job.get("title"):
        parts.append(job["title"])
    if job.get("company"):
        parts.append(f"Company: {job['company']}")
    if job.get("location"):
        parts.append(f"Location: {job['location']}")
    if job.get("description"):
        parts.append(job["description"])
    skills = job.get("skills")
    if skills:
        parts.append("Skills: " + (", ".join(skills) if isinstance(skills, list) else str(skills)))
    doc = "\n\n".join(parts) if parts else ""
    if len(doc) > max_chars:
        doc = doc[:max_chars].rstrip() + "…"
    return doc


def build_resume_improvement_context(
    job_id: int,
    user_id: int,
    conn: Any,
) -> dict[str, Any]:
    """
    Load resume, job, and RAG context for resume improvement.

    Returns dict with: resume_text, job, job_document, similar_jobs, similar_resumes.
    """
    job = None
    resume_text = ""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source, source_id, title, company, location,
                   salary_min, salary_max, description, skills, posted_at, url
            FROM jobs WHERE id = %s
            """,
            (job_id,),
        )
        row = cur.fetchone()
        if row:
            (
                id_,
                source,
                source_id,
                title,
                company,
                location,
                salary_min,
                salary_max,
                description,
                skills,
                posted_at,
                url,
            ) = row
            job = {
                "id": id_,
                "source": source,
                "source_id": source_id,
                "title": title or "Job",
                "company": company or "",
                "location": location or "",
                "salary_min": salary_min,
                "salary_max": salary_max,
                "description": description or "",
                "skills": list(skills) if skills else [],
                "posted_at": posted_at,
                "url": url or "",
            }
        cur.execute(
            "SELECT resume_file, resume_filename FROM profiles WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        if row and row[0]:
            from career_copilot.resume_io import extract_resume_text

            resume_text = extract_resume_text(row[0], row[1]) or ""

    if not job:
        return {
            "resume_text": resume_text,
            "job": None,
            "job_document": "",
            "similar_jobs": [],
            "similar_resumes": [],
        }

    job_document = _job_dict_to_document(job)
    similar_jobs = get_similar_jobs_for_resume_improvement(job_document, n_results=5)
    similar_resumes = get_similar_resumes_for_resume_improvement(
        job_document, exclude_user_id=user_id, n_results=5
    )

    return {
        "resume_text": resume_text,
        "job": job,
        "job_document": job_document,
        "similar_jobs": similar_jobs,
        "similar_resumes": similar_resumes,
    }


def _get_openai_client():
    import os

    from career_copilot.database.db import load_env

    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env for the resume improvement chatbot.")
    from openai import OpenAI

    return OpenAI()


def _build_system_prompt(
    resume_text: str,
    job: dict[str, Any],
    similar_jobs: list[dict],
    similar_resumes: list[dict],
) -> str:
    """Build the system prompt for the resume improvement agent."""
    company_info = f"Company: {job.get('company') or 'Unknown'}. Role: {job.get('title') or 'Position'}."
    if job.get("location"):
        company_info += f" Location: {job['location']}."
    job_desc = (job.get("description") or "")[:4000]
    skills_line = ""
    if job.get("skills"):
        skills_line = "Required/mentioned skills: " + ", ".join(job["skills"]) + "."

    similar_jobs_text = ""
    if similar_jobs:
        similar_jobs_text = "\n\n--- Similar job listings (for keyword/requirement context) ---\n"
        for i, r in enumerate(similar_jobs[:3], 1):
            doc = (r.get("document") or "")[:800]
            if doc:
                similar_jobs_text += f"\n[Similar job {i}]\n{doc}\n"

    similar_resumes_text = ""
    if similar_resumes:
        similar_resumes_text = "\n\n--- Example profiles/resumes similar to this role (for style/level) ---\n"
        for i, r in enumerate(similar_resumes[:3], 1):
            doc = (r.get("document") or "")[:600]
            if doc:
                similar_resumes_text += f"\n[Example profile {i}]\n{doc}\n"

    return f"""You are a resume improvement coach for Career Copilot. The user is applying to a specific job. Your job is to:

1. Improve their resume to match the job description and company.
2. Identify missing skills or keywords they should add.
3. Suggest concrete bullet points and section rewrites.
4. When asked, give an ATS (Applicant Tracking System) compatibility score and tips.
5. In follow-up messages, do exactly what the user asks: e.g. "shorten bullet 2", "add more ML pipeline experience", "rewrite summary".

Use the following context. The user's current resume and the target job description are the primary inputs; similar jobs and example profiles are for extra keyword and style context only.

--- Target role and company ---
{company_info}
{skills_line}

--- Job description (excerpt) ---
{job_desc}

--- User's current resume ---
{resume_text or "(No resume uploaded yet.)"}
{similar_jobs_text}
{similar_resumes_text}

Respond in clear, helpful markdown. When suggesting changes, show before/after or explicit bullet text the user can paste. For ATS score, give a number 0-100 and 2-3 concrete improvements."""


def get_initial_resume_analysis(
    resume_text: str,
    job: dict[str, Any],
    similar_jobs: list[dict],
    similar_resumes: list[dict],
) -> str:
    """
    First message from the bot: analyze resume vs job, missing skills, suggested bullets, ATS score.
    """
    if not job:
        return "I couldn't load the job details. Please go back and select a job."
    client = _get_openai_client()
    system = _build_system_prompt(resume_text, job, similar_jobs, similar_resumes)
    user_message = """Please analyze my resume for this role and give me:
1. **Missing skills or keywords** I should add (from the job description).
2. **2–3 suggested bullet points** I could add or adapt for this role.
3. **One section you'd rewrite** (e.g. summary or a job entry) — show before and after.
4. **ATS score** (0–100) and 2–3 quick tips to improve ATS compatibility.

Keep it concise and actionable."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    msg = response.choices[0].message
    return (msg.content or "").strip()


def chat_resume_improvement(
    user_message: str,
    conversation_history: list[dict[str, str]],
    resume_text: str,
    job: dict[str, Any],
    similar_jobs: list[dict],
    similar_resumes: list[dict],
) -> str:
    """
    One conversational turn: user says e.g. "shorten bullet 2", "add more ML experience", "rewrite summary".
    Returns the assistant reply.
    """
    if not job:
        return "I don't have the job context. Please start from the job page again."
    if not (user_message or "").strip():
        return "Please type a message (e.g. 'shorten bullet 2', 'add more ML pipeline experience', 'rewrite my summary')."

    client = _get_openai_client()
    system = _build_system_prompt(resume_text, job, similar_jobs, similar_resumes)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for h in conversation_history:
        role = h.get("role")
        content = h.get("content") or ""
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message.strip()})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
    )
    msg = response.choices[0].message
    return (msg.content or "").strip()
