"""
Resume Improvement Chatbot Agent (Chatbot Agent #1).

RAG pipeline: job description, similar jobs, similar resumes (user profiles).
LLM: improve resume to match job, suggest bullets, rewrites, ATS score; then conversational follow-ups.
"""

from __future__ import annotations

import json
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
    try:
        similar_jobs = get_similar_jobs_for_resume_improvement(job_document, n_results=5)
    except Exception:
        similar_jobs = []
    try:
        similar_resumes = get_similar_resumes_for_resume_improvement(
            job_document, exclude_user_id=user_id, n_results=5
        )
    except Exception:
        similar_resumes = []

    return {
        "resume_text": resume_text,
        "job": job,
        "job_document": job_document,
        "similar_jobs": similar_jobs,
        "similar_resumes": similar_resumes,
    }


def build_resume_improvement_context_from_job_dict(
    job: dict[str, Any],
    user_id: int,
    conn: Any,
) -> dict[str, Any]:
    """
    Build same context as build_resume_improvement_context but from an existing job dict
    (e.g. from user_jobs). Loads resume from profile and RAG similar jobs/resumes.
    """
    resume_text = ""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT resume_file, resume_filename FROM profiles WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        if row and row[0]:
            from career_copilot.resume_io import extract_resume_text

            resume_text = extract_resume_text(row[0], row[1]) or ""

    job_document = _job_dict_to_document(job)
    try:
        similar_jobs = get_similar_jobs_for_resume_improvement(job_document, n_results=5)
    except Exception:
        similar_jobs = []
    try:
        similar_resumes = get_similar_resumes_for_resume_improvement(
            job_document, exclude_user_id=user_id, n_results=5
        )
    except Exception:
        similar_resumes = []
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
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env for the resume improvement chatbot."
        )
    from openai import OpenAI

    return OpenAI()


def _build_system_prompt(
    resume_text: str,
    job: dict[str, Any],
    similar_jobs: list[dict],
    similar_resumes: list[dict],
) -> str:
    """Build the system prompt for the resume improvement agent."""
    company_info = (
        f"Company: {job.get('company') or 'Unknown'}. Role: {job.get('title') or 'Position'}."
    )
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
        similar_resumes_text = (
            "\n\n--- Example profiles/resumes similar to this role (for style/level) ---\n"
        )
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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_more_similar_jobs",
                "description": (
                    "Fetch additional similar job listings from the RAG store. "
                    "Use this if you need more examples of how other companies "
                    "describe similar roles."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_more_similar_resumes",
                "description": (
                    "Fetch additional example resumes/profiles similar to this role "
                    "from the RAG store. Use this when you want more style/level "
                    "examples to base your suggestions on."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    job_document = _job_dict_to_document(job)

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
            )
            for tool_call in tool_calls:
                name = tool_call.function.name
                if name == "get_more_similar_jobs":
                    result = get_similar_jobs_for_resume_improvement(job_document, n_results=5)
                elif name == "get_more_similar_resumes":
                    # Exclude current user if present in metadata; user id is not
                    # strictly required for usefulness here.
                    result = get_similar_resumes_for_resume_improvement(
                        job_document, exclude_user_id=None, n_results=5
                    )
                else:
                    result = {"error": f"Unknown tool {name}"}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
            continue

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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_more_similar_jobs",
                "description": (
                    "Fetch additional similar job listings from the RAG store. "
                    "Use this if you need more examples of how other companies "
                    "describe similar roles."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_more_similar_resumes",
                "description": (
                    "Fetch additional example resumes/profiles similar to this role "
                    "from the RAG store. Use this when you want more style/level "
                    "examples to base your suggestions on."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for h in conversation_history:
        role = h.get("role")
        content = h.get("content") or ""
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message.strip()})

    job_document = _job_dict_to_document(job)

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
            )
            for tool_call in tool_calls:
                name = tool_call.function.name
                if name == "get_more_similar_jobs":
                    result = get_similar_jobs_for_resume_improvement(job_document, n_results=5)
                elif name == "get_more_similar_resumes":
                    result = get_similar_resumes_for_resume_improvement(
                        job_document, exclude_user_id=None, n_results=5
                    )
                else:
                    result = {"error": f"Unknown tool {name}"}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
            continue

        return (msg.content or "").strip()


def format_resume_via_mcp(
    improved_text: str,
    style_profile_json: str,
    output_format: str,
    mcp_server_url: str,
) -> bytes:
    """
    Use the OpenAI Responses API with the remote MCP server to format an improved resume.

    output_format: "pdf" or "docx"
    Returns raw file bytes.
    Raises RuntimeError if the MCP tool does not return a result.
    """
    import base64

    tool_name = "generate_pdf_tool" if output_format == "pdf" else "generate_docx_tool"
    client = _get_openai_client()

    response = client.responses.create(
        model="gpt-4o-mini",
        tools=[
            {
                "type": "mcp",
                "server_label": "resume-formatter",
                "server_url": mcp_server_url,
                "require_approval": "never",
            }
        ],
        input=(
            f"Call the {tool_name} tool with the following arguments.\n\n"
            f"improved_text:\n{improved_text}\n\n"
            f"style_profile_json:\n{style_profile_json}"
        ),
    )

    for item in response.output:
        item_type = getattr(item, "type", "")
        if item_type == "mcp_call":
            name = getattr(item, "name", "")
            if name == tool_name:
                raw = getattr(item, "output", None)
                print(f"[MCP] {tool_name} output type={type(raw).__name__} len={len(raw) if raw else 0}")
                if isinstance(raw, str) and raw:
                    return base64.b64decode(raw)

    output_types = [(getattr(i, "type", "?"), getattr(i, "name", "")) for i in response.output]
    raise RuntimeError(
        f"MCP tool '{tool_name}' returned no result. Output: {output_types}"
    )


def generate_full_resume(
    conversation_history: list[dict[str, str]],
    resume_text: str,
    job: dict[str, Any],
    similar_jobs: list[dict],
    similar_resumes: list[dict],
) -> str:
    """
    Generate a full, updated resume that matches what the user saw in the chat.
    Uses the full conversation (including your "After:" rewrites) so the PDF
    reproduces the agreed wording, not a new variant.
    """
    if not job:
        return "I don't have the job context. Please start from the job page again."

    client = _get_openai_client()
    system = _build_system_prompt(resume_text, job, similar_jobs, similar_resumes)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for h in conversation_history:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": (
                "Using the conversation above, output the complete updated resume as formatted plain text.\n\n"
                'Critical: Where you already gave specific text in your replies (e.g. under "After:", '
                "rewritten bullets, or suggested wording), use that exact wording in the resume so "
                "the document matches what the user saw in the chat. Do not invent new wording for "
                "those sections.\n\n"
                "Formatting rules — follow exactly:\n"
                "1. Copy the name and contact block (name, location, phone, email, website) EXACTLY as they appear "
                "in the original resume — same line breaks, same separators (tabs, pipes, spaces), same order. "
                "Do not reformat, reorder, or combine these lines.\n"
                "2. Section headers (EXPERIENCE, EDUCATION, SKILLS, etc.) on their own line, no ** around them.\n"
                "3. Project sub-headers within a job (e.g. 'Blend Optimization and Inventory Management - Site, Location') "
                "must be preserved on their own line between the job title line and its bullets. Do not omit them.\n"
                "4. Bullet points start with - (dash space).\n"
                "5. Do NOT use ** anywhere in your output. Do not bold any text. Plain text only — "
                "bold formatting is applied separately and must not be added here.\n"
                "6. No # headings, no ``` code blocks, no other markdown.\n"
                "7. No commentary, analysis, ATS scores, or instructions — resume content only.\n"
                "8. Include ALL jobs, projects, education entries, and bullets from the original resume unless the user "
                "explicitly asked to remove something. Do not silently drop any section or sub-section."
            ),
        }
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.25,
    )
    msg = response.choices[0].message
    return (msg.content or "").strip()
