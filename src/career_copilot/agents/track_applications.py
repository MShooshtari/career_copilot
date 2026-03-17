"""
Track Applications Agent: conversational agent with tool calling to list, add,
and manage applications (resume improvement or interview preparation for a job).
"""

from __future__ import annotations

import json
from typing import Any

from career_copilot.database.applications import (
    add_application as db_add_application,
)
from career_copilot.database.applications import (
    application_row_to_dict,
    enrich_applications_with_job_info,
)
from career_copilot.database.applications import (
    get_application as db_get_application,
)
from career_copilot.database.applications import (
    list_applications as db_list_applications,
)
from career_copilot.database.applications import (
    remove_application as db_remove_application,
)
from career_copilot.database.jobs import (
    list_ingested_jobs_snippet,
    list_user_jobs,
)


def _get_openai_client():
    import os

    from career_copilot.database.db import load_env

    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env for the track applications agent."
        )
    from openai import OpenAI

    return OpenAI()


TRACK_APPLICATIONS_SYSTEM = """You are an application-tracking assistant for Career Copilot. The user tracks "applications" — each application is either:
1. **Resume improvement** for a specific job (user is tailoring their resume to that role)
2. **Interview preparation** for a specific job (user is preparing for an interview)

You have tools to:
- **list_applications** — Show the user's tracked applications (optional filters: stage, status). Use this when they ask "what are my applications?", "show resume improvements", "what am I preparing for?", etc.
- **list_available_jobs** — List jobs the user can start an application for (user-added jobs and recently ingested jobs). Use when the user wants to add an application and you need to find a job (e.g. "add interview prep for Google" → list jobs, find Google, then add_application).
- **add_application** — Start tracking an application: choose job_id, job_source ('user' or 'ingested'), and stage ('resume_improvement' or 'interview_preparation'). After adding, tell the user the action_url so they can open the resume or interview prep page.
- **get_application** — Get one application by id (e.g. to show details or confirm before removing).
- **remove_application** — Stop tracking an application by id.

When the user says they want to "add" or "start" resume improvement or interview prep for a job, use list_available_jobs to find the right job (match by company name or title), then call add_application with that job_id and job_source. Always confirm the job (title, company) before adding. If the user's request is ambiguous (e.g. "add one for Google" and there are multiple Google jobs), list the options and ask which one, or pick the most recent match.

Respond in clear, concise language. After tool calls, summarize what you did and give the user the link (action_url) when you add an application."""


def _tool_list_applications(
    conn: Any,
    user_id: int,
    stage: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    rows = db_list_applications(conn, user_id, stage=stage, status=status)
    enriched = enrich_applications_with_job_info(conn, rows)
    return {
        "applications": enriched,
        "count": len(enriched),
    }


def _tool_list_available_jobs(conn: Any, user_id: int) -> dict[str, Any]:
    user_rows = list_user_jobs(conn, user_id)
    user_jobs = [
        {
            "job_id": r[0],
            "job_source": "user",
            "title": r[2] or "Job",
            "company": r[3] or "",
        }
        for r in user_rows
    ]
    ingested = list_ingested_jobs_snippet(conn, limit=40)
    ingested_jobs = [
        {
            "job_id": id_,
            "job_source": "ingested",
            "title": title,
            "company": company,
        }
        for id_, title, company in ingested
    ]
    return {
        "user_jobs": user_jobs,
        "ingested_jobs": ingested_jobs,
        "all": user_jobs + ingested_jobs,
    }


def _tool_add_application(
    conn: Any,
    user_id: int,
    job_id: int,
    job_source: str,
    stage: str,
) -> dict[str, Any]:
    if job_source not in ("ingested", "user") or stage not in (
        "resume_improvement",
        "interview_preparation",
    ):
        return {"success": False, "error": "Invalid job_source or stage."}
    app_id = db_add_application(conn, user_id, job_id, job_source, stage)
    if not app_id:
        return {"success": False, "error": "Could not add application (duplicate or invalid)."}
    conn.commit()
    row = db_get_application(conn, user_id, app_id)
    if not row:
        return {"success": True, "application_id": app_id, "action_url": None}
    enriched = enrich_applications_with_job_info(conn, [row])
    app = enriched[0] if enriched else {}
    return {
        "success": True,
        "application_id": app_id,
        "job_title": app.get("job_title"),
        "job_company": app.get("job_company"),
        "stage": stage,
        "action_url": app.get("action_url"),
    }


def _tool_get_application(conn: Any, user_id: int, application_id: int) -> dict[str, Any]:
    row = db_get_application(conn, user_id, application_id)
    if not row:
        return {"found": False, "application": None}
    enriched = enrich_applications_with_job_info(conn, [row])
    return {"found": True, "application": enriched[0] if enriched else application_row_to_dict(row)}


def _tool_remove_application(conn: Any, user_id: int, application_id: int) -> dict[str, Any]:
    ok = db_remove_application(conn, user_id, application_id)
    if ok:
        conn.commit()
    return {"removed": ok, "application_id": application_id}


TRACK_APPLICATIONS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_applications",
            "description": "List the user's tracked applications (resume improvement or interview preparation for jobs). Optionally filter by stage or status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "enum": ["resume_improvement", "interview_preparation"],
                        "description": "Filter by stage: resume_improvement or interview_preparation",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "done"],
                        "description": "Filter by status",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_jobs",
            "description": "List jobs the user can start an application for: user-added jobs and recently ingested jobs. Use this to find job_id and job_source when the user wants to add resume improvement or interview prep for a job (e.g. 'add interview prep for Google').",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_application",
            "description": "Start tracking an application: resume improvement or interview preparation for a specific job. Use list_available_jobs first to get job_id and job_source. Then call this with that job_id, job_source ('user' or 'ingested'), and stage ('resume_improvement' or 'interview_preparation').",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "The job id (from list_available_jobs)"},
                    "job_source": {
                        "type": "string",
                        "enum": ["user", "ingested"],
                        "description": "Whether the job is from user-added list ('user') or ingested recommendations ('ingested')",
                    },
                    "stage": {
                        "type": "string",
                        "enum": ["resume_improvement", "interview_preparation"],
                        "description": "Type of application: resume_improvement or interview_preparation",
                    },
                },
                "required": ["job_id", "job_source", "stage"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_application",
            "description": "Get details of one application by its id. Use after list_applications when the user asks about a specific application.",
            "parameters": {
                "type": "object",
                "properties": {
                    "application_id": {"type": "integer", "description": "The application id"},
                },
                "required": ["application_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_application",
            "description": "Stop tracking an application (remove it from the list) by its id. Use when the user says to remove, delete, or stop tracking an application.",
            "parameters": {
                "type": "object",
                "properties": {
                    "application_id": {"type": "integer", "description": "The application id to remove"},
                },
                "required": ["application_id"],
            },
        },
    },
]


def chat_track_applications(
    message: str,
    history: list[dict[str, str]],
    conn: Any,
    user_id: int,
) -> str:
    """
    Run the track-applications agent with tool calling. Returns the assistant reply.
    """
    client = _get_openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": TRACK_APPLICATIONS_SYSTEM},
    ]
    for h in history:
        role = h.get("role") or "user"
        content = h.get("content") or h.get("message") or ""
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": message or "What are my applications?"})

    max_steps = 12
    for _ in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TRACK_APPLICATIONS_TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            return (msg.content or "").strip()

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

            if name == "list_applications":
                result = _tool_list_applications(
                    conn,
                    user_id,
                    stage=args.get("stage"),
                    status=args.get("status"),
                )
            elif name == "list_available_jobs":
                result = _tool_list_available_jobs(conn, user_id)
            elif name == "add_application":
                result = _tool_add_application(
                    conn,
                    user_id,
                    job_id=int(args.get("job_id") or 0),
                    job_source=(args.get("job_source") or "user").strip(),
                    stage=(args.get("stage") or "resume_improvement").strip(),
                )
            elif name == "get_application":
                result = _tool_get_application(
                    conn,
                    user_id,
                    application_id=int(args.get("application_id") or 0),
                )
            elif name == "remove_application":
                result = _tool_remove_application(
                    conn,
                    user_id,
                    application_id=int(args.get("application_id") or 0),
                )
            else:
                result = {"error": f"Unknown tool: {name}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    return "I hit the step limit. Please try a shorter request or ask again."
