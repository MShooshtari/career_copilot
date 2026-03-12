from __future__ import annotations

from pathlib import Path
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from career_copilot.database.db import connect


def _strip_nul(s: str) -> str:
    """Remove NUL bytes; PostgreSQL TEXT cannot store them."""
    return s.replace("\x00", "")


# OpenAI text-embedding-3-large max context is 8192 tokens; ~4 chars/token → safe limit
EMBEDDING_MAX_CHARS = 28_000


def _truncate_for_embedding(text: str, max_chars: int = EMBEDDING_MAX_CHARS) -> str:
    """Truncate text so it fits within the embedding model's token limit."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _extract_resume_text(content_bytes: bytes, filename: str | None) -> str:
    """Extract plain text from uploaded resume (PDF or UTF-8 text)."""
    if not content_bytes:
        return ""
    filename = (filename or "").lower()
    is_pdf = filename.endswith(".pdf") or content_bytes.startswith(b"%PDF")
    if is_pdf:
        try:
            from pypdf import PdfReader
            from io import BytesIO
            reader = PdfReader(BytesIO(content_bytes))
            parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            return "\n\n".join(parts) if parts else ""
        except Exception:
            return ""
    try:
        return content_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""


ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = ROOT / "templates"

app = FastAPI(title="Career Copilot - User Profile")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_db() -> psycopg.Connection:
    return connect()


def init_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        # Basic users table (demo-only; assume user_id=1 is logged in)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE
            );
            """
        )
        # One profile per user
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                skill_tags TEXT,
                years_experience INTEGER,
                current_location TEXT,
                preferred_roles TEXT,
                industries TEXT,
                work_mode TEXT,          -- remote / hybrid / onsite
                employment_type TEXT,    -- full_time / part_time
                preferred_locations TEXT,
                salary_min INTEGER,
                salary_max INTEGER,
                resume_text TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_skills (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                skill TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_embeddings (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                collection_name TEXT NOT NULL,
                document_id TEXT NOT NULL
            );
            """
        )

        # Ensure demo user exists
        cur.execute("INSERT INTO users (email) VALUES (%s) ON CONFLICT (email) DO NOTHING", ("demo@example.com",))
    conn.commit()


def upsert_user_profile(
    conn: psycopg.Connection,
    *,
    user_id: int,
    skill_tags: str,
    years_experience: int | None,
    current_location: str,
    preferred_roles: str,
    industries: str,
    work_mode: str,
    employment_type: str,
    preferred_locations: str,
    salary_min: int | None,
    salary_max: int | None,
    resume_text: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM user_skills WHERE user_id = %s", (user_id,))
        for raw in skill_tags.split(","):
            skill = raw.strip()
            if not skill:
                continue
            cur.execute(
                "INSERT INTO user_skills (user_id, skill) VALUES (%s, %s)",
                (user_id, skill),
            )

        cur.execute(
            """
            INSERT INTO profiles (
                user_id,
                skill_tags,
                years_experience,
                current_location,
                preferred_roles,
                industries,
                work_mode,
                employment_type,
                preferred_locations,
                salary_min,
                salary_max,
                resume_text
            )
            VALUES (%(user_id)s, %(skill_tags)s, %(years_experience)s, %(current_location)s,
                    %(preferred_roles)s, %(industries)s, %(work_mode)s, %(employment_type)s,
                    %(preferred_locations)s, %(salary_min)s, %(salary_max)s, %(resume_text)s)
            ON CONFLICT (user_id) DO UPDATE
            SET
                skill_tags = EXCLUDED.skill_tags,
                years_experience = EXCLUDED.years_experience,
                current_location = EXCLUDED.current_location,
                preferred_roles = EXCLUDED.preferred_roles,
                industries = EXCLUDED.industries,
                work_mode = EXCLUDED.work_mode,
                employment_type = EXCLUDED.employment_type,
                preferred_locations = EXCLUDED.preferred_locations,
                salary_min = EXCLUDED.salary_min,
                salary_max = EXCLUDED.salary_max,
                resume_text = EXCLUDED.resume_text;
            """,
            {
                "user_id": user_id,
                "skill_tags": skill_tags,
                "years_experience": years_experience,
                "current_location": current_location,
                "preferred_roles": preferred_roles,
                "industries": industries,
                "work_mode": work_mode,
                "employment_type": employment_type,
                "preferred_locations": preferred_locations,
                "salary_min": salary_min,
                "salary_max": salary_max,
                "resume_text": resume_text,
            },
        )
    conn.commit()


def index_user_embedding(
    *,
    user_id: int,
    resume_text: str,
    skill_tags: str,
    preferred_roles: str,
    industries: str,
    work_mode: str,
    employment_type: str,
    preferred_locations: str,
) -> tuple[str, str]:
    """
    Store the user's profile embedding in a Chroma collection.

    Returns (collection_name, document_id).
    """
    import chromadb

    from career_copilot.rag.embedding import get_embedding_function

    root = Path(__file__).resolve().parents[2]
    persist_path = root / "data" / "chroma"
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    collection_name = "user_profiles"
    ef = get_embedding_function()
    try:
        coll = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Career Copilot user profiles"},
            embedding_function=ef,
        )
    except ValueError as e:
        if "embedding function" in str(e).lower() and "conflict" in str(e).lower():
            client.delete_collection(name=collection_name)
            coll = client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Career Copilot user profiles"},
                embedding_function=ef,
            )
        else:
            raise

    pieces = [
        resume_text or "",
        f"Skills: {skill_tags}",
        f"Preferred roles: {preferred_roles}",
        f"Industries: {industries}",
        f"Work mode: {work_mode}",
        f"Employment type: {employment_type}",
        f"Preferred locations: {preferred_locations}",
    ]
    document = "\n\n".join(p for p in pieces if p.strip())
    document = _truncate_for_embedding(document)
    document_id = f"user:{user_id}"

    coll.upsert(ids=[document_id], documents=[document], metadatas=[{"user_id": user_id}])
    return collection_name, document_id


@app.on_event("startup")
def _startup() -> None:
    conn = get_db()
    try:
        init_schema(conn)
    finally:
        conn.close()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return RedirectResponse(url="/profile", status_code=303)


@app.get("/profile", response_class=HTMLResponse)
async def get_profile(request: Request, conn: Annotated[psycopg.Connection, Depends(get_db)]) -> HTMLResponse:
    user_id = 1
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                p.skill_tags,
                p.years_experience,
                p.current_location,
                p.preferred_roles,
                p.industries,
                p.work_mode,
                p.employment_type,
                p.preferred_locations,
                p.salary_min,
                p.salary_max,
                p.resume_text
            FROM profiles p
            WHERE p.user_id = %s
            """,
            (user_id,),
        )
        row = cur.fetchone()
    conn.close()

    context = {
        "request": request,
        "profile": row,
        "user_id": user_id,
    }
    return templates.TemplateResponse("profile.html", context)


@app.post("/profile", response_class=HTMLResponse)
async def post_profile(
    request: Request,
    skill_tags: str = Form(...),
    years_experience: int | None = Form(None),
    current_location: str = Form(""),
    preferred_roles: str = Form(""),
    industries: str = Form(""),
    work_mode: str = Form(""),
    employment_type: str = Form(""),
    preferred_locations: str = Form(""),
    salary_min: int | None = Form(None),
    salary_max: int | None = Form(None),
    resume_file: UploadFile | None = File(None),
) -> HTMLResponse:
    user_id = 1

    resume_text = ""
    if resume_file is not None:
        content_bytes = await resume_file.read()
        resume_text = _extract_resume_text(content_bytes, resume_file.filename)

    # PostgreSQL TEXT cannot contain NUL bytes; strip from all text inputs
    skill_tags = _strip_nul(skill_tags or "")
    current_location = _strip_nul(current_location or "")
    preferred_roles = _strip_nul(preferred_roles or "")
    industries = _strip_nul(industries or "")
    work_mode = _strip_nul(work_mode or "")
    employment_type = _strip_nul(employment_type or "")
    preferred_locations = _strip_nul(preferred_locations or "")
    resume_text = _strip_nul(resume_text)

    conn = get_db()
    try:
        upsert_user_profile(
            conn,
            user_id=user_id,
            skill_tags=skill_tags,
            years_experience=years_experience,
            current_location=current_location,
            preferred_roles=preferred_roles,
            industries=industries,
            work_mode=work_mode,
            employment_type=employment_type,
            preferred_locations=preferred_locations,
            salary_min=salary_min,
            salary_max=salary_max,
            resume_text=resume_text,
        )

        collection_name, document_id = index_user_embedding(
            user_id=user_id,
            resume_text=resume_text,
            skill_tags=skill_tags,
            preferred_roles=preferred_roles,
            industries=industries,
            work_mode=work_mode,
            employment_type=employment_type,
            preferred_locations=preferred_locations,
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_embeddings (user_id, collection_name, document_id)
                VALUES (%s, %s, %s)
                """,
                (user_id, collection_name, document_id),
            )
        conn.commit()
    finally:
        conn.close()

    return RedirectResponse(url="/profile", status_code=303)

