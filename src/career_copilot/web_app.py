from __future__ import annotations

from pathlib import Path
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from career_copilot.database.db import connect
from career_copilot.rag.chroma_store import get_recommended_job_results
from career_copilot.resume_io import extract_resume_text


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
        # One profile per user (resume stored as file bytes + filename)
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
                work_mode TEXT,
                employment_type TEXT,
                preferred_locations TEXT,
                salary_min INTEGER,
                salary_max INTEGER,
                resume_file BYTEA,
                resume_filename TEXT
            );
            """
        )
        # Commit so CREATE TABLE is persisted; then run migrations in separate transactions
        conn.commit()
        for sql in (
            "ALTER TABLE profiles ADD COLUMN resume_file BYTEA",
            "ALTER TABLE profiles ADD COLUMN resume_filename TEXT",
            "ALTER TABLE profiles DROP COLUMN IF EXISTS resume_text",
        ):
            try:
                cur.execute(sql)
                conn.commit()
            except psycopg.ProgrammingError as e:
                conn.rollback()
                if e.sqlstate != "42701":  # duplicate_column
                    raise
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_skills (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                skill TEXT NOT NULL
            );
            """
        )
        # Embeddings live only in Chroma (user_profiles collection); no vector storage in Postgres
        cur.execute("DROP TABLE IF EXISTS user_embeddings")

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
    resume_file: bytes | None,
    resume_filename: str | None,
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
                resume_file,
                resume_filename
            )
            VALUES (%(user_id)s, %(skill_tags)s, %(years_experience)s, %(current_location)s,
                    %(preferred_roles)s, %(industries)s, %(work_mode)s, %(employment_type)s,
                    %(preferred_locations)s, %(salary_min)s, %(salary_max)s, %(resume_file)s, %(resume_filename)s)
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
                resume_file = EXCLUDED.resume_file,
                resume_filename = EXCLUDED.resume_filename;
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
                "resume_file": resume_file,
                "resume_filename": resume_filename or None,
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

    Uses OpenAI text-embedding-3-large (same as jobs; see career_copilot.rag.embedding).
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

    # Resume is first so it is a central part of the user embedding; then preferences
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


def _norm_sid(sid: str | int | float | None) -> str | None:
    """Normalize source_id to str for consistent map keys."""
    if sid is None:
        return None
    if isinstance(sid, str):
        return sid or None
    return str(int(sid))


def _chroma_id_to_source_source_id(chroma_id: str) -> tuple[str | None, str | None]:
    """Parse Chroma doc id 'source:source_id' into (source, source_id)."""
    if ":" in chroma_id:
        a, b = chroma_id.split(":", 1)
        return (a or None, b or None)
    return (chroma_id or None, None)


def _resolve_job_ids(
    conn: psycopg.Connection,
    results: list[dict],
) -> dict[tuple[str | None, str | None], int]:
    """Map (source, source_id) to Postgres job id for each result. Returns dict keyed by (source, source_id) -> id."""
    if not results:
        return {}
    pairs: list[tuple[str | None, str | None]] = []
    for r in results:
        meta = r.get("metadata") or {}
        src = meta.get("source")
        sid = _norm_sid(meta.get("source_id"))
        if src is None and sid is None:
            chroma_id = r.get("id") or ""
            src, sid = _chroma_id_to_source_source_id(chroma_id)
        pairs.append((src, sid))
    pairs = list(dict.fromkeys(pairs))  # unique
    out: dict[tuple[str | None, str | None], int] = {}
    with conn.cursor() as cur:
        for (src, sid) in pairs:
            if src is None:
                continue
            cur.execute(
                "SELECT id FROM jobs WHERE source = %s AND source_id = %s",
                (src, sid),
            )
            row = cur.fetchone()
            if row:
                out[(src, sid)] = int(row[0])
    return out


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return RedirectResponse(url="/profile", status_code=303)


@app.get("/recommendations", response_class=HTMLResponse)
async def get_recommendations(
    request: Request,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> HTMLResponse:
    """Candidate retrieval: top 100 jobs by similarity to current user's profile embedding."""
    user_id = 1
    raw = get_recommended_job_results(user_id=user_id, n_results=100)
    id_map = _resolve_job_ids(conn, raw)

    jobs_for_template: list[dict] = []
    for r in raw:
        meta = r.get("metadata") or {}
        src = meta.get("source")
        sid = _norm_sid(meta.get("source_id"))
        postgres_id = id_map.get((src, sid)) if src is not None else None
        doc = (r.get("document") or "")[:400]
        if len(r.get("document") or "") > 400:
            doc = doc.rstrip() + "…"
        jobs_for_template.append({
            "job_id": postgres_id,
            "title": meta.get("title") or "Job",
            "company": meta.get("company") or "",
            "location": meta.get("location") or "",
            "url": meta.get("url") or "",
            "snippet": doc,
            "distance": r.get("distance"),
            "salary_min": meta.get("salary_min"),
            "salary_max": meta.get("salary_max"),
            "skills": (meta.get("skills") or "").split(",") if meta.get("skills") else [],
        })

    return templates.TemplateResponse(
        "recommendations.html",
        {"request": request, "jobs": jobs_for_template, "user_id": user_id},
    )


@app.get("/jobs/{job_id:int}", response_class=HTMLResponse, response_model=None)
async def get_job_detail(
    request: Request,
    job_id: int,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> HTMLResponse | RedirectResponse:
    """Full job details for a single job (by Postgres id)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source, source_id, title, company, location,
                   salary_min, salary_max, description, skills,
                   posted_at, url
            FROM jobs
            WHERE id = %s
            """,
            (job_id,),
        )
        row = cur.fetchone()
    conn.close()
    if not row:
        return RedirectResponse(url="/recommendations", status_code=303)
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
    return templates.TemplateResponse(
        "job_detail.html",
        {"request": request, "job": job, "user_id": 1},
    )


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
                p.resume_filename
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


@app.get("/profile/resume")
async def get_resume(conn: Annotated[psycopg.Connection, Depends(get_db)]) -> Response:
    """Download the current user's stored resume file."""
    user_id = 1
    with conn.cursor() as cur:
        cur.execute(
            "SELECT resume_file, resume_filename FROM profiles WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    conn.close()
    if not row or row[0] is None:
        return Response(status_code=404)
    data = bytes(row[0])
    filename = (row[1] or "resume").replace('"', "")
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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

    content_bytes: bytes | None = None
    resume_filename: str | None = None
    if resume_file is not None and resume_file.filename:
        content_bytes = await resume_file.read()
        resume_filename = resume_file.filename.strip() or None
        if not content_bytes:
            content_bytes = None
            resume_filename = None

    # PostgreSQL TEXT cannot contain NUL bytes; strip from all text inputs
    skill_tags = _strip_nul(skill_tags or "")
    current_location = _strip_nul(current_location or "")
    preferred_roles = _strip_nul(preferred_roles or "")
    industries = _strip_nul(industries or "")
    work_mode = _strip_nul(work_mode or "")
    employment_type = _strip_nul(employment_type or "")
    preferred_locations = _strip_nul(preferred_locations or "")

    resume_text_for_embedding = ""
    if content_bytes:
        resume_text_for_embedding = _strip_nul(extract_resume_text(content_bytes, resume_filename))

    conn = get_db()
    try:
        # If no new file uploaded, keep existing resume in DB (fetch current)
        if content_bytes is None:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT resume_file, resume_filename FROM profiles WHERE user_id = %s",
                    (user_id,),
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    content_bytes = row[0]
                    resume_filename = row[1]
                # else leave content_bytes=None, resume_filename=None to clear or leave as-is
                # Actually if row is None (no profile yet), we pass None, None. If row exists but resume_file is None, we pass None, None. So we need to pass (content_bytes, resume_filename) which might be (existing_bytes, existing_name) when no new upload.
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
            resume_file=content_bytes,
            resume_filename=resume_filename,
        )

        # Re-extract text for embedding when we have file bytes (either new upload or kept existing)
        if content_bytes and not resume_text_for_embedding:
            resume_text_for_embedding = _strip_nul(extract_resume_text(content_bytes, resume_filename))
        index_user_embedding(
            user_id=user_id,
            resume_text=resume_text_for_embedding,
            skill_tags=skill_tags,
            preferred_roles=preferred_roles,
            industries=industries,
            work_mode=work_mode,
            employment_type=employment_type,
            preferred_locations=preferred_locations,
        )

        conn.commit()
    finally:
        conn.close()

    return RedirectResponse(url="/profile", status_code=303)

