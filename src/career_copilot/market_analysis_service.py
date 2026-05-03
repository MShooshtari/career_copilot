"""
Market analysis: SQL-filtered cohort, ranked by user profile embedding, aggregates + chunk RAG.
"""

from __future__ import annotations

import copy
import hashlib
import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg

try:
    from pgvector.psycopg import register_vector
except ModuleNotFoundError:  # pragma: no cover

    def register_vector(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError("pgvector is required (pip install pgvector)")


from career_copilot.constants import (
    MARKET_ANALYSIS_DEFAULT_POSTED_WITHIN_DAYS,
    MARKET_ANALYSIS_FALLBACK_COHORT_LIMIT,
    MARKET_ANALYSIS_RAG_EMBEDDING_CACHE_TTL_SECONDS,
    MARKET_ANALYSIS_RAG_TOP_CHUNKS,
    MARKET_ANALYSIS_REPORT_CACHE_TTL_SECONDS,
    MARKET_ANALYSIS_TOP_LOCATIONS,
    MARKET_ANALYSIS_TOP_SKILLS_CHART,
)
from career_copilot.database.profiles import get_profile_by_user_id, list_user_skills_lower
from career_copilot.ingestion.skill_extraction import normalize_skill_tag, skill_specificity_score
from career_copilot.rag.embedding import OPENAI_API_KEY_ENV, embed_texts
from career_copilot.rag.pgvector_rag import fetch_user_profile_embedding


@dataclass
class MarketCohortFilters:
    posted_within_days: int = MARKET_ANALYSIS_DEFAULT_POSTED_WITHIN_DAYS
    location_contains: str | None = None
    title_contains: str | None = None
    source_equals: str | None = None
    remote_only: bool = False
    salary_at_least: int | None = None


class _TTLCache:
    def __init__(
        self,
        *,
        ttl_seconds: int,
        max_entries: int,
        clone=copy.deepcopy,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clone = clone
        self._items: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        now = time.time()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if now >= expires_at:
                self._items.pop(key, None)
                return None
            return self._clone(value)

    def set(self, key: Any, value: Any) -> None:
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            if len(self._items) >= self._max_entries and key not in self._items:
                oldest_key = min(self._items, key=lambda k: self._items[k][0])
                self._items.pop(oldest_key, None)
            self._items[key] = (now + self._ttl_seconds, self._clone(value))

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _purge_expired(self, now: float) -> None:
        expired = [key for key, (expires_at, _value) in self._items.items() if now >= expires_at]
        for key in expired:
            self._items.pop(key, None)


_market_analysis_report_cache = _TTLCache(
    ttl_seconds=MARKET_ANALYSIS_REPORT_CACHE_TTL_SECONDS,
    max_entries=128,
)
_rag_query_embedding_cache = _TTLCache(
    ttl_seconds=MARKET_ANALYSIS_RAG_EMBEDDING_CACHE_TTL_SECONDS,
    max_entries=256,
    clone=list,
)


def clear_market_analysis_caches() -> None:
    _market_analysis_report_cache.clear()
    _rag_query_embedding_cache.clear()


def _profile_version(profile_row: tuple | None, skills: list[str]) -> str:
    payload = repr((tuple(profile_row) if profile_row else None, tuple(sorted(skills))))
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _cutoff_utc(days: int) -> datetime:
    return datetime.now(tz=UTC) - timedelta(days=max(1, days))


def _normalized_filter_value(value: str | None, *, case_insensitive: bool = False) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized.lower() if case_insensitive else normalized


def _market_analysis_cache_key(
    *,
    user_id: int,
    filters: MarketCohortFilters,
    cohort_limit: int,
    include_rag: bool,
) -> tuple[Any, ...]:
    return (
        int(user_id),
        max(1, int(filters.posted_within_days)),
        _normalized_filter_value(filters.location_contains, case_insensitive=True),
        _normalized_filter_value(filters.title_contains, case_insensitive=True),
        _normalized_filter_value(filters.source_equals),
        bool(filters.remote_only),
        filters.salary_at_least,
        max(1, min(int(cohort_limit), 5_000)),
        bool(include_rag),
    )


def _register(conn: psycopg.Connection) -> None:
    register_vector(conn)


def count_filtered_jobs(
    conn: psycopg.Connection,
    filters: MarketCohortFilters,
) -> int:
    """Jobs matching SQL filters and having a row in jobs_embeddings."""
    _register(conn)
    cutoff = _cutoff_utc(filters.posted_within_days)
    conds = [
        "j.posted_at IS NULL OR j.posted_at >= %s",
        "EXISTS (SELECT 1 FROM jobs_embeddings e WHERE e.job_id = j.id)",
    ]
    params: list[Any] = [cutoff]
    if filters.location_contains and filters.location_contains.strip():
        conds.append("j.location ILIKE %s")
        params.append(f"%{filters.location_contains.strip()}%")
    if filters.title_contains and filters.title_contains.strip():
        conds.append("j.title ILIKE %s")
        params.append(f"%{filters.title_contains.strip()}%")
    if filters.source_equals and filters.source_equals.strip():
        conds.append("j.source = %s")
        params.append(filters.source_equals.strip())
    if filters.remote_only:
        conds.append("(j.location ILIKE %s OR j.description ILIKE %s)")
        params.extend(["%remote%", "%remote%"])
    if filters.salary_at_least is not None:
        conds.append(
            "(j.salary_max IS NOT NULL AND j.salary_max >= %s) "
            "OR (j.salary_min IS NOT NULL AND j.salary_min >= %s)"
        )
        params.extend([filters.salary_at_least, filters.salary_at_least])

    where_sql = " AND ".join(conds)
    sql = f"SELECT count(*)::int FROM jobs j WHERE {where_sql}"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def cohort_job_ids(
    conn: psycopg.Connection,
    *,
    user_id: int,
    filters: MarketCohortFilters,
    cohort_limit: int,
) -> tuple[list[int], dict[str, Any]]:
    """
    Return job ids: SQL filters first, then ORDER BY embedding distance to the user's
    profile (if available), else by posted_at desc.
    """
    _register(conn)
    cutoff = _cutoff_utc(filters.posted_within_days)
    conds = [
        "j.posted_at IS NULL OR j.posted_at >= %s",
        "EXISTS (SELECT 1 FROM jobs_embeddings e WHERE e.job_id = j.id)",
    ]
    params: list[Any] = [cutoff]
    if filters.location_contains and filters.location_contains.strip():
        conds.append("j.location ILIKE %s")
        params.append(f"%{filters.location_contains.strip()}%")
    if filters.title_contains and filters.title_contains.strip():
        conds.append("j.title ILIKE %s")
        params.append(f"%{filters.title_contains.strip()}%")
    if filters.source_equals and filters.source_equals.strip():
        conds.append("j.source = %s")
        params.append(filters.source_equals.strip())
    if filters.remote_only:
        conds.append("(j.location ILIKE %s OR j.description ILIKE %s)")
        params.extend(["%remote%", "%remote%"])
    if filters.salary_at_least is not None:
        conds.append(
            "(j.salary_max IS NOT NULL AND j.salary_max >= %s) "
            "OR (j.salary_min IS NOT NULL AND j.salary_min >= %s)"
        )
        params.extend([filters.salary_at_least, filters.salary_at_least])

    where_sql = " AND ".join(conds)
    filtered_count = count_filtered_jobs(conn, filters)

    user_vec = fetch_user_profile_embedding(conn, user_id)
    lim = max(1, min(cohort_limit, 5_000))

    if user_vec:
        sql = f"""
            SELECT j.id
            FROM jobs j
            INNER JOIN jobs_embeddings e ON e.job_id = j.id
            WHERE {where_sql}
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        """
        qparams = params + [user_vec, lim]
        with conn.cursor() as cur:
            cur.execute(sql, qparams)
            rows = cur.fetchall()
        ids = [int(r[0]) for r in rows]
        meta = {
            "filtered_count": filtered_count,
            "used_vector_ranking": True,
            "user_has_embedding": True,
        }
        return ids, meta

    sql = f"""
        SELECT j.id
        FROM jobs j
        INNER JOIN jobs_embeddings e ON e.job_id = j.id
        WHERE {where_sql}
        ORDER BY j.posted_at DESC NULLS LAST, j.id DESC
        LIMIT %s
    """
    qparams = params + [min(lim, MARKET_ANALYSIS_FALLBACK_COHORT_LIMIT)]
    with conn.cursor() as cur:
        cur.execute(sql, qparams)
        rows = cur.fetchall()
    ids = [int(r[0]) for r in rows]
    meta = {
        "filtered_count": filtered_count,
        "used_vector_ranking": False,
        "user_has_embedding": False,
        "message": "Save your profile (with OpenAI key set) to rank this cohort by match to you.",
    }
    return ids, meta


def _aggregates_for_cohort(conn: psycopg.Connection, job_ids: list[int]) -> dict[str, Any]:
    if not job_ids:
        return {
            "weekly_posted": [],
            "top_skills": [],
            "salary": {},
            "top_locations": [],
        }

    _register(conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT date_trunc('week', COALESCE(j.posted_at, j.created_at))::date AS week_start,
                   count(*)::int AS cnt
            FROM jobs j
            WHERE j.id = ANY(%s::bigint[])
            GROUP BY 1
            ORDER BY 1
            """,
            (job_ids,),
        )
        weekly = [{"week_start": str(r[0]), "count": int(r[1])} for r in cur.fetchall()]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT lower(trim(u.skill)) AS sk, count(*)::int AS cnt
            FROM jobs j
            CROSS JOIN LATERAL unnest(
                COALESCE(
                    NULLIF(j.ai_extracted_skills, ARRAY[]::text[]),
                    NULLIF(j.extracted_skills, ARRAY[]::text[]),
                    NULLIF(j.skills, ARRAY[]::text[]),
                    ARRAY[]::text[]
                )
            ) AS u(skill)
            WHERE j.id = ANY(%s::bigint[])
              AND trim(u.skill) <> ''
            GROUP BY 1
            ORDER BY cnt DESC
            LIMIT %s
            """,
            (job_ids, MARKET_ANALYSIS_TOP_SKILLS_CHART * 20),
        )
        raw_top_skills = [(str(r[0]), int(r[1])) for r in cur.fetchall()]
    top_skills = _filtered_top_skills(raw_top_skills)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              count(*) FILTER (
                WHERE j.salary_min IS NOT NULL OR j.salary_max IS NOT NULL
              )::int AS n_any,
              count(*) FILTER (
                WHERE j.salary_min IS NOT NULL AND j.salary_max IS NOT NULL
              )::int AS n_both,
              min(j.salary_min) AS min_min,
              max(j.salary_max) AS max_max,
              avg((j.salary_min + j.salary_max) / 2.0) FILTER (
                WHERE j.salary_min IS NOT NULL AND j.salary_max IS NOT NULL
              ) AS avg_mid
            FROM jobs j
            WHERE j.id = ANY(%s::bigint[])
            """,
            (job_ids,),
        )
        row = cur.fetchone()
    salary_block: dict[str, Any] = {}
    if row:
        salary_block = {
            "jobs_with_any_salary": int(row[0] or 0),
            "jobs_with_min_and_max": int(row[1] or 0),
            "global_min_salary_min": row[2],
            "global_max_salary_max": row[3],
            "avg_mid_when_both": float(row[4]) if row[4] is not None else None,
        }

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              COALESCE(NULLIF(trim(j.location), ''), '(unspecified)') AS loc,
              count(*)::int AS cnt
            FROM jobs j
            WHERE j.id = ANY(%s::bigint[])
            GROUP BY 1
            ORDER BY cnt DESC
            LIMIT %s
            """,
            (job_ids, MARKET_ANALYSIS_TOP_LOCATIONS),
        )
        top_locations = [{"location": str(r[0]), "count": int(r[1])} for r in cur.fetchall()]

    return {
        "weekly_posted": weekly,
        "top_skills": top_skills,
        "salary": salary_block,
        "top_locations": top_locations,
    }


def _filtered_top_skills(raw_top_skills: list[tuple[str, int]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for raw_skill, count in raw_top_skills:
        normalized = normalize_skill_tag(raw_skill)
        if not normalized:
            continue
        key = normalized.casefold()
        counts[key] = counts.get(key, 0) + int(count)

    scored = []
    for skill, count in counts.items():
        specificity = skill_specificity_score(skill)
        if specificity <= 0:
            continue
        market_score = (count**0.72) * specificity
        scored.append(
            {
                "skill": skill,
                "count": count,
                "market_score": round(market_score, 3),
                "specificity_score": round(specificity, 3),
            }
        )

    return sorted(
        scored,
        key=lambda item: (-item["market_score"], -item["count"], item["skill"]),
    )[:MARKET_ANALYSIS_TOP_SKILLS_CHART]


def _fit_metrics(
    user_skills: set[str],
    top_skills: list[dict[str, Any]],
) -> dict[str, Any]:
    top_names = [str(x["skill"]) for x in top_skills[:15]]
    matched = [s for s in top_names if s in user_skills]
    missing = [s for s in top_names[:10] if s not in user_skills]
    strengths = matched[:8]
    weaknesses = missing[:8]
    return {
        "user_skill_count": len(user_skills),
        "matched_skills_in_top_market": strengths,
        "missing_top_market_skills": weaknesses,
        "overlap_count_in_top_15": len(matched),
    }


def _profile_blurb(profile_row: tuple | None, skills: list[str]) -> str:
    if not profile_row:
        return "Skills: " + ", ".join(skills) if skills else "(no profile)"
    (
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
        _resume_fn,
    ) = profile_row
    parts = [
        f"Skills (tags): {skill_tags or ''}",
        f"Years experience: {years_experience}",
        f"Current location: {current_location or ''}",
        f"Preferred roles: {preferred_roles or ''}",
        f"Industries: {industries or ''}",
        f"Work mode: {work_mode or ''}",
        f"Employment type: {employment_type or ''}",
        f"Preferred locations: {preferred_locations or ''}",
        f"Salary range: {salary_min} – {salary_max}",
        f"Structured user skills: {', '.join(skills)}",
    ]
    return "\n".join(p for p in parts if p.strip())


def retrieve_rag_chunks(
    conn: psycopg.Connection,
    *,
    job_ids: list[int],
    query_embedding: list[float],
    top_k: int = MARKET_ANALYSIS_RAG_TOP_CHUNKS,
) -> list[dict[str, Any]]:
    if not job_ids or not query_embedding:
        return []
    _register(conn)
    k = max(1, min(top_k, 40))
    sql = """
        SELECT c.job_id, c.chunk_index, c.content, j.title, j.company,
               (c.embedding <=> %(q)s::vector) AS dist
        FROM job_description_chunks c
        INNER JOIN jobs j ON j.id = c.job_id
        WHERE c.job_id = ANY(%(ids)s::bigint[])
        ORDER BY c.embedding <=> %(q)s::vector
        LIMIT %(k)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"q": query_embedding, "ids": job_ids, "k": k})
        rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "job_id": int(r[0]),
                "chunk_index": int(r[1]),
                "content": (r[2] or "")[:2_000],
                "title": r[3],
                "company": r[4],
                "distance": float(r[5]) if r[5] is not None else None,
            }
        )
    return out


def _cached_rag_query_embedding(
    conn: psycopg.Connection,
    *,
    user_id: int,
    profile_version: str,
    query_hash: str,
) -> list[float] | None:
    try:
        _register(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT embedding
                FROM market_analysis_rag_query_embeddings
                WHERE user_id = %s
                  AND profile_version = %s
                  AND query_hash = %s
                """,
                (user_id, profile_version, query_hash),
            )
            row = cur.fetchone()
    except psycopg.Error:
        conn.rollback()
        return None
    if not row or row[0] is None:
        return None
    return list(row[0])


def _store_rag_query_embedding(
    conn: psycopg.Connection,
    *,
    user_id: int,
    profile_version: str,
    query_hash: str,
    embedding: list[float],
) -> None:
    try:
        _register(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO market_analysis_rag_query_embeddings (
                    user_id,
                    profile_version,
                    query_hash,
                    embedding
                )
                VALUES (%s, %s, %s, %s::vector)
                ON CONFLICT (user_id, profile_version, query_hash) DO UPDATE SET
                  embedding = EXCLUDED.embedding,
                  updated_at = now()
                """,
                (user_id, profile_version, query_hash, embedding),
            )
        conn.commit()
    except psycopg.Error:
        conn.rollback()


def _rag_query_embedding(
    profile_blurb: str,
    *,
    conn: psycopg.Connection | None = None,
    user_id: int | None = None,
    profile_version: str | None = None,
) -> list[float]:
    q = (
        "Job requirements, responsibilities, tech stack, seniority, and qualifications "
        "relevant to this candidate profile:\n\n" + profile_blurb[:6_000]
    )
    query = q.strip()
    query_hash = hashlib.sha256(query.encode("utf-8", errors="replace")).hexdigest()
    cache_key: tuple[Any, ...]
    if user_id is not None and profile_version:
        cache_key = ("user_profile", int(user_id), profile_version, query_hash)
    else:
        cache_key = ("query", query_hash)
    cached = _rag_query_embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    if conn is not None and user_id is not None and profile_version:
        cached = _cached_rag_query_embedding(
            conn,
            user_id=user_id,
            profile_version=profile_version,
            query_hash=query_hash,
        )
        if cached is not None:
            _rag_query_embedding_cache.set(cache_key, cached)
            return cached
    embedding = embed_texts([query])[0]
    _rag_query_embedding_cache.set(cache_key, embedding)
    if conn is not None and user_id is not None and profile_version:
        _store_rag_query_embedding(
            conn,
            user_id=user_id,
            profile_version=profile_version,
            query_hash=query_hash,
            embedding=embedding,
        )
    return embedding


def generate_rag_insights(
    *,
    profile_blurb: str,
    fit: dict[str, Any],
    aggregates_summary: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    if not os.environ.get(OPENAI_API_KEY_ENV):
        return {
            "available": False,
            "narrative": "",
            "error": f"{OPENAI_API_KEY_ENV} is not set; narrative insights are disabled.",
        }
    if not chunks:
        return {
            "available": False,
            "narrative": "",
            "error": "No description chunks indexed yet. Run the job embeddings worker and "
            "job_chunks_backfill (or wait for chunk rebuilds after ingestion).",
        }

    from openai import OpenAI

    client = OpenAI()
    chunk_lines = []
    for i, ch in enumerate(chunks, 1):
        chunk_lines.append(
            f"[{i}] job_id={ch['job_id']} title={ch.get('title')!r} company={ch.get('company')!r}\n"
            f"{ch.get('content', '')}"
        )
    context = "\n\n".join(chunk_lines)

    system = """You are a career market analyst for Career Copilot.
Use ONLY the retrieved job snippets and the numeric summary provided.
Write concise markdown (headers optional) covering:
1) What employers in this cohort emphasize (themes, tools, seniority).
2) 3–5 bullet **strengths** of the candidate relative to this cohort (from profile + overlap).
3) 3–5 bullet **gaps** or **weaknesses** (skills/themes common in listings but weak in profile).
Reference snippet numbers like [1], [2] where helpful. Do not invent employers or salaries not in context."""

    user_msg = f"""--- Candidate profile ---
{profile_blurb}

--- Cohort stats (from database) ---
{aggregates_summary}

--- Retrieved job description snippets ---
{context}
"""

    user_extra = f"""

--- Skill overlap (structured) ---
{fit}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg + user_extra},
        ],
        temperature=0.35,
        max_tokens=900,
    )
    text = (resp.choices[0].message.content or "").strip()
    citations = [
        {
            "job_id": ch["job_id"],
            "chunk_index": ch["chunk_index"],
            "title": ch.get("title"),
            "company": ch.get("company"),
        }
        for ch in chunks[:8]
    ]
    return {"available": True, "narrative": text, "citations": citations, "error": None}


def build_market_analysis_report(
    conn: psycopg.Connection,
    *,
    user_id: int,
    filters: MarketCohortFilters,
    cohort_limit: int,
    include_rag: bool = False,
) -> dict[str, Any]:
    cache_key = _market_analysis_cache_key(
        user_id=user_id,
        filters=filters,
        cohort_limit=cohort_limit,
        include_rag=include_rag,
    )
    cached = _market_analysis_report_cache.get(cache_key)
    if cached is not None:
        return cached

    job_ids, cohort_meta = cohort_job_ids(
        conn, user_id=user_id, filters=filters, cohort_limit=cohort_limit
    )
    aggregates = _aggregates_for_cohort(conn, job_ids)
    user_skills_list = list_user_skills_lower(conn, user_id)
    user_skills = set(user_skills_list)
    fit = _fit_metrics(user_skills, aggregates["top_skills"])
    rag: dict[str, Any] = {
        "available": False,
        "narrative": "",
        "citations": [],
        "error": None,
    }
    if include_rag and job_ids:
        try:
            profile_row = get_profile_by_user_id(conn, user_id)
            profile_blurb = _profile_blurb(profile_row, user_skills_list)
            sal = aggregates.get("salary") or {}
            top_skill_bits = [
                f"{s['skill']}({s['count']})" for s in aggregates.get("top_skills", [])[:5]
            ]
            aggregates_summary = (
                f"Cohort size: {len(job_ids)} jobs. "
                f"Top skills: {', '.join(top_skill_bits)}. "
                f"Salary (when min+max present): avg mid ~ {sal.get('avg_mid_when_both')}. "
                f"Locations: {', '.join(loc['location'] for loc in aggregates['top_locations'][:5])}."
            )
            qvec = _rag_query_embedding(
                profile_blurb,
                conn=conn,
                user_id=user_id,
                profile_version=_profile_version(profile_row, user_skills_list),
            )
            rag_chunks = retrieve_rag_chunks(conn, job_ids=job_ids, query_embedding=qvec)
            rag = generate_rag_insights(
                profile_blurb=profile_blurb,
                fit=str(fit),
                aggregates_summary=aggregates_summary,
                chunks=rag_chunks,
            )
        except Exception as e:
            rag = {
                "available": False,
                "narrative": "",
                "citations": [],
                "error": str(e)[:500],
            }

    report = {
        "cohort": {
            "size": len(job_ids),
            "job_ids_sample": job_ids[:20],
            **cohort_meta,
        },
        "filters": {
            "posted_within_days": filters.posted_within_days,
            "location_contains": filters.location_contains,
            "title_contains": filters.title_contains,
            "source_equals": filters.source_equals,
            "remote_only": filters.remote_only,
            "salary_at_least": filters.salary_at_least,
        },
        "weekly_posted": aggregates["weekly_posted"],
        "top_skills": aggregates["top_skills"],
        "salary": aggregates["salary"],
        "top_locations": aggregates["top_locations"],
        "fit": fit,
        "rag": rag,
    }
    _market_analysis_report_cache.set(cache_key, report)
    return report
