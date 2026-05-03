"""Dynamic skill tag extraction for job descriptions.

The extractor avoids a fixed skill taxonomy. It mines skill-like phrases from
source tags, explicit skills/requirements sections, and experience/proficiency
language so it can work across job families instead of only technical roles.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from typing import Any

_SECTION_HEADING_RE = re.compile(
    r"""
    ^\s*
    (?:
        required|preferred|minimum|basic|desired|nice[-\s]to[-\s]have
    )?\s*
    (?:
        skills?|qualifications?|requirements?|experience|competenc(?:y|ies)|
        licenses?|certifications?|what\s+you(?:'ll|\s+will)\s+bring
    )
    \s*:?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_INLINE_SECTION_RE = re.compile(
    r"""
    ^\s*
    (?:
        required|preferred|minimum|basic|desired|nice[-\s]to[-\s]have
    )?\s*
    (?:
        skills?|qualifications?|requirements?|experience|competenc(?:y|ies)|
        licenses?|certifications?
    )
    \s*:\s*(?P<body>.+)$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CUE_RE = re.compile(
    r"""
    \b(?:
        experience|experienced|proficient|proficiency|knowledge|familiarity|
        familiar|skilled|expertise|certified|certification|licensed|licensure|
        competency|competence|background
    )
    \s+(?:with|in|using|of|for)?\s*
    (?P<body>[^.;:\n]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_STRIP_CHARS = " \t\r\n-\u2013\u2014:;,.()[]{}"
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")
_SPACE_RE = re.compile(r"\s+")
_YEARS_PREFIX_RE = re.compile(
    r"^(?:\d+\+?\s*)?(?:years?|yrs?)\s+(?:of\s+)?(?:hands[-\s]on\s+)?",
    re.IGNORECASE,
)

_TRAILING_NOISE_RE = re.compile(
    r"\s+(?:experience|skills?|knowledge|proficiency|certification|license|licence)$",
    re.IGNORECASE,
)

_GENERIC_CANDIDATES = {
    "a plus",
    "an asset",
    "benefits",
    "bonus",
    "building",
    "code",
    "coder",
    "coding",
    "competitive pay",
    "communication",
    "degree",
    "designing",
    "developer",
    "development",
    "digital nomad",
    "diploma",
    "excellent communication",
    "engineer",
    "engineering",
    "increase your earnings with higher rates",
    "interviewing",
    "full time",
    "high school",
    "job description",
    "must have",
    "nice to have",
    "preferred",
    "required",
    "requirements",
    "responsibilities",
    "salary",
    "senior",
    "software",
    "start taking covered clients sooner",
    "team player",
    "work experience",
}

_GENERIC_PATTERNS = (
    r"\b(?:digital\s+nomad|remote\s+work|work\s+from\s+home)\b",
    r"\b(?:earnings?|higher\s+rates?|covered\s+clients?|start\s+taking)\b",
)

_LOW_VALUE_SKILLS = {
    "adaptability",
    "attention to detail",
    "attention to details",
    "collaboration",
    "critical thinking",
    "detail oriented",
    "multitasking",
    "organization",
    "problem solving",
    "self motivation",
    "team collaboration",
    "time management",
}

_STOP_PREFIXES = (
    "ability to ",
    "able to ",
    "be able to ",
    "demonstrated ",
    "excellent ",
    "good ",
    "hands-on ",
    "prior ",
    "proven ",
    "required ",
    "strong ",
    "working ",
)

_STOP_SUFFIXES = (
    " is preferred",
    " is required",
    " preferred",
    " required",
    " strongly preferred",
    " would be an asset",
    " would be a plus",
)

_LOWERCASE_WORDS = {"and", "for", "in", "of", "on", "or", "the", "to", "with"}
OPENAI_SKILL_EXTRACTION_MODEL = "gpt-4o-mini"

_AI_EXTRACTION_SYSTEM = """You extract concise, specific skill tags from job descriptions.
Return a JSON object with a single key "skills" whose value is an array of strings.

Rules:
- Include concrete skills, tools, technologies, certifications, methods, domain capabilities, and job-specific competencies.
- Prefer skills that would be useful as resume keywords or market-demand tags.
- Exclude marketing copy, platform benefits, hiring-process words, generic job words, seniority, benefits, degree requirements, schedules, work mode, location, salary, and vague traits.
- Do not include broad terms such as engineering, development, code, communication, problem solving, attention to detail, team collaboration, team player, interviewing, required, preferred, responsibilities, or job description.
- Do not include phrases like "increase your earnings with higher rates" or "start taking covered clients sooner"; those are not skills.
- Keep each tag short, usually 1-4 words.
- Return at most the requested number of tags.
- If no specific skill tags are present, return {"skills": []}."""

_AI_RESUME_EXTRACTION_SYSTEM = """You extract concise, specific skill tags from resumes.
Return a JSON object with a single key "skills" whose value is an array of strings.

Rules:
- Include concrete skills, tools, technologies, certifications, methods, domain capabilities, and role-specific competencies demonstrated by the candidate.
- Prefer skills that would be useful as resume keywords or job matching tags.
- Exclude employer names, schools, locations, dates, job titles, seniority, degrees by themselves, contact information, generic resume section headings, and vague traits.
- Do not include broad terms such as engineering, development, code, communication, problem solving, attention to detail, team collaboration, team player, responsibilities, or experience.
- Keep each tag short, usually 1-4 words.
- Return at most the requested number of tags.
- If no specific skill tags are present, return {"skills": []}."""


def extract_skill_tags(
    text: str | None,
    source_skills: Iterable[str] | None = None,
    *,
    max_tags: int = 30,
) -> list[str]:
    """Extract normalized skill tags without relying on a fixed skills list."""
    found: list[str] = []
    seen: set[str] = set()

    for skill in source_skills or ():
        _append_skill(found, seen, skill, max_tags=max_tags)

    if text:
        for candidate in _extract_text_candidates(text):
            _append_skill(found, seen, candidate, max_tags=max_tags)
            if len(found) >= max_tags:
                break

    return _prefer_specific_skills(found)


def extract_ai_skill_tags(
    text: str | None,
    *,
    max_tags: int = 30,
    model: str | None = None,
    client: Any | None = None,
) -> list[str]:
    """Use OpenAI to extract specific skill tags, then apply local tag normalization."""
    return _extract_ai_skill_tags(
        text,
        system_prompt=_AI_EXTRACTION_SYSTEM,
        document_label="Job description",
        instruction="Extract up to {max_tags} specific skill tags from this job description.",
        max_tags=max_tags,
        model=model,
        client=client,
    )


def extract_ai_resume_skill_tags(
    text: str | None,
    *,
    max_tags: int = 30,
    model: str | None = None,
    client: Any | None = None,
) -> list[str]:
    """Use OpenAI to extract normalized skill tags from resume text."""
    return _extract_ai_skill_tags(
        text,
        system_prompt=_AI_RESUME_EXTRACTION_SYSTEM,
        document_label="Resume",
        instruction="Extract up to {max_tags} specific skill tags from this resume.",
        max_tags=max_tags,
        model=model,
        client=client,
    )


def _extract_ai_skill_tags(
    text: str | None,
    *,
    system_prompt: str,
    document_label: str,
    instruction: str,
    max_tags: int,
    model: str | None,
    client: Any | None,
) -> list[str]:
    if not text or not text.strip():
        return []

    openai_client = client or _get_openai_client()
    response = openai_client.chat.completions.create(
        model=model
        or os.environ.get("OPENAI_SKILL_EXTRACTION_MODEL")
        or OPENAI_SKILL_EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{instruction.format(max_tags=max_tags)}\n\n"
                    f"Generic examples to exclude: {', '.join(sorted(_GENERIC_CANDIDATES))}\n\n"
                    f"{document_label}:\n{text.strip()[:30000]}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = (response.choices[0].message.content or "").strip()
    candidates = _parse_ai_skill_response(raw)

    found: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        _append_skill(found, seen, candidate, max_tags=max_tags)
    return _prefer_specific_skills(found)


def normalize_skill_tag(value: str | None) -> str | None:
    """Normalize and reject low-value skill tags from any source."""
    if value is None:
        return None
    return _clean_candidate(value)


def skill_specificity_score(value: str | None) -> float:
    """Score how useful a tag is for market skill charts; vague soft skills are demoted."""
    cleaned = _clean_candidate(value or "")
    if not cleaned:
        return 0.0

    lower = cleaned.casefold()
    words = re.findall(r"[A-Za-z0-9+#./'-]+", cleaned)
    if lower in _LOW_VALUE_SKILLS:
        return 0.07

    score = 1.0
    if len(words) >= 2:
        score += 0.15
    if any(ch in cleaned for ch in "+#./"):
        score += 0.25
    if any(token.isupper() and len(token) <= 8 for token in words):
        score += 0.2
    if re.search(r"\b(?:certification|certified|license|licensed)\b", lower):
        score += 0.15
    return min(score, 1.6)


def _get_openai_client() -> Any:
    from career_copilot.database.db import load_env

    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot extract AI skill tags.")

    from openai import OpenAI

    return OpenAI()


def _parse_ai_skill_response(raw: str) -> list[str]:
    if "```" in raw:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    skills = data.get("skills") if isinstance(data, dict) else None
    if not isinstance(skills, list):
        return []
    return [skill for skill in skills if isinstance(skill, str)]


def _extract_text_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    in_skill_section = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            in_skill_section = False
            continue

        bullet_line = _BULLET_RE.sub("", line).strip()
        inline_section = _INLINE_SECTION_RE.match(bullet_line)
        if inline_section:
            candidates.extend(_split_candidate_span(inline_section.group("body")))
            in_skill_section = True
            continue

        if _SECTION_HEADING_RE.match(bullet_line):
            in_skill_section = True
            continue

        if in_skill_section and _looks_like_candidate_line(raw_line):
            candidates.extend(_split_candidate_span(bullet_line))

        for match in _CUE_RE.finditer(bullet_line):
            candidates.extend(_split_candidate_span(match.group("body")))

    return candidates


def _looks_like_candidate_line(raw_line: str) -> bool:
    stripped = raw_line.strip()
    return bool(
        _BULLET_RE.match(stripped) or "," in stripped or ";" in stripped or len(stripped) <= 80
    )


def _split_candidate_span(span: str) -> list[str]:
    span = re.split(r"\b(?:including|such as|like)\b", span, maxsplit=1, flags=re.IGNORECASE)[-1]
    span = re.split(r"\b(?:while|where|when|that|who|which)\b", span, maxsplit=1)[0]
    parts = re.split(r"\s*(?:,|;|\||\band/or\b|\bor\b|\band\b)\s*", span)
    return [part for part in (_clean_candidate(part) for part in parts) if part]


def _clean_candidate(candidate: str) -> str | None:
    value = candidate.strip()
    value = value.strip(_STRIP_CHARS)
    value = _SPACE_RE.sub(" ", value)
    if not value:
        return None

    lower = value.lower()
    for prefix in _STOP_PREFIXES:
        if lower.startswith(prefix):
            value = value[len(prefix) :].strip()
            lower = value.lower()
            break

    value = _YEARS_PREFIX_RE.sub("", value).strip()
    value = _TRAILING_NOISE_RE.sub("", value).strip()

    lower = value.lower()
    for suffix in _STOP_SUFFIXES:
        if lower.endswith(suffix):
            value = value[: -len(suffix)].strip()
            lower = value.lower()
            break

    value = value.strip(_STRIP_CHARS)
    if not _is_plausible_skill(value):
        return None
    return _canonicalize(value)


def _is_plausible_skill(value: str) -> bool:
    lower = value.lower()
    words = re.findall(r"[A-Za-z0-9+#./'-]+", value)
    if len(value) < 2 or len(value) > 60:
        return False
    if not words or len(words) > 6:
        return False
    if lower in _GENERIC_CANDIDATES:
        return False
    if len(words) <= 2 and any(re.search(pattern, lower) for pattern in _GENERIC_PATTERNS):
        return False
    if re.search(r"\b(?:salary|benefits?|schedule|shift|remote|hybrid|onsite)\b", lower):
        return False
    if re.search(r"\b(?:bachelor|master|phd|degree|diploma|equivalent)\b", lower):
        return False
    if re.search(r"\b(?:years?|yrs?)\b", lower):
        return False
    if lower.startswith(("the ", "this ", "our ", "your ", "you ", "we ")):
        return False
    return any(char.isalpha() for char in value)


def _canonicalize(value: str) -> str:
    tokens = re.split(r"(\s+)", value)
    canonical: list[str] = []
    word_index = 0

    for token in tokens:
        if not token.strip():
            canonical.append(token)
            continue

        lower = token.lower()
        if word_index > 0 and lower in _LOWERCASE_WORDS:
            canonical.append(lower)
        elif _should_preserve_token(token):
            canonical.append(token)
        else:
            canonical.append(token[:1].upper() + token[1:].lower())
        word_index += 1

    return "".join(canonical)


def _should_preserve_token(token: str) -> bool:
    letters = "".join(ch for ch in token if ch.isalpha())
    return (
        (letters.isupper() and len(letters) <= 8)
        or any(ch in token for ch in "+#./")
        or any(ch.isupper() for ch in token[1:])
    )


def _append_skill(found: list[str], seen: set[str], candidate: str, *, max_tags: int) -> None:
    if len(found) >= max_tags:
        return
    cleaned = _clean_candidate(candidate)
    if cleaned is None:
        return
    key = cleaned.casefold()
    if key in seen:
        return
    seen.add(key)
    found.append(cleaned)


def _prefer_specific_skills(skills: list[str]) -> list[str]:
    if any(skill_specificity_score(skill) >= 0.5 for skill in skills):
        return [skill for skill in skills if skill_specificity_score(skill) >= 0.5]
    return skills
