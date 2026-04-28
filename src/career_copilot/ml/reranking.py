from __future__ import annotations

import hashlib
import random
import re
from datetime import UTC, date, datetime
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "developer",
    "engineer",
    "for",
    "in",
    "of",
    "remote",
    "senior",
    "software",
    "the",
}


def _base_score(candidate: dict[str, Any]) -> float:
    if candidate.get("model_score") is not None:
        try:
            return float(candidate["model_score"])
        except (TypeError, ValueError):
            pass
    distance = candidate.get("distance")
    try:
        dist = float(distance) if distance is not None else None
    except (TypeError, ValueError):
        dist = None
    return 1.0 / (1.0 + max(dist or 0.0, 0.0))


def _metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    meta = candidate.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _skills(candidate: dict[str, Any]) -> set[str]:
    raw = _metadata(candidate).get("skills")
    if isinstance(raw, str):
        return {s.strip().lower() for s in raw.split(",") if s.strip()}
    if isinstance(raw, list):
        return {str(s).strip().lower() for s in raw if str(s).strip()}
    return set()


def _tokens(candidate: dict[str, Any]) -> set[str]:
    meta = _metadata(candidate)
    text = " ".join(str(meta.get(key) or "") for key in ("title", "company", "location", "source"))
    tokens = {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}
    return tokens | _skills(candidate)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _category(candidate: dict[str, Any]) -> str:
    skills = sorted(_skills(candidate))
    if skills:
        return f"skill:{skills[0]}"
    title_tokens = sorted(_tokens(candidate))
    if title_tokens:
        return f"title:{title_tokens[0]}"
    source = str(_metadata(candidate).get("source") or "").strip().lower()
    return f"source:{source}" if source else "unknown"


def _stable_seed(user_id: int | None, candidates: list[dict[str, Any]], today: date | None) -> int:
    hasher = hashlib.sha256()
    hasher.update(str(user_id or 0).encode("utf-8"))
    hasher.update(str(today or datetime.now(tz=UTC).date()).encode("utf-8"))
    for candidate in candidates[:50]:
        hasher.update(
            str(candidate.get("postgres_job_id") or candidate.get("id") or "").encode("utf-8")
        )
    return int(hasher.hexdigest()[:16], 16)


def rerank_with_diversity_and_exploration(
    candidates: list[dict[str, Any]],
    *,
    window_size: int,
    user_id: int | None = None,
    diversity_penalty: float,
    category_penalty: float,
    exploration_rate: float,
    seed: int | None = None,
    today: date | None = None,
) -> list[dict[str, Any]]:
    if not candidates or window_size <= 0:
        return []

    pool = [dict(c) for c in candidates]
    pool.sort(key=_base_score, reverse=True)
    want = min(window_size, len(pool))
    explore_count = min(max(round(want * exploration_rate), 0), max(want - 1, 0))
    exploit_count = want - explore_count

    selected: list[dict[str, Any]] = []
    remaining = pool.copy()
    selected_tokens: list[set[str]] = []
    category_counts: dict[str, int] = {}

    for _ in range(exploit_count):
        best_index = 0
        best_score = float("-inf")
        for idx, candidate in enumerate(remaining):
            tokens = _tokens(candidate)
            max_similarity = max((_jaccard(tokens, seen) for seen in selected_tokens), default=0.0)
            category = _category(candidate)
            adjusted_score = (
                _base_score(candidate)
                - diversity_penalty * max_similarity
                - category_penalty * category_counts.get(category, 0)
            )
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_index = idx

        chosen = remaining.pop(best_index)
        chosen["rerank_score"] = float(best_score)
        chosen["rerank_reason"] = "exploit_diverse"
        selected.append(chosen)
        selected_tokens.append(_tokens(chosen))
        category = _category(chosen)
        category_counts[category] = category_counts.get(category, 0) + 1

    if explore_count > 0 and remaining:
        rng = random.Random(seed if seed is not None else _stable_seed(user_id, pool, today))
        explore_pool = remaining[: max(explore_count * 5, explore_count)]
        rng.shuffle(explore_pool)
        for candidate in explore_pool[:explore_count]:
            chosen = dict(candidate)
            chosen["rerank_score"] = _base_score(chosen)
            chosen["rerank_reason"] = "explore"
            selected.append(chosen)

    return selected[:want]
