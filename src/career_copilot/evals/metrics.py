from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from career_copilot.ml.ranking_metrics import ranking_metrics_at_k

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _token_set(value: Any) -> set[str]:
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value or "")
    return set(_TOKEN_RE.findall(text.casefold()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _numeric_match(actual: Any, expected: Any, *, tolerance: float = 0.05) -> float:
    if actual in (None, "") and expected in (None, ""):
        return 1.0
    if actual in (None, "") or expected in (None, ""):
        return 0.0
    try:
        actual_float = float(actual)
        expected_float = float(expected)
    except (TypeError, ValueError):
        return 0.0
    if expected_float == 0:
        return 1.0 if actual_float == 0 else 0.0
    return 1.0 if math.isclose(actual_float, expected_float, rel_tol=tolerance) else 0.0


def _field_score(actual: Any, expected: Any) -> float:
    if isinstance(expected, list):
        return _jaccard(_token_set(actual), _token_set(expected))
    if isinstance(expected, int | float):
        return _numeric_match(actual, expected)
    normalized_expected = _normalize_text(expected)
    normalized_actual = _normalize_text(actual)
    if not normalized_expected and not normalized_actual:
        return 1.0
    if not normalized_expected or not normalized_actual:
        return 0.0
    if normalized_actual == normalized_expected:
        return 1.0
    return _jaccard(_token_set(normalized_actual), _token_set(normalized_expected))


def score_job_extraction(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    fields: Iterable[str] | None = None,
) -> dict[str, float]:
    """Score extracted job fields against a golden job record."""
    field_names = list(fields or expected.keys())
    if not field_names:
        return {"field_accuracy": 0.0, "required_completeness": 0.0}

    per_field = {
        f"field_{field_name}": _field_score(actual.get(field_name), expected.get(field_name))
        for field_name in field_names
    }
    present = [field_name for field_name in field_names if actual.get(field_name) not in (None, "", [])]
    return {
        **per_field,
        "field_accuracy": sum(per_field.values()) / len(per_field),
        "required_completeness": len(present) / len(field_names),
    }


def aggregate_metric_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(sum(row[key] for row in rows if key in row) / sum(1 for row in rows if key in row))
        for key in keys
    }


def score_retrieval_case(
    retrieved_ids: Iterable[Any],
    relevant_ids: Iterable[Any],
    *,
    k: int,
) -> dict[str, float]:
    """Score top-k retrieval against known relevant ids."""
    retrieved = [str(item) for item in retrieved_ids][:k]
    relevant = {str(item) for item in relevant_ids}
    if not relevant:
        return {f"retrieval_precision_at_{k}": 0.0, f"retrieval_recall_at_{k}": 0.0}
    hits = sum(1 for item in retrieved if item in relevant)
    return {
        f"retrieval_precision_at_{k}": hits / k,
        f"retrieval_recall_at_{k}": hits / len(relevant),
    }


def score_ranking_cases(cases: list[dict[str, Any]], *, k: int) -> dict[str, float]:
    """Score recommendation ranking cases with the repo's existing ranking metrics."""
    labels: list[float] = []
    scores: list[float] = []
    groups: list[str] = []
    for case in cases:
        request_id = str(case.get("id") or case.get("request_id") or len(groups))
        for candidate in case.get("candidates", []):
            labels.append(float(candidate.get("label", 0.0)))
            scores.append(float(candidate.get("score", candidate.get("model_score", 0.0))))
            groups.append(request_id)
    return ranking_metrics_at_k(labels, scores, k=k, group_ids=groups)

