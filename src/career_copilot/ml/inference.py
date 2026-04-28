from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from career_copilot.database.db import load_env
from career_copilot.ml.mlflow_tracking import get_mlflow_tracking_uri
from career_copilot.ml.ranking_dataset import FEATURE_COLUMNS

FRESHNESS_NEW_DAYS = 3
FRESHNESS_DECAY_LAMBDA = 0.05


@lru_cache(maxsize=1)
def get_ranking_model() -> Any | None:
    """
    Load the MLflow ranking model configured by MLFLOW_RANKING_RUN_ID.

    Uses MLFLOW_TRACKING_URI when set (remote server or database store); otherwise
    local SQLite at data/mlflow.db. Returns None if no run id is set.
    """
    load_env()
    run_id = (os.environ.get("MLFLOW_RANKING_RUN_ID") or "").strip()
    if not run_id:
        return None

    mlflow.set_tracking_uri(get_mlflow_tracking_uri())

    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)


def _build_feature_frame_from_distances(
    distances: Iterable[float | None],
) -> pd.DataFrame:
    """
    Build a feature DataFrame expected by the ranking model from raw vector distances.

    For now we derive only embedding_similarity from distance (lower = more similar)
    and set all other features to 0.0, relying on the model's preprocessing to handle scaling.
    """
    rows: list[dict[str, float]] = []
    for d in distances:
        try:
            dist = float(d) if d is not None else None
        except (TypeError, ValueError):
            dist = None
        if dist is None:
            embedding_similarity = 0.0
        else:
            # Convert distance (lower is better) to a bounded similarity signal.
            embedding_similarity = float(1.0 / (1.0 + max(dist, 0.0)))

        row = dict.fromkeys(FEATURE_COLUMNS, 0.0)
        if "embedding_similarity" in row:
            row["embedding_similarity"] = embedding_similarity
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))

    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLUMNS]


def _parse_posted_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _freshness_features(posted_at: Any, *, now: datetime | None = None) -> dict[str, float]:
    ref = now or datetime.now(tz=UTC)
    posted = _parse_posted_at(posted_at)
    if posted is None:
        return {"days_since_posted": 180.0, "is_new": 0.0, "decay_score": 0.0}

    age_seconds = max((ref - posted).total_seconds(), 0.0)
    days_since_posted = age_seconds / 86_400
    return {
        "days_since_posted": float(days_since_posted),
        "is_new": float(days_since_posted <= FRESHNESS_NEW_DAYS),
        "decay_score": float(np.exp(-FRESHNESS_DECAY_LAMBDA * days_since_posted)),
    }


def _build_feature_frame_from_candidates(
    raw_results: list[dict],
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for item in raw_results:
        distance = item.get("distance")
        row = _build_feature_frame_from_distances([distance]).iloc[0].to_dict()
        meta = item.get("metadata") or {}
        row.update(_freshness_features(meta.get("posted_at"), now=now))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))

    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLUMNS]


def score_candidates_by_distance(raw_results: list[dict]) -> list[dict]:
    """
    Attach ranking model scores to candidate results and sort in descending order.

    Each item in raw_results is expected to have a 'distance' field (lower = more similar).
    If the MLflow model is not configured or loading/scoring fails, the input list
    is returned unchanged.
    """
    if not raw_results:
        return raw_results
    model = get_ranking_model()
    if model is None:
        return raw_results

    try:
        X = _build_feature_frame_from_candidates(raw_results)
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        return raw_results

    scored: list[dict] = []
    for item, p in zip(raw_results, proba, strict=False):
        new_item = dict(item)
        new_item["model_score"] = float(p)
        scored.append(new_item)

    scored.sort(key=lambda r: r.get("model_score", 0.0), reverse=True)
    return scored
