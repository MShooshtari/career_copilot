from __future__ import annotations

import os
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from career_copilot.database.db import load_env
from career_copilot.ml.dataset_store import get_data_dir
from career_copilot.ml.ranking_dataset import FEATURE_COLUMNS


@lru_cache(maxsize=1)
def get_ranking_model() -> Any | None:
    """
    Load the MLflow ranking model configured by MLFLOW_RANKING_RUN_ID.

    Returns the deserialized sklearn Pipeline (or None if no run id is set).
    """
    load_env()
    run_id = (os.environ.get("MLFLOW_RANKING_RUN_ID") or "").strip()
    if not run_id:
        return None

    data_dir = get_data_dir()
    db_path = (data_dir / "mlflow.db").resolve()
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)


def _build_feature_frame_from_distances(
    distances: Iterable[float | None],
) -> pd.DataFrame:
    """
    Build a feature DataFrame expected by the ranking model from raw Chroma distances.

    For now we derive only embedding_similarity from distance and set all other
    features to 0.0, relying on the model's preprocessing to handle scaling.
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


def score_candidates_by_distance(raw_results: list[dict]) -> list[dict]:
    """
    Attach ranking model scores to candidate results and sort in descending order.

    Each item in raw_results is expected to have a 'distance' field (from Chroma).
    If the MLflow model is not configured or loading/scoring fails, the input list
    is returned unchanged.
    """
    model = get_ranking_model()
    if model is None or not raw_results:
        return raw_results

    try:
        distances = [r.get("distance") for r in raw_results]
        X = _build_feature_frame_from_distances(distances)
        proba = model.predict_proba(X.to_numpy(dtype=np.float64))[:, 1]
    except Exception:
        return raw_results

    scored: list[dict] = []
    for item, p in zip(raw_results, proba, strict=False):
        new_item = dict(item)
        new_item["model_score"] = float(p)
        scored.append(new_item)

    scored.sort(key=lambda r: r.get("model_score", 0.0), reverse=True)
    return scored
