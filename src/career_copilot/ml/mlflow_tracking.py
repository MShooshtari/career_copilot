"""MLflow tracking URI resolution and experiment creation (local or remote)."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from career_copilot.ml.dataset_store import get_data_dir


def _normalize_postgresql_tracking_uri(uri: str) -> str:
    """
    MLflow's SQL store passes the URI to SQLAlchemy. A plain ``postgresql://`` URL
    selects the ``psycopg2`` dialect, which this image does not install; we ship
    ``psycopg`` (v3) only. Rewrite to ``postgresql+psycopg://`` so the v3 driver is used.
    """
    if uri.startswith("postgresql+") or uri.startswith("postgres+"):
        return uri
    if uri.startswith("postgresql://"):
        return "postgresql+psycopg://" + uri[len("postgresql://") :]
    if uri.startswith("postgres://"):
        return "postgresql+psycopg://" + uri[len("postgres://") :]
    return uri


def get_mlflow_tracking_uri() -> str:
    """
    Tracking backend for ranking models.

    If MLFLOW_TRACKING_URI is set (http(s) MLflow server, postgresql://, etc.), use it.
    Otherwise use local SQLite at data/mlflow.db (development default).
    """
    env = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if env:
        return _normalize_postgresql_tracking_uri(env)
    data_dir = get_data_dir()
    db_path = (data_dir / "mlflow.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"


def ensure_experiment_for_training(
    client: MlflowClient,
    *,
    experiment_name: str,
    data_dir: Path,
) -> None:
    """
    Create the experiment if it does not exist.

    - Local SQLite: artifacts under data/mlflow_artifacts (file://).
    - HTTP(S) tracking server: use the server's default artifact root.
    - Direct DB URI (e.g. postgresql://): set MLFLOW_EXPERIMENT_ARTIFACT_LOCATION to a
      remote root (e.g. wasbs://container@account.blob.core.windows.net/mlflow).
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is not None:
        return

    tracking = (mlflow.get_tracking_uri() or "").strip()
    custom = (os.environ.get("MLFLOW_EXPERIMENT_ARTIFACT_LOCATION") or "").strip()

    if custom:
        client.create_experiment(name=experiment_name, artifact_location=custom)
        return
    if tracking.startswith("http://") or tracking.startswith("https://"):
        client.create_experiment(name=experiment_name)
        return

    if tracking.startswith("postgresql") or tracking.startswith("mysql"):
        raise RuntimeError(
            "Database tracking URI requires MLFLOW_EXPERIMENT_ARTIFACT_LOCATION "
            "(e.g. wasbs://container@account.blob.core.windows.net/mlflow). "
            "Or use an MLflow server (MLFLOW_TRACKING_URI=https://...) so the server supplies a default artifact root."
        )

    artifacts_dir = (data_dir / "mlflow_artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    client.create_experiment(name=experiment_name, artifact_location=artifacts_dir.as_uri())
