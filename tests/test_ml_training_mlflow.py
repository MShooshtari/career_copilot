from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from career_copilot.ml import (
    dataset_store,
    mlflow_tracking,
    train_logreg_mlflow,
    train_xgboost_mlflow,
)
from career_copilot.ml.ranking_dataset import FEATURE_COLUMNS, make_mock_ranking_dataset


class _MlflowRecorder:
    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: list[tuple[str, str]] = []
        self.models: list[dict[str, Any]] = []
        self.tracking_uris: list[str] = []
        self.experiments: list[str] = []


def _make_small_similarity_df() -> pd.DataFrame:
    """Return a tiny similarity dataframe with both classes present."""
    n_rows = 8
    # Weak labels in {0.0, 0.5, 1.0}.
    label_values = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 1.0], dtype=float)
    assert label_values.shape[0] == n_rows

    rng = np.random.default_rng(7)
    base = rng.uniform(0.1, 0.9, size=n_rows)

    data: dict[str, Any] = {
        "embedding_similarity": base,
        "title_similarity": np.clip(base + rng.normal(0, 0.05, size=n_rows), 0.0, 1.0),
        "skill_overlap_count": np.arange(n_rows, dtype=int),
        "location_match": (np.arange(n_rows) % 2).astype(int),
        "experience_gap": np.clip(3.0 - base, 0.0, 10.0),
        "salary_match": np.clip(base + 0.1, 0.0, 1.0),
        "location_km": np.linspace(5.0, 100.0, num=n_rows),
        "skill_similarity": np.clip(base + 0.2, 0.0, 1.0),
        "role_similarity": np.clip(base + 0.15, 0.0, 1.0),
        "work_mode_similarity": np.clip(base + 0.1, 0.0, 1.0),
        "employment_type_similarity": np.clip(base + 0.05, 0.0, 1.0),
        "preferred_locations_similarity": np.clip(base + 0.12, 0.0, 1.0),
        "days_since_posted": np.array([1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 45.0, 60.0]),
        "is_new": np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=int),
        "decay_score": np.exp(-0.05 * np.array([1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 45.0, 60.0])),
        "label": label_values,
    }

    df = pd.DataFrame(data)
    # Sanity check: all required feature columns present.
    assert all(c in df.columns for c in FEATURE_COLUMNS)
    return df


def _patch_dataset_store(
    monkeypatch,
    module,
    df: pd.DataFrame,
    tmp_path: Path,
    *,
    blob_uris: dict[str, str] | None = None,
) -> None:
    """Patch dataset_store helpers inside a training module to use an in-memory dataframe."""
    blob_uris = blob_uris or {}

    def _fake_get_data_dir() -> Path:
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    monkeypatch.setattr(mlflow_tracking, "get_data_dir", _fake_get_data_dir)

    def _fake_load(version: str, kind: str = "similarity") -> tuple[pd.DataFrame, str]:
        assert kind == "similarity"
        # Return a copy so tests cannot mutate shared state accidentally.
        return df.copy(), "mock_v1"

    def _fake_get_meta(version: str) -> dict[str, Any]:
        meta: dict[str, Any] = {"label_scheme": "weak_supervision_v1", "n_rows": len(df)}
        if blob_uris:
            meta["blob_uris"] = dict(blob_uris)
        return meta

    def _fake_get_blob_uris(version: str) -> dict[str, str]:
        return dict(blob_uris)

    def _fake_get_path(version: str, kind: str = "similarity") -> Path:
        assert kind == "similarity"
        return tmp_path / "datasets" / "ranking" / "mock_similarity_mock_v1.csv"

    monkeypatch.setattr(module, "load", _fake_load)
    monkeypatch.setattr(module, "get_meta", _fake_get_meta)
    monkeypatch.setattr(module, "get_blob_uris", _fake_get_blob_uris)
    monkeypatch.setattr(module, "get_path", _fake_get_path)
    monkeypatch.setattr(module, "get_data_dir", _fake_get_data_dir)


def _patch_mlflow(monkeypatch, module) -> _MlflowRecorder:
    """Patch mlflow usage in a training module to a lightweight in-memory recorder."""
    rec = _MlflowRecorder()
    monkeypatch.setattr(module, "load_env", lambda: None)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ARTIFACT_LOCATION", raising=False)

    class _DummyClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def create_experiment(
            self, name: str, artifact_location: str | None = None, tags: Any = None
        ) -> str:
            rec.experiments.append(name)
            return "1"

    class _RunContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _start_run(*, run_name: str | None = None, **_: Any) -> _RunContext:  # type: ignore[override]
        # We only care that a run was started; we do not track run IDs here.
        return _RunContext()

    def _log_param(key: str, value: Any) -> None:
        rec.params[key] = value

    def _log_metric(key: str, value: float) -> None:
        rec.metrics[key] = float(value)

    def _log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        rec.artifacts.append((local_path, artifact_path or ""))

    def _set_tracking_uri(uri: str) -> None:
        rec.tracking_uris.append(uri)

    def _set_experiment(name: str) -> None:
        rec.experiments.append(name)

    def _get_tracking_uri() -> str:
        return rec.tracking_uris[-1] if rec.tracking_uris else ""

    def _get_experiment_by_name(name: str) -> Any:
        # Returning None forces the codepath that creates the experiment.
        return None

    def _log_model_sklearn(
        model: Any,
        name: str,
        serialization_format: str,
        pip_requirements: list[str] | None = None,
    ) -> None:
        rec.models.append(
            {
                "name": name,
                "serialization_format": serialization_format,
                "pip_requirements": list(pip_requirements or []),
                "model_type": type(model).__name__,
            }
        )

    # Patch the mlflow symbol inside the training module with a small namespace.
    import types

    mlflow_stub = types.SimpleNamespace(
        start_run=_start_run,
        log_param=_log_param,
        log_metric=_log_metric,
        log_artifact=_log_artifact,
        set_tracking_uri=_set_tracking_uri,
        set_experiment=_set_experiment,
        get_tracking_uri=_get_tracking_uri,
        get_experiment_by_name=_get_experiment_by_name,
        sklearn=types.SimpleNamespace(log_model=_log_model_sklearn),
    )

    monkeypatch.setattr(module, "mlflow", mlflow_stub)
    monkeypatch.setattr(module, "MlflowClient", _DummyClient)
    monkeypatch.setattr(mlflow_tracking, "mlflow", mlflow_stub)
    monkeypatch.setattr(mlflow_tracking, "MlflowClient", _DummyClient)
    return rec


def test_train_logreg_mlflow_logs_params_metrics_and_artifacts(tmp_path, monkeypatch) -> None:
    df = _make_small_similarity_df()
    blob_uris = {
        "similarity": "https://storage.example/training-data/ranking/mock_v1/similarity.csv",
        "embeddings": "https://storage.example/training-data/ranking/mock_v1/embeddings.csv",
        "manifest": "https://storage.example/training-data/ranking/manifest.json",
    }
    _patch_dataset_store(
        monkeypatch,
        train_logreg_mlflow,
        df,
        tmp_path=Path(tmp_path),
        blob_uris=blob_uris,
    )
    rec = _patch_mlflow(monkeypatch, train_logreg_mlflow)

    train_logreg_mlflow.train_and_log(
        dataset_version="latest",
        seed=7,
        test_size=0.25,
        positive_threshold=1.0,
        experiment_name="career-copilot-ranking",
        run_name="logreg-baseline",
        max_iter=100,
        c=1.0,
        undersample=False,
        ranking_k=15,
    )

    # Tracking URI should point at a local SQLite DB under the test data dir.
    assert rec.tracking_uris
    assert rec.tracking_uris[0].startswith("sqlite:///")

    # Model and dataset metadata are logged as params.
    assert rec.params["model_type"] == "logistic_regression"
    logged_features = json.loads(rec.params["features"])
    assert set(logged_features) == set(FEATURE_COLUMNS)
    assert rec.params["dataset_version"] == "mock_v1"
    assert "mock_similarity_mock_v1.csv" in rec.params["dataset_path"]
    assert rec.params["dataset_similarity_blob_uri"] == blob_uris["similarity"]
    assert rec.params["dataset_embeddings_blob_uri"] == blob_uris["embeddings"]
    assert rec.params["dataset_manifest_blob_uri"] == blob_uris["manifest"]
    assert rec.params["positive_threshold"] == 1.0
    assert rec.params["undersampling_applied"] is False
    assert rec.params["sample_weighting"] == "balanced_binary_cross_entropy"
    assert rec.params["ranking_k"] == 15
    assert rec.params["ranking_group_column"] == "none"
    assert json.loads(rec.params["train_class_counts_before_undersampling"]) == {
        "class_0": 4,
        "class_1": 2,
    }
    assert json.loads(rec.params["train_class_counts_after_undersampling"]) == {
        "class_0": 4,
        "class_1": 2,
    }

    # Core evaluation metrics are logged.
    for key in [
        "auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "precision_at_15",
        "recall_at_15",
        "ndcg_at_15",
        "ranking_eval_groups",
    ]:
        assert key in rec.metrics

    # Confusion matrix is logged as an artifact under the eval/ directory.
    assert any(
        path.endswith("confusion_matrix.csv") and artifact_path == "eval"
        for path, artifact_path in rec.artifacts
    )

    # Model is logged once, with skops serialization.
    assert len(rec.models) == 1
    model_info = rec.models[0]
    assert model_info["name"] == "model"
    assert model_info["serialization_format"] == "skops"
    assert "scikit-learn>=1.5.0" in model_info["pip_requirements"]


def test_train_logreg_uses_request_groups_for_ranking_metrics(tmp_path, monkeypatch) -> None:
    df = make_mock_ranking_dataset(n_rows=100, seed=7, candidates_per_user=10).similarity_df
    _patch_dataset_store(monkeypatch, train_logreg_mlflow, df, tmp_path=Path(tmp_path))
    rec = _patch_mlflow(monkeypatch, train_logreg_mlflow)

    train_logreg_mlflow.train_and_log(
        dataset_version="latest",
        seed=7,
        test_size=0.2,
        positive_threshold=1.0,
        experiment_name="career-copilot-ranking",
        run_name="logreg-grouped",
        max_iter=100,
        c=1.0,
        undersample=False,
        ranking_k=15,
    )

    assert rec.params["ranking_group_column"] == "request_id"
    assert rec.metrics["ranking_eval_groups"] > 1.0


def test_train_xgboost_mlflow_logs_params_metrics_and_artifacts(tmp_path, monkeypatch) -> None:
    df = _make_small_similarity_df()
    _patch_dataset_store(monkeypatch, train_xgboost_mlflow, df, tmp_path=Path(tmp_path))
    rec = _patch_mlflow(monkeypatch, train_xgboost_mlflow)

    train_xgboost_mlflow.train_and_log(
        dataset_version="latest",
        seed=7,
        test_size=0.25,
        positive_threshold=1.0,
        experiment_name="career-copilot-ranking",
        run_name="xgboost-baseline",
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        undersample=True,
        ranking_k=15,
    )

    assert rec.tracking_uris
    assert rec.tracking_uris[0].startswith("sqlite:///")

    assert rec.params["model_type"] == "xgboost"
    logged_features = json.loads(rec.params["features"])
    assert set(logged_features) == set(FEATURE_COLUMNS)
    assert rec.params["dataset_version"] == "mock_v1"
    assert "mock_similarity_mock_v1.csv" in rec.params["dataset_path"]
    assert rec.params["positive_threshold"] == 1.0
    assert rec.params["undersampling_applied"] is True
    assert rec.params["sample_weighting"] == "balanced_binary_cross_entropy"
    assert rec.params["ranking_k"] == 15
    assert rec.params["ranking_group_column"] == "none"
    assert json.loads(rec.params["train_class_counts_before_undersampling"]) == {
        "class_0": 4,
        "class_1": 2,
    }
    assert json.loads(rec.params["train_class_counts_after_undersampling"]) == {
        "class_0": 2,
        "class_1": 2,
    }

    for key in [
        "auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "precision_at_15",
        "recall_at_15",
        "ndcg_at_15",
        "ranking_eval_groups",
    ]:
        assert key in rec.metrics

    assert any(
        path.endswith("confusion_matrix.csv") and artifact_path == "eval"
        for path, artifact_path in rec.artifacts
    )

    assert len(rec.models) == 1
    model_info = rec.models[0]
    assert model_info["name"] == "model"
    assert model_info["serialization_format"] == "cloudpickle"
    assert "xgboost>=2.0.0" in model_info["pip_requirements"]
    assert "scikit-learn>=1.5.0" in model_info["pip_requirements"]


def test_train_logreg_raises_on_single_class_after_threshold(tmp_path, monkeypatch) -> None:
    # All weak labels below the positive_threshold -> only class 0 after binarisation.
    df = _make_small_similarity_df()
    df["label"] = 0.0

    _patch_dataset_store(monkeypatch, train_logreg_mlflow, df, tmp_path=Path(tmp_path))
    _patch_mlflow(monkeypatch, train_logreg_mlflow)

    from pytest import raises

    with raises(RuntimeError) as excinfo:
        train_logreg_mlflow.train_and_log(
            dataset_version="latest",
            seed=7,
            test_size=0.25,
            positive_threshold=1.0,
            experiment_name="career-copilot-ranking",
            run_name=None,
            max_iter=50,
            c=1.0,
            undersample=False,
            ranking_k=15,
        )

    msg = str(excinfo.value)
    assert "single class after thresholding" in msg
    assert "positive-threshold" in msg


def test_save_version_uploads_training_data_and_manifest_to_blob(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CAREER_COPILOT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAINING_DATA_STORAGE_MODE", "blob")
    monkeypatch.setenv("AZURE_TRAINING_DATA_CONTAINER", "training-data")
    monkeypatch.setenv("AZURE_TRAINING_DATA_PREFIX", "ranking")
    monkeypatch.setenv(
        "AZURE_STORAGE_CONNECTION_STRING",
        "DefaultEndpointsProtocol=https;AccountName=testacct;AccountKey=fake;EndpointSuffix=core.windows.net",
    )

    uploads: list[tuple[str, str, bytes, bool]] = []

    class _FakeBlobClient:
        def __init__(self, *, container: str, blob: str) -> None:
            self.container = container
            self.blob = blob

        def upload_blob(self, content: bytes, *, overwrite: bool) -> None:
            uploads.append((self.container, self.blob, content, overwrite))

    class _FakeBlobService:
        def create_container(self, container: str) -> None:
            assert container == "training-data"

        def get_blob_client(self, *, container: str, blob: str) -> _FakeBlobClient:
            return _FakeBlobClient(container=container, blob=blob)

    monkeypatch.setattr(dataset_store, "_blob_service", lambda: _FakeBlobService())

    df = _make_small_similarity_df()
    version = dataset_store.save_version(df, df, version="vblob", n_rows=len(df))

    assert version == "vblob"
    uploaded_blobs = [blob for _, blob, _, _ in uploads]
    assert uploaded_blobs == [
        "ranking/vblob/mock_similarity_vblob.csv",
        "ranking/vblob/mock_embeddings_vblob.csv",
        "ranking/manifest.json",
    ]
    assert all(container == "training-data" for container, _, _, _ in uploads)
    assert all(overwrite is True for _, _, _, overwrite in uploads)

    meta = dataset_store.get_meta("vblob")
    assert meta["blob_uris"] == {
        "similarity": "https://testacct.blob.core.windows.net/training-data/ranking/vblob/mock_similarity_vblob.csv",
        "embeddings": "https://testacct.blob.core.windows.net/training-data/ranking/vblob/mock_embeddings_vblob.csv",
        "manifest": "https://testacct.blob.core.windows.net/training-data/ranking/manifest.json",
    }


def test_get_mlflow_tracking_uri_prefers_env(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example/api")
    assert mlflow_tracking.get_mlflow_tracking_uri() == "https://mlflow.example/api"


def test_get_mlflow_tracking_uri_postgresql_uses_psycopg3_driver(monkeypatch) -> None:
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "postgresql://user:pass@host:5432/mlflow?sslmode=require",
    )
    assert mlflow_tracking.get_mlflow_tracking_uri() == (
        "postgresql+psycopg://user:pass@host:5432/mlflow?sslmode=require"
    )


def test_get_mlflow_tracking_uri_postgresql_explicit_driver_unchanged(monkeypatch) -> None:
    uri = "postgresql+psycopg2://user:pass@host:5432/mlflow"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    assert mlflow_tracking.get_mlflow_tracking_uri() == uri


def test_get_mlflow_tracking_uri_default_sqlite(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    def _fake_data_dir() -> Path:
        d = tmp_path / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(mlflow_tracking, "get_data_dir", _fake_data_dir)
    uri = mlflow_tracking.get_mlflow_tracking_uri()
    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")
