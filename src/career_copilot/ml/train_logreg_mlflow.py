from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from career_copilot.ml.dataset_store import get_meta, get_path, load
from career_copilot.ml.ranking_dataset import FEATURE_COLUMNS


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_mlflow_local_store() -> Path:
    root = _project_root()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    df.to_csv(path, index=True)


def train_and_log(
    *,
    dataset_version: str,
    seed: int,
    test_size: float,
    positive_threshold: float,
    experiment_name: str,
    run_name: str | None,
    max_iter: int,
    c: float,
) -> None:
    data_dir = _ensure_mlflow_local_store()

    # Use a local SQLite backend store (more future-proof than the deprecated filesystem store).
    db_path = (data_dir / "mlflow.db").resolve()
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    artifacts_dir = (data_dir / "mlflow_artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(name=experiment_name, artifact_location=artifacts_dir.as_uri())
    mlflow.set_experiment(experiment_name)

    df, resolved_version = load(dataset_version)
    dataset_path = get_path(dataset_version).resolve().as_posix()
    meta = get_meta(dataset_version)
    label_scheme = meta.get("label_scheme", "weak_supervision_v1")

    # Convert weak labels {0, 0.5, 1} to binary; keep original label as sample weight
    y_weak = df["label"].astype(float).to_numpy()
    y = (y_weak >= positive_threshold).astype(int)
    sample_weight = np.clip(y_weak, 0.0, 1.0)

    if np.unique(y).size < 2:
        raise RuntimeError(
            "Mock dataset produced a single class after thresholding. "
            "Increase --n-rows or adjust --positive-threshold."
        )

    X = df[FEATURE_COLUMNS]
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weight, test_size=test_size, random_state=seed, stratify=y
    )

    numeric_features = [
        "embedding_similarity",
        "title_similarity",
        "skill_overlap_count",
        "experience_gap",
    ]
    passthrough_features = ["location_match"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("passthrough", "passthrough", passthrough_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        max_iter=max_iter,
        C=c,
        solver="lbfgs",
        n_jobs=None,
        random_state=seed,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("features", json.dumps(FEATURE_COLUMNS))
        mlflow.log_param("label_scheme", label_scheme)
        mlflow.log_param("dataset_version", resolved_version)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("seed", seed)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("positive_threshold", positive_threshold)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", c)

        pipe.fit(X_train, y_train, clf__sample_weight=w_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        cm = confusion_matrix(y_test, pred)

        mlflow.log_metric("auc", float(auc))
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall", float(rec))
        mlflow.log_metric("f1", float(f1))

        # Artifacts
        artifacts_dir = Path("artifacts_tmp")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cm_path = artifacts_dir / "confusion_matrix.csv"
        _save_confusion_matrix(cm, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="eval")

        mlflow.sklearn.log_model(
            pipe,
            name="model",
            serialization_format="skops",
            pip_requirements=["scikit-learn>=1.5.0", "skops>=0.13.0"],
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Train ranking baseline and log to MLflow (local).")
    p.add_argument(
        "--dataset-version",
        type=str,
        default="latest",
        help="Versioned dataset to load from data/datasets/ranking/ (e.g. v1, latest).",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--positive-threshold",
        type=float,
        default=0.5,
        help="Binary target is weak_label >= threshold. Use 0.5 to treat weak+strong as positive.",
    )
    p.add_argument("--experiment", type=str, default="career-copilot-ranking")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--c", type=float, default=1.0)
    args = p.parse_args()

    train_and_log(
        dataset_version=args.dataset_version,
        seed=args.seed,
        test_size=args.test_size,
        positive_threshold=args.positive_threshold,
        experiment_name=args.experiment,
        run_name=args.run_name,
        max_iter=args.max_iter,
        c=args.c,
    )


if __name__ == "__main__":
    main()

