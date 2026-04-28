from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
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

from career_copilot.ml.dataset_store import get_data_dir, get_meta, get_path, load
from career_copilot.ml.mlflow_tracking import (
    ensure_experiment_for_training,
    get_mlflow_tracking_uri,
)
from career_copilot.ml.ranking_dataset import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_NAMES,
    PASSTHROUGH_FEATURE_NAMES,
)
from career_copilot.ml.training_utils import (
    class_counts,
    make_balanced_sample_weight,
    make_binary_target,
    undersample_majority_class,
)


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
    undersample: bool,
) -> None:
    data_dir = get_data_dir()
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    client = MlflowClient()
    ensure_experiment_for_training(client, experiment_name=experiment_name, data_dir=data_dir)
    mlflow.set_experiment(experiment_name)

    df, resolved_version = load(dataset_version, kind="similarity")
    dataset_path = get_path(dataset_version, kind="similarity").resolve().as_posix()
    meta = get_meta(dataset_version)
    label_scheme = meta.get("label_scheme", "weak_supervision_v1")

    # Binary target: by default only weak label 1.0 is positive; 0.0 and 0.5 are negative.
    y_weak = df["label"].astype(float).to_numpy()
    y = make_binary_target(y_weak, positive_threshold=positive_threshold)

    if np.unique(y).size < 2:
        raise RuntimeError(
            "Mock dataset produced a single class after thresholding. "
            "Increase --n-rows or adjust --positive-threshold."
        )

    # Use only columns present in the dataset (older versions may lack embedding_similarity).
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not feature_cols:
        raise RuntimeError(f"Dataset has none of the expected features: {FEATURE_COLUMNS}")
    X = df[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    train_class_counts_before = class_counts(y_train)
    if undersample:
        X_train, y_train = undersample_majority_class(X_train, y_train, seed=seed)
    train_class_counts_after = class_counts(y_train)
    w_train = make_balanced_sample_weight(y_train)

    numeric_features = [f for f in NUMERIC_FEATURE_NAMES if f in feature_cols]
    passthrough_features = [f for f in PASSTHROUGH_FEATURE_NAMES if f in feature_cols]

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

    params = {
        "model_type": "logistic_regression",
        "features": json.dumps(feature_cols),
        "label_scheme": label_scheme,
        "dataset_version": resolved_version,
        "dataset_path": dataset_path,
        "n_rows": len(df),
        "seed": seed,
        "test_size": test_size,
        "positive_threshold": positive_threshold,
        "undersampling_applied": undersample,
        "train_class_counts_before_undersampling": json.dumps(train_class_counts_before),
        "train_class_counts_after_undersampling": json.dumps(train_class_counts_after),
        "sample_weighting": "balanced_binary_cross_entropy",
        "max_iter": max_iter,
        "C": c,
    }

    with mlflow.start_run(run_name=run_name):
        for key, value in params.items():
            mlflow.log_param(key, value)

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

        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = Path(tmpdir) / "confusion_matrix.csv"
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
        default=1.0,
        help="Binary target: weak_label >= threshold is class 1. Default 1.0 treats only 1.0 as positive.",
    )
    p.add_argument(
        "--undersample",
        action="store_true",
        help="Downsample the majority class in the training split before fitting.",
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
        undersample=args.undersample,
    )


if __name__ == "__main__":
    main()
