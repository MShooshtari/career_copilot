# ML

Machine-learning utilities for recommendation ranking experiments and serving.

This package contains dataset creation, feature utilities, MLflow training scripts, inference helpers, and reranking policies used by the recommendation flow.

## Contents

- `create_ranking_dataset.py`, `ranking_dataset.py`, and `dataset_store.py` build and manage ranking datasets.
- `train_logreg_mlflow.py` and `train_xgboost_mlflow.py` train baseline rankers with MLflow tracking.
- `inference.py` loads and applies ranking models.
- `ranking_metrics.py` and `training_utils.py` provide shared evaluation and training helpers.
- `reranking.py` applies post-model diversity and exploration policies.

Keep experiment code reproducible and make serving behavior explicit when moving training features into production paths.
