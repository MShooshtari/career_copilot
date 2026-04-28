from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _dcg(relevances: np.ndarray) -> float:
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2, dtype=float))
    gains = np.power(2.0, relevances) - 1.0
    return float(np.sum(gains / discounts))


def _metrics_for_ranked_list(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    k: int,
    positive_threshold: float,
) -> dict[str, float]:
    if labels.size == 0 or scores.size == 0 or k <= 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}

    limit = min(k, labels.size)
    order = np.argsort(scores, kind="mergesort")[::-1]
    ranked_labels = labels[order]
    top_labels = ranked_labels[:limit]

    relevant = labels >= positive_threshold
    top_relevant = top_labels >= positive_threshold
    relevant_count = int(np.sum(relevant))

    precision_at_k = float(np.sum(top_relevant) / k)
    recall_at_k = float(np.sum(top_relevant) / relevant_count) if relevant_count else 0.0

    ideal_labels = np.sort(labels)[::-1][:limit]
    ideal_dcg = _dcg(ideal_labels)
    ndcg_at_k = _dcg(top_labels) / ideal_dcg if ideal_dcg > 0.0 else 0.0

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "ndcg_at_k": float(ndcg_at_k),
    }


def ranking_metrics_at_k(
    labels: Iterable[float],
    scores: Iterable[float],
    *,
    k: int = 15,
    positive_threshold: float = 1.0,
    group_ids: Sequence[object] | None = None,
) -> dict[str, float]:
    """
    Compute ranking metrics for top-k recommendations.

    If group_ids are supplied, metrics are calculated per group, then macro-averaged
    across groups. Each group should represent one user or recommendation request.
    """
    labels_arr = _as_float_array(labels)
    scores_arr = _as_float_array(scores)
    if labels_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError("labels and scores must have the same length")
    if group_ids is not None and len(group_ids) != labels_arr.shape[0]:
        raise ValueError("group_ids must have the same length as labels")

    if labels_arr.size == 0:
        return {
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            f"ndcg_at_{k}": 0.0,
            "ranking_eval_groups": 0.0,
        }

    grouped_indices: list[np.ndarray]
    if group_ids is None:
        grouped_indices = [np.arange(labels_arr.shape[0])]
    else:
        buckets: dict[object, list[int]] = defaultdict(list)
        for idx, group_id in enumerate(group_ids):
            buckets[group_id].append(idx)
        grouped_indices = [np.asarray(indices, dtype=int) for indices in buckets.values()]

    per_group = [
        _metrics_for_ranked_list(
            labels_arr[indices],
            scores_arr[indices],
            k=k,
            positive_threshold=positive_threshold,
        )
        for indices in grouped_indices
    ]

    return {
        f"precision_at_{k}": float(np.mean([m["precision_at_k"] for m in per_group])),
        f"recall_at_{k}": float(np.mean([m["recall_at_k"] for m in per_group])),
        f"ndcg_at_{k}": float(np.mean([m["ndcg_at_k"] for m in per_group])),
        "ranking_eval_groups": float(len(grouped_indices)),
    }
