from __future__ import annotations

import pytest

from career_copilot.ml.ranking_metrics import ranking_metrics_at_k


def test_ranking_metrics_at_k_for_single_ranked_list() -> None:
    metrics = ranking_metrics_at_k(
        labels=[1.0, 0.0, 1.0, 0.5],
        scores=[0.9, 0.8, 0.7, 0.6],
        k=2,
        positive_threshold=1.0,
    )

    assert metrics["precision_at_2"] == pytest.approx(0.5)
    assert metrics["recall_at_2"] == pytest.approx(0.5)
    assert metrics["ndcg_at_2"] == pytest.approx(0.6131471927654584)
    assert metrics["ranking_eval_groups"] == 1.0


def test_ranking_metrics_macro_average_across_groups() -> None:
    metrics = ranking_metrics_at_k(
        labels=[1.0, 0.0, 0.0, 1.0],
        scores=[0.9, 0.8, 0.95, 0.1],
        k=1,
        positive_threshold=1.0,
        group_ids=["u1", "u1", "u2", "u2"],
    )

    assert metrics["precision_at_1"] == pytest.approx(0.5)
    assert metrics["recall_at_1"] == pytest.approx(0.5)
    assert metrics["ndcg_at_1"] == pytest.approx(0.5)
    assert metrics["ranking_eval_groups"] == 2.0


def test_ranking_metrics_validate_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        ranking_metrics_at_k(labels=[1.0], scores=[0.9, 0.8])
