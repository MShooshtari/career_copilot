from __future__ import annotations

from datetime import UTC, datetime, timedelta

from career_copilot.ml.inference import _build_feature_frame_from_candidates
from career_copilot.ml.ranking_dataset import FEATURE_COLUMNS, make_mock_ranking_dataset
from career_copilot.ml.reranking import rerank_with_diversity_and_exploration


def test_mock_ranking_dataset_includes_freshness_features() -> None:
    ds = make_mock_ranking_dataset(n_rows=100, seed=7, candidates_per_user=25)

    for column in ["user_id", "request_id", "job_id"]:
        assert column in ds.similarity_df.columns
        assert column in ds.embeddings_df.columns
    for column in ["days_since_posted", "is_new", "decay_score"]:
        assert column in ds.similarity_df.columns
        assert column in FEATURE_COLUMNS

    assert ds.similarity_df["user_id"].nunique() == 4
    assert ds.similarity_df.groupby("request_id").size().tolist() == [25, 25, 25, 25]
    assert ds.similarity_df["job_id"].is_unique
    assert ds.similarity_df["days_since_posted"].min() >= 0
    assert set(ds.similarity_df["is_new"].unique()).issubset({0, 1})
    assert ds.similarity_df["decay_score"].between(0.0, 1.0).all()
    assert ds.label_scheme == "weak_supervision_v2"


def test_inference_feature_frame_computes_freshness_from_posted_at() -> None:
    now = datetime(2026, 4, 28, tzinfo=UTC)
    df = _build_feature_frame_from_candidates(
        [
            {
                "distance": 0.5,
                "metadata": {"posted_at": (now - timedelta(days=2)).isoformat()},
            },
            {"distance": 1.0, "metadata": {"posted_at": None}},
        ],
        now=now,
    )

    assert list(df.columns) == FEATURE_COLUMNS
    assert df.loc[0, "days_since_posted"] == 2.0
    assert df.loc[0, "is_new"] == 1.0
    assert 0.0 < df.loc[0, "decay_score"] < 1.0
    assert df.loc[1, "days_since_posted"] == 180.0
    assert df.loc[1, "is_new"] == 0.0
    assert df.loc[1, "decay_score"] == 0.0


def test_rerank_adds_diversity_and_exploration_slots() -> None:
    candidates = [
        {
            "id": "python-1",
            "model_score": 0.99,
            "metadata": {
                "title": "Senior Python Engineer",
                "ai_extracted_skills": "python, fastapi",
            },
        },
        {
            "id": "python-2",
            "model_score": 0.98,
            "metadata": {"title": "Python API Engineer", "ai_extracted_skills": "python, fastapi"},
        },
        {
            "id": "data-1",
            "model_score": 0.90,
            "metadata": {"title": "Data Engineer", "ai_extracted_skills": "sql, airflow"},
        },
        {
            "id": "frontend-1",
            "model_score": 0.30,
            "metadata": {"title": "Frontend Engineer", "ai_extracted_skills": "react, typescript"},
        },
    ]

    reranked = rerank_with_diversity_and_exploration(
        candidates,
        window_size=3,
        diversity_penalty=0.25,
        category_penalty=0.05,
        exploration_rate=1 / 3,
        seed=1,
    )

    assert len(reranked) == 3
    assert reranked[0]["id"] == "python-1"
    assert "data-1" in {item["id"] for item in reranked}
    assert sum(item.get("rerank_reason") == "explore" for item in reranked) == 1
