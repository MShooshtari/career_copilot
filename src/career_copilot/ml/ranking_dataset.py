from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


LabelScheme = Literal["weak_supervision_v1"]


@dataclass(frozen=True)
class RankingDataset:
    """
    Small, bootstrapped ranking dataset.

    Labels are weak supervision derived from embedding similarity:
    - similarity > 0.8  -> 1.0 (strong positive)
    - 0.6..0.8 inclusive -> 0.5 (weak positive)
    - < 0.6            -> 0.0
    """

    df: pd.DataFrame
    label_scheme: LabelScheme
    dataset_version: str


FEATURE_COLUMNS = [
    "embedding_similarity",
    "title_similarity",
    "skill_overlap_count",
    "location_match",
    "experience_gap",
]


def _weak_label(similarity: float) -> float:
    if similarity > 0.8:
        return 1.0
    if similarity >= 0.6:
        return 0.5
    return 0.0


def make_mock_ranking_dataset(
    *,
    n_rows: int = 2000,
    seed: int = 7,
    label_scheme: LabelScheme = "weak_supervision_v1",
) -> RankingDataset:
    """
    Generate a mock dataset with realistic-ish feature correlations.

    This is intentionally fake data: it exists to bootstrap a first ranking model
    and demonstrate experiment tracking and feature engineering.
    """
    rng = np.random.default_rng(seed)

    # Mixture distribution so we *always* have some <0.6 negatives and some strong positives.
    n_low = max(1, int(0.22 * n_rows))
    n_mid = max(1, int(0.56 * n_rows))
    n_high = max(1, n_rows - n_low - n_mid)
    embedding_similarity = np.concatenate(
        [
            rng.beta(2.0, 6.0, size=n_low),   # low-sim tail
            rng.beta(2.4, 2.4, size=n_mid),   # broad middle
            rng.beta(6.0, 2.0, size=n_high),  # high-sim tail
        ]
    )
    rng.shuffle(embedding_similarity)
    title_similarity = np.clip(
        0.55 * embedding_similarity + rng.normal(0.0, 0.18, size=n_rows), 0.0, 1.0
    )

    # Skill overlap: roughly increases with similarity but with big variance
    base_skill = (embedding_similarity * 10).astype(int)
    skill_overlap_count = np.clip(base_skill + rng.integers(-2, 3, size=n_rows), 0, 20)

    # Location match: more likely when similarity is higher (proxy for role fit)
    location_match_prob = np.clip(0.15 + 0.7 * embedding_similarity, 0.05, 0.95)
    location_match = rng.binomial(1, location_match_prob, size=n_rows).astype(int)

    # Experience gap: smaller gap for better matches, but noisy (years)
    experience_gap = np.clip(rng.normal(2.2 - 3.0 * embedding_similarity, 1.5, size=n_rows), 0, 12)

    labels = np.array([_weak_label(float(s)) for s in embedding_similarity], dtype=float)

    df = pd.DataFrame(
        {
            "embedding_similarity": embedding_similarity.astype(float),
            "title_similarity": title_similarity.astype(float),
            "skill_overlap_count": skill_overlap_count.astype(int),
            "location_match": location_match.astype(int),
            "experience_gap": experience_gap.astype(float),
            "label": labels.astype(float),
        }
    )

    # Stable dataset version hash (content + scheme + seed + n_rows)
    hasher = hashlib.sha256()
    hasher.update(label_scheme.encode("utf-8"))
    hasher.update(str(seed).encode("utf-8"))
    hasher.update(str(n_rows).encode("utf-8"))
    hasher.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    version = hasher.hexdigest()[:16]

    return RankingDataset(df=df, label_scheme=label_scheme, dataset_version=version)

