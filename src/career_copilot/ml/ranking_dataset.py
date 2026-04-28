"""
Mock ranking dataset for job–resume fit.

Label = weak supervision from a latent ranking utility that combines job/resume
similarity with freshness. Features are similarity scores, freshness signals, and
other scalar ranking signals.

Two outputs per run:
- Similarity dataset: for tree-based / linear models (title_similarity, skill_*, etc.).
- Embeddings dataset: same rows, raw embedding dimensions + label for neural networks.
  Embedding groups: job/resume summary, user/company location, preferred locations,
  titles, LLM-extracted skills, work mode, employment type, preferred roles.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

LabelScheme = Literal["weak_supervision_v1", "weak_supervision_v2"]

# Embedding dimension for mock (each embedding group is d-dimensional).
MOCK_EMBEDDING_DIM = 16

# Embedding groups in the embeddings dataset (prefix for columns: {prefix}_0, ..., {prefix}_{d-1}).
# Pairs are (resume/user side, job/company side) for matching.
EMBEDDING_GROUPS = [
    ("job_emb", "resume_emb"),  # job summary, resume summary
    ("company_location_emb", "user_location_emb"),  # current location
    ("job_preferred_locations_emb", "user_preferred_locations_emb"),
    ("job_title_emb", "resume_title_emb"),
    ("job_preferred_roles_emb", "user_preferred_roles_emb"),
    ("job_skills_emb", "resume_skills_emb"),  # LLM-extracted skills
    ("job_work_mode_emb", "resume_work_mode_emb"),
    ("job_employment_type_emb", "resume_employment_type_emb"),
]

# For tree-based / linear models: embedding similarity + other similarity/scalar features.
FEATURE_COLUMNS = [
    "embedding_similarity",
    "title_similarity",
    "skill_overlap_count",
    "location_match",
    "experience_gap",
    "salary_match",
    "location_km",
    "skill_similarity",
    "role_similarity",
    "work_mode_similarity",
    "employment_type_similarity",
    "preferred_locations_similarity",
    "days_since_posted",
    "is_new",
    "decay_score",
]

# For preprocessing: numeric features get StandardScaler, passthrough stay as-is.
PASSTHROUGH_FEATURE_NAMES = [c for c in ("location_match", "is_new") if c in FEATURE_COLUMNS]
NUMERIC_FEATURE_NAMES = [c for c in FEATURE_COLUMNS if c not in PASSTHROUGH_FEATURE_NAMES]


def _weak_label(similarity: float) -> float:
    if similarity > 0.8:
        return 1.0
    if similarity >= 0.6:
        return 0.5
    return 0.0


def _random_unit_vector(rng: np.random.Generator, d: int, size: int) -> np.ndarray:
    x = rng.standard_normal((size, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(axis=1)


def _correlated_pair(
    rng: np.random.Generator,
    d: int,
    n_rows: int,
    mix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (a, b) as unit vectors with b = mix*a + (1-mix)*noise so cosine_sim(a,b) ~ mix."""
    a = _random_unit_vector(rng, d, n_rows)
    noise = _random_unit_vector(rng, d, n_rows)
    b = a * mix + noise * (1 - mix)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a, b


@dataclass(frozen=True)
class MockRankingDatasets:
    """
    Two datasets from the same mock generation:
    - similarity_df: similarity-based features + label (for LogReg / tree models).
    - embeddings_df: raw embedding columns for all groups (summary, location, titles,
      skills, work mode, employment type, preferred roles/locations) + label (for NN).
    """

    similarity_df: pd.DataFrame
    embeddings_df: pd.DataFrame
    label_scheme: LabelScheme
    dataset_version: str


def make_mock_ranking_dataset(
    *,
    n_rows: int = 2000,
    seed: int = 7,
    label_scheme: LabelScheme = "weak_supervision_v2",
    embedding_dim: int = MOCK_EMBEDDING_DIM,
) -> MockRankingDatasets:
    """
    Generate mock data with label = binned latent utility. Utility combines
    relevance and freshness so the model can learn that sometimes freshness wins
    and sometimes strong relevance wins.

    Returns similarity DataFrame and embeddings DataFrame.
    """
    rng = np.random.default_rng(seed)
    d = embedding_dim

    # Relevance signal: similarity between job-summary and resume-summary embeddings (mock).
    job_emb = _random_unit_vector(rng, d, n_rows)
    resume_emb = _random_unit_vector(rng, d, n_rows)
    mix = rng.uniform(0.3, 0.95, (n_rows, 1))
    resume_emb = resume_emb * (1 - mix) + job_emb * mix
    resume_emb = resume_emb / np.linalg.norm(resume_emb, axis=1, keepdims=True)
    label_sim = np.clip(_cosine_similarity(job_emb, resume_emb), 0.0, 1.0)

    # All embedding pairs (job/resume or company/user side). First pair = summary; rest correlated with mix.
    pair_arrays: list[tuple[np.ndarray, np.ndarray]] = [(job_emb, resume_emb)]
    for _ in range(len(EMBEDDING_GROUPS) - 1):
        job_side, resume_side = _correlated_pair(rng, d, n_rows, mix)
        pair_arrays.append((job_side, resume_side))

    # Features: embedding_similarity (job–resume summary cosine sim) + others correlated with it.
    t = label_sim
    title_similarity = np.clip(0.4 * t + 0.3 + rng.normal(0, 0.2, n_rows), 0.0, 1.0)
    skill_overlap_count = np.clip((t * 12).astype(int) + rng.integers(-2, 3, n_rows), 0, 20)
    location_match_prob = np.clip(0.2 + 0.6 * t, 0.05, 0.95)
    location_match = rng.binomial(1, location_match_prob, n_rows).astype(int)
    experience_gap = np.clip(rng.normal(2.5 - 2.5 * t, 1.2, n_rows), 0, 10)
    salary_match = np.clip(0.3 + 0.6 * t + rng.normal(0, 0.15, n_rows), 0.0, 1.0)
    location_km = np.clip(rng.exponential(50 - 35 * t, n_rows), 0, 500)
    skill_similarity = np.clip(0.35 + 0.5 * t + rng.normal(0, 0.15, n_rows), 0.0, 1.0)
    role_similarity = np.clip(0.4 + 0.5 * t + rng.normal(0, 0.12, n_rows), 0.0, 1.0)
    work_mode_similarity = np.clip(0.5 + 0.4 * t + rng.normal(0, 0.1, n_rows), 0.0, 1.0)
    employment_type_similarity = np.clip(0.45 + 0.45 * t + rng.normal(0, 0.1, n_rows), 0.0, 1.0)
    preferred_locations_similarity = np.clip(0.3 + 0.55 * t + rng.normal(0, 0.15, n_rows), 0.0, 1.0)

    # Freshness: relevant jobs skew newer, but age still has independent noise.
    # This gives the model examples where freshness can break ties without simply
    # turning recommendations into "newest first".
    age_scale_days = np.clip(55 - 35 * t, 7, 60)
    days_since_posted = np.clip(rng.exponential(age_scale_days), 0, 180)
    is_new = (days_since_posted <= 3).astype(int)
    freshness_decay_lambda = 0.05
    decay_score = np.exp(-freshness_decay_lambda * days_since_posted)

    latent_utility = np.clip(
        0.72 * label_sim + 0.20 * decay_score + 0.08 * is_new + rng.normal(0, 0.06, n_rows),
        0.0,
        1.0,
    )
    labels = np.array([_weak_label(float(s)) for s in latent_utility], dtype=float)

    similarity_df = pd.DataFrame(
        {
            "embedding_similarity": label_sim.astype(float),
            "title_similarity": title_similarity.astype(float),
            "skill_overlap_count": skill_overlap_count.astype(int),
            "location_match": location_match.astype(int),
            "experience_gap": experience_gap.astype(float),
            "salary_match": salary_match.astype(float),
            "location_km": location_km.astype(float),
            "skill_similarity": skill_similarity.astype(float),
            "role_similarity": role_similarity.astype(float),
            "work_mode_similarity": work_mode_similarity.astype(float),
            "employment_type_similarity": employment_type_similarity.astype(float),
            "preferred_locations_similarity": preferred_locations_similarity.astype(float),
            "days_since_posted": days_since_posted.astype(float),
            "is_new": is_new.astype(int),
            "decay_score": decay_score.astype(float),
            "label": labels.astype(float),
        }
    )

    # Embeddings dataset: all groups flattened as columns (job_*_i, resume_*_i per group) + label.
    emb_cols: list[str] = []
    emb_blocks: list[np.ndarray] = []
    for (job_prefix, resume_prefix), (job_arr, resume_arr) in zip(
        EMBEDDING_GROUPS,
        pair_arrays,
        strict=False,
    ):
        emb_cols.extend([f"{job_prefix}_{i}" for i in range(d)])
        emb_cols.extend([f"{resume_prefix}_{i}" for i in range(d)])
        emb_blocks.append(job_arr)
        emb_blocks.append(resume_arr)
    emb_data = np.hstack(emb_blocks)
    embeddings_df = pd.DataFrame({c: emb_data[:, i].astype(float) for i, c in enumerate(emb_cols)})
    embeddings_df["label"] = labels.astype(float)

    hasher = hashlib.sha256()
    hasher.update(label_scheme.encode("utf-8"))
    hasher.update(str(seed).encode("utf-8"))
    hasher.update(str(n_rows).encode("utf-8"))
    hasher.update(
        ",".join(
            EMBEDDING_GROUPS[i][j] for i in range(len(EMBEDDING_GROUPS)) for j in range(2)
        ).encode("utf-8")
    )
    hasher.update(pd.util.hash_pandas_object(similarity_df, index=True).values.tobytes())
    hasher.update(pd.util.hash_pandas_object(embeddings_df, index=True).values.tobytes())
    version = hasher.hexdigest()[:16]

    return MockRankingDatasets(
        similarity_df=similarity_df,
        embeddings_df=embeddings_df,
        label_scheme=label_scheme,
        dataset_version=version,
    )
