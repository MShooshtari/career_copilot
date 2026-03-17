"""CLI to create a new versioned mock ranking dataset and save it to the dataset store."""

from __future__ import annotations

import argparse

from career_copilot.ml.dataset_store import save_version
from career_copilot.ml.ranking_dataset import make_mock_ranking_dataset


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create a versioned mock ranking dataset (similarity + embeddings) in data/datasets/ranking/."
    )
    p.add_argument("--version", type=str, default=None, help="Version label (e.g. v1, v2). Auto if not set.")
    p.add_argument("--n-rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    ds = make_mock_ranking_dataset(n_rows=args.n_rows, seed=args.seed)
    version = save_version(
        ds.similarity_df,
        ds.embeddings_df,
        version=args.version,
        n_rows=len(ds.similarity_df),
        label_scheme=ds.label_scheme,
    )
    print(f"Saved mock dataset version: {version}")
    print("  - mock_similarity_{}.csv (for tree-based / linear models)".format(version))
    print("  - mock_embeddings_{}.csv (for neural networks)".format(version))
