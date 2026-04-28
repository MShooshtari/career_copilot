"""CLI to create a new versioned mock ranking dataset and save it to the dataset store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from career_copilot.database.db import load_env
from career_copilot.ml.dataset_store import get_path, get_store_dir, save_version
from career_copilot.ml.ranking_dataset import make_mock_ranking_dataset


def _log(msg: str) -> None:
    """Print to stderr so output is visible even if stdout is captured."""
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    load_env()
    _log("create_ranking_dataset: starting")
    p = argparse.ArgumentParser(
        description="Create a versioned mock ranking dataset (similarity + embeddings) in data/datasets/ranking/."
    )
    p.add_argument(
        "--version", type=str, default=None, help="Version label (e.g. v1, v2). Auto if not set."
    )
    p.add_argument("--n-rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--candidates-per-user",
        type=int,
        default=100,
        help="Number of candidate jobs per mock user/request group.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Force project root (data/datasets/ranking will be under this). Overrides CAREER_COPILOT_PROJECT_ROOT.",
    )
    args = p.parse_args()

    if args.output_dir:
        import os

        os.environ["CAREER_COPILOT_PROJECT_ROOT"] = args.output_dir

    store_dir = get_store_dir().resolve()
    _log(f"Store directory: {store_dir}")

    _log(
        "Creating dataset "
        f"(n_rows={args.n_rows}, seed={args.seed}, "
        f"candidates_per_user={args.candidates_per_user}) ..."
    )
    ds = make_mock_ranking_dataset(
        n_rows=args.n_rows,
        seed=args.seed,
        candidates_per_user=args.candidates_per_user,
    )
    version = save_version(
        ds.similarity_df,
        ds.embeddings_df,
        version=args.version,
        n_rows=len(ds.similarity_df),
        label_scheme=ds.label_scheme,
    )
    sim_path = get_path(version, kind="similarity").resolve()
    emb_path = get_path(version, kind="embeddings").resolve()
    ok_sim = sim_path.exists()
    ok_emb = emb_path.exists()
    _log(f"Saved mock dataset version: {version}")
    _log(f"  - {sim_path.name} -> {'OK' if ok_sim else 'MISSING'}")
    _log(f"  - {emb_path.name} -> {'OK' if ok_emb else 'MISSING'}")
    if not ok_sim or not ok_emb:
        _log(f"  WARNING: Files not found. Store: {store_dir}")

    # Write paths to a file in cwd so you can see where files went
    try:
        marker = Path.cwd().resolve() / ".last_ranking_dataset.txt"
        marker.write_text(
            f"version={version}\nstore={store_dir}\nsimilarity={sim_path}\nembeddings={emb_path}\n",
            encoding="utf-8",
        )
        _log(f"Paths written to: {marker}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
