"""Versioned dataset store: single source of truth for ranking training data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

RANKING_STORE_DIR_NAME = "ranking"
MANIFEST_FILENAME = "manifest.json"

# Per-version we store two files: similarity features (for tree/linear) and raw embeddings (for NN).
DatasetKind = Literal["similarity", "embeddings"]

MOCK_SIMILARITY_PREFIX = "mock_similarity"
MOCK_EMBEDDINGS_PREFIX = "mock_embeddings"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ranking_store_dir() -> Path:
    root = _project_root()
    store = root / "data" / "datasets" / RANKING_STORE_DIR_NAME
    store.mkdir(parents=True, exist_ok=True)
    return store


def _manifest_path() -> Path:
    return _ranking_store_dir() / MANIFEST_FILENAME


def _read_manifest() -> dict:
    path = _manifest_path()
    if not path.exists():
        return {"versions": [], "latest": None, "meta": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(manifest: dict) -> None:
    with open(_manifest_path(), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def list_versions() -> list[str]:
    """Return ordered list of dataset versions (oldest first)."""
    manifest = _read_manifest()
    return list(manifest.get("versions", []))


def get_path(version: str, kind: DatasetKind = "similarity") -> Path:
    """
    Resolve version to full path. 'latest' → latest version.
    kind: 'similarity' → mock_similarity_vN.csv, 'embeddings' → mock_embeddings_vN.csv.
    """
    manifest = _read_manifest()
    versions = manifest.get("versions", [])
    if not versions:
        raise FileNotFoundError("No dataset versions in store. Create one with create_ranking_dataset.")
    resolved = manifest.get("latest") if version == "latest" else version
    if resolved not in versions:
        raise FileNotFoundError(f"Dataset version '{version}' not found. Available: {versions}")
    prefix = MOCK_SIMILARITY_PREFIX if kind == "similarity" else MOCK_EMBEDDINGS_PREFIX
    return _ranking_store_dir() / f"{prefix}_{resolved}.csv"


def load(version: str = "latest", kind: DatasetKind = "similarity") -> tuple[pd.DataFrame, str]:
    """
    Load dataset by version and kind. Returns (dataframe, resolved_version).
    kind: 'similarity' for tree/linear models, 'embeddings' for neural networks.
    """
    path = get_path(version, kind=kind)
    resolved = path.stem.replace(f"{MOCK_SIMILARITY_PREFIX}_", "").replace(f"{MOCK_EMBEDDINGS_PREFIX}_", "")
    df = pd.read_csv(path)
    return df, resolved


def get_meta(version: str) -> dict:
    """Return metadata for a version (n_rows, label_scheme, created_at). Resolves 'latest'."""
    manifest = _read_manifest()
    versions = manifest.get("versions", [])
    if not versions:
        return {}
    resolved = manifest.get("latest") if version == "latest" else version
    if resolved not in versions:
        return {}
    return manifest.get("meta", {}).get(resolved, {})


def save_version(
    similarity_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    version: str | None = None,
    *,
    n_rows: int | None = None,
    label_scheme: str = "weak_supervision_v1",
) -> str:
    """
    Save both similarity and embeddings datasets as a new version.
    Files: mock_similarity_vN.csv, mock_embeddings_vN.csv.
    Returns the version string used.
    """
    store = _ranking_store_dir()
    manifest = _read_manifest()
    versions = manifest.get("versions", [])

    if version is None:
        n = len(versions) + 1
        version = f"v{n}"

    if version in versions:
        raise ValueError(f"Version '{version}' already exists. Use a new version or overwrite explicitly.")

    n = int(n_rows) if n_rows is not None else len(similarity_df)
    (store / f"{MOCK_SIMILARITY_PREFIX}_{version}.csv").parent.mkdir(parents=True, exist_ok=True)
    similarity_df.to_csv(store / f"{MOCK_SIMILARITY_PREFIX}_{version}.csv", index=False)
    embeddings_df.to_csv(store / f"{MOCK_EMBEDDINGS_PREFIX}_{version}.csv", index=False)

    meta = manifest.get("meta", {})
    meta[version] = {
        "n_rows": n,
        "label_scheme": label_scheme,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    versions = versions + [version]
    manifest["versions"] = versions
    manifest["latest"] = version
    manifest["meta"] = meta
    _write_manifest(manifest)
    return version
