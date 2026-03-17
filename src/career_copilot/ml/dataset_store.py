"""Versioned dataset store: single source of truth for ranking training data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RANKING_STORE_DIR_NAME = "ranking"
MANIFEST_FILENAME = "manifest.json"


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


def get_path(version: str) -> Path:
    """Resolve version to full path to the CSV file. 'latest' → latest version."""
    manifest = _read_manifest()
    versions = manifest.get("versions", [])
    if not versions:
        raise FileNotFoundError("No dataset versions in store. Create one with create_ranking_dataset.")
    resolved = manifest.get("latest") if version == "latest" else version
    if resolved not in versions:
        raise FileNotFoundError(f"Dataset version '{version}' not found. Available: {versions}")
    return _ranking_store_dir() / f"{resolved}.csv"


def load(version: str = "latest") -> tuple[pd.DataFrame, str]:
    """
    Load dataset by version. Returns (dataframe, resolved_version).
    """
    path = get_path(version)
    resolved = path.stem
    df = pd.read_csv(path)
    return df, resolved


def get_meta(version: str) -> dict:
    """Return metadata for a version (n_rows, label_scheme, created_at). Resolves 'latest'."""
    path = get_path(version)
    resolved = path.stem
    manifest = _read_manifest()
    return manifest.get("meta", {}).get(resolved, {})


def save_version(
    df: pd.DataFrame,
    version: str | None = None,
    *,
    n_rows: int | None = None,
    label_scheme: str = "weak_supervision_v1",
) -> str:
    """
    Save a dataset as a new version. If version is None, assign next (v1, v2, ...).
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

    path = store / f"{version}.csv"
    df.to_csv(path, index=False)

    meta = manifest.get("meta", {})
    meta[version] = {
        "n_rows": int(n_rows) if n_rows is not None else len(df),
        "label_scheme": label_scheme,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    versions = versions + [version]
    manifest["versions"] = versions
    manifest["latest"] = version
    manifest["meta"] = meta
    _write_manifest(manifest)
    return version
