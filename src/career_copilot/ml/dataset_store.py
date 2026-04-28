"""Versioned dataset store: single source of truth for ranking training data."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import pandas as pd

RANKING_STORE_DIR_NAME = "ranking"
MANIFEST_FILENAME = "manifest.json"

# Per-version we store two files: similarity features (for tree/linear) and raw embeddings (for NN).
DatasetKind = Literal["similarity", "embeddings"]

MOCK_SIMILARITY_PREFIX = "mock_similarity"
MOCK_EMBEDDINGS_PREFIX = "mock_embeddings"


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _project_root() -> Path:
    """Project root (career_copilot repo). Env CAREER_COPILOT_PROJECT_ROOT overrides."""
    env_root = os.environ.get("CAREER_COPILOT_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists():
        return cwd
    p = Path(__file__).resolve().parent
    for _ in range(8):
        if (p / "pyproject.toml").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return cwd


def get_data_dir() -> Path:
    """Project data directory (e.g. for MLflow). Creates dir if needed."""
    root = _project_root()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def training_data_storage_mode() -> str:
    return (_env("TRAINING_DATA_STORAGE_MODE", "local") or "local").strip().lower()


def _training_data_container() -> str:
    return (_env("AZURE_TRAINING_DATA_CONTAINER", "training-data") or "training-data").strip()


def _training_data_prefix() -> str:
    return (_env("AZURE_TRAINING_DATA_PREFIX", RANKING_STORE_DIR_NAME) or "").strip().strip("/")


def _conn_str() -> str | None:
    v = (_env("AZURE_STORAGE_CONNECTION_STRING") or "").strip()
    return v or None


def _account_name_from_connection_string(conn_str: str | None) -> str | None:
    if not conn_str:
        return None
    for part in conn_str.split(";"):
        if part.startswith("AccountName="):
            return part.split("=", 1)[1]
    return None


def _blob_service():
    from azure.storage.blob import BlobServiceClient

    cs = _conn_str()
    if not cs:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING is required for TRAINING_DATA_STORAGE_MODE=blob"
        )
    return BlobServiceClient.from_connection_string(cs)


def _blob_name(*parts: str) -> str:
    cleaned_parts = [part.strip("/") for part in parts if part.strip("/")]
    prefix = _training_data_prefix()
    if prefix:
        cleaned_parts = [prefix, *cleaned_parts]
    return "/".join(cleaned_parts)


def _blob_uri(*, container: str, blob_name: str) -> str:
    account_name = _account_name_from_connection_string(_conn_str())
    quoted_blob_name = quote(blob_name, safe="/")
    if account_name:
        return f"https://{account_name}.blob.core.windows.net/{container}/{quoted_blob_name}"
    return f"azure://{container}/{quoted_blob_name}"


def _upload_file_to_training_data_container(*, path: Path, blob_name: str) -> str:
    from azure.core.exceptions import ResourceExistsError

    container = _training_data_container()
    service = _blob_service()
    try:
        service.create_container(container)
    except ResourceExistsError:
        pass
    client = service.get_blob_client(container=container, blob=blob_name)
    client.upload_blob(path.read_bytes(), overwrite=True)
    return _blob_uri(container=container, blob_name=blob_name)


def _ranking_store_dir() -> Path:
    root = _project_root()
    store = root / "data" / "datasets" / RANKING_STORE_DIR_NAME
    store.mkdir(parents=True, exist_ok=True)
    return store


def get_store_dir() -> Path:
    """Return the ranking dataset store directory (for CLI / debugging)."""
    return _ranking_store_dir()


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


def _resolve_version(version: str, manifest: dict) -> str:
    """Resolve 'latest' to concrete version and validate. Raises if not found."""
    versions = manifest.get("versions", [])
    if not versions:
        raise FileNotFoundError(
            "No dataset versions in store. Create one with create_ranking_dataset."
        )
    resolved = manifest.get("latest") if version == "latest" else version
    if resolved not in versions:
        raise FileNotFoundError(f"Dataset version '{version}' not found. Available: {versions}")
    return resolved


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
    resolved = _resolve_version(version, manifest)
    prefix = MOCK_SIMILARITY_PREFIX if kind == "similarity" else MOCK_EMBEDDINGS_PREFIX
    return _ranking_store_dir() / f"{prefix}_{resolved}.csv"


def load(version: str = "latest", kind: DatasetKind = "similarity") -> tuple[pd.DataFrame, str]:
    """
    Load dataset by version and kind. Returns (dataframe, resolved_version).
    kind: 'similarity' for tree/linear models, 'embeddings' for neural networks.
    """
    path = get_path(version, kind=kind)
    resolved = path.stem.replace(f"{MOCK_SIMILARITY_PREFIX}_", "").replace(
        f"{MOCK_EMBEDDINGS_PREFIX}_", ""
    )
    df = pd.read_csv(path)
    return df, resolved


def get_meta(version: str) -> dict:
    """Return metadata for a version (n_rows, label_scheme, created_at). Resolves 'latest'."""
    manifest = _read_manifest()
    try:
        resolved = _resolve_version(version, manifest)
    except FileNotFoundError:
        return {}
    return manifest.get("meta", {}).get(resolved, {})


def get_blob_uris(version: str) -> dict[str, str]:
    """Return online training data URIs for a version, if the manifest has them."""
    meta = get_meta(version)
    blob_uris = meta.get("blob_uris", {})
    return dict(blob_uris) if isinstance(blob_uris, dict) else {}


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
        raise ValueError(
            f"Version '{version}' already exists. Use a new version or overwrite explicitly."
        )

    n = int(n_rows) if n_rows is not None else len(similarity_df)
    similarity_path = store / f"{MOCK_SIMILARITY_PREFIX}_{version}.csv"
    embeddings_path = store / f"{MOCK_EMBEDDINGS_PREFIX}_{version}.csv"
    similarity_df.to_csv(similarity_path, index=False)
    embeddings_df.to_csv(embeddings_path, index=False)

    blob_uris: dict[str, str] = {}
    if training_data_storage_mode() == "blob":
        blob_uris = {
            "similarity": _upload_file_to_training_data_container(
                path=similarity_path,
                blob_name=_blob_name(version, similarity_path.name),
            ),
            "embeddings": _upload_file_to_training_data_container(
                path=embeddings_path,
                blob_name=_blob_name(version, embeddings_path.name),
            ),
            "manifest": _blob_uri(
                container=_training_data_container(),
                blob_name=_blob_name(MANIFEST_FILENAME),
            ),
        }

    meta = manifest.get("meta", {})
    meta[version] = {
        "n_rows": n,
        "label_scheme": label_scheme,
        "created_at": datetime.now(tz=UTC).isoformat(),
    }
    if blob_uris:
        meta[version]["blob_uris"] = blob_uris
    versions = versions + [version]
    manifest["versions"] = versions
    manifest["latest"] = version
    manifest["meta"] = meta
    _write_manifest(manifest)
    if training_data_storage_mode() == "blob":
        _upload_file_to_training_data_container(
            path=_manifest_path(),
            blob_name=_blob_name(MANIFEST_FILENAME),
        )
    return version
