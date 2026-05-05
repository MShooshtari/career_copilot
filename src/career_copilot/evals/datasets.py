from __future__ import annotations

import json
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import Any

EXAMPLES_PACKAGE = "career_copilot.evals.examples"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL eval dataset."""
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Eval case at {path}:{line_number} must be an object")
            cases.append(value)
    return cases


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL rows with stable key ordering for reviewable fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def example_dataset_path(name: str) -> Path:
    """Return a versioned package eval dataset path."""
    if not name.endswith(".jsonl"):
        name = f"{name}.jsonl"
    return Path(str(resources.files(EXAMPLES_PACKAGE).joinpath(name)))


def load_dataset(name_or_path: str) -> list[dict[str, Any]]:
    """
    Load an eval dataset by package example name or filesystem path.

    Larger private datasets can live under data/evals locally; committed smoke
    datasets live in career_copilot.evals.examples.
    """
    path = Path(name_or_path)
    if path.exists():
        return read_jsonl(path)
    return read_jsonl(example_dataset_path(name_or_path))

