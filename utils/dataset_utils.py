"""Small helpers for validating local LeRobot dataset paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    """Load ``meta/info.json`` from a local dataset root."""

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_local_dataset_root_and_repo_id(
    dataset_root: Path,
    repo_id: str,
    *,
    require_data_dir: bool = True,
) -> Path:
    """Validate that a local dataset root exists and matches the repo id.

    If ``meta/info.json`` contains ``repo_id``, that value is authoritative.
    Otherwise the final path component of ``repo_id`` must match the dataset
    root directory name.
    """

    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {resolved_root}")
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {resolved_root}")

    if require_data_dir and not (resolved_root / "data").is_dir():
        raise FileNotFoundError(f"Dataset data directory not found: {resolved_root / 'data'}")

    info = load_dataset_info(resolved_root)
    meta_repo_id = info.get("repo_id")
    expected_repo_id = str(meta_repo_id) if meta_repo_id else resolved_root.name
    actual_repo_id = str(repo_id)
    actual_repo_name = actual_repo_id.split("/")[-1]
    expected_repo_name = expected_repo_id.split("/")[-1]

    if meta_repo_id is not None and actual_repo_id != expected_repo_id:
        raise ValueError(
            f"repo_id does not match dataset meta: repo_id={actual_repo_id!r}, "
            f"meta repo_id={expected_repo_id!r}"
        )
    if meta_repo_id is None and actual_repo_name != expected_repo_name:
        raise ValueError(
            f"repo_id does not match dataset root name: repo_id={actual_repo_id!r}, "
            f"dataset root name={expected_repo_name!r}"
        )

    return resolved_root
