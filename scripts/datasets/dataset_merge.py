#!/usr/bin/env python
"""Merge local LeRobot Dataset v3 datasets.

Datasets can be selected either by scanning one or more parent directories, or
by passing their dataset roots directly.

Examples:
    python scripts/datasets/dataset_merge.py \
        --parent-roots data/batch_a data/batch_b \
        --output-repo-id merged_dataset

    python scripts/datasets/dataset_merge.py \
        --dataset-roots data/dataset_a data/dataset_b \
        --output-repo-id merged_dataset \
        --output-root data/merged_dataset
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from lerobot.datasets import CODEBASE_VERSION, LeRobotDataset, merge_datasets

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.const import DATA_PATH  # noqa: E402


LOGGER = logging.getLogger(__name__)
REQUIRED_DATASET_PATHS = (
    Path("meta/info.json"),
    Path("meta/tasks.parquet"),
    Path("meta/episodes"),
    Path("data"),
)


def is_dataset_root(root: Path) -> bool:
    """Return whether ``root`` has the required LeRobot Dataset v3 layout."""
    return root.is_dir() and all(
        (root / relative_path).exists() for relative_path in REQUIRED_DATASET_PATHS
    )


def find_dataset_roots(parent_roots: list[Path], output_root: Path) -> list[Path]:
    """Find dataset roots among the direct children of all parent roots."""
    dataset_roots: list[Path] = []
    for parent_root in parent_roots:
        parent_root = parent_root.expanduser().resolve()
        if not parent_root.is_dir():
            raise NotADirectoryError(f"Dataset parent directory not found: {parent_root}")

        for child in sorted(parent_root.iterdir()):
            child = child.resolve()
            if child != output_root and is_dataset_root(child):
                dataset_roots.append(child)
    return dataset_roots


def unique_roots(roots: list[Path]) -> list[Path]:
    """Resolve paths and remove duplicates while preserving input order."""
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def validate_dataset_roots(dataset_roots: list[Path], output_root: Path) -> None:
    """Validate source paths before LeRobot loads any datasets."""
    if len(dataset_roots) < 2:
        raise ValueError(f"At least two datasets are required, found {len(dataset_roots)}")

    for root in dataset_roots:
        missing = [str(path) for path in REQUIRED_DATASET_PATHS if not (root / path).exists()]
        if missing:
            raise ValueError(f"Invalid LeRobot Dataset v3 root {root}: missing {', '.join(missing)}")
        if root == output_root:
            raise ValueError(f"A source dataset cannot also be the output dataset: {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--parent-roots", nargs="+", type=Path, help="Parent directories whose direct child datasets will all be merged.")
    sources.add_argument("--dataset-roots", nargs="+", type=Path, help="Dataset root directories to merge.")

    parser.add_argument("--output-repo-id", required=True, help="Repo id/name for the merged dataset.")
    parser.add_argument("--output-root", type=Path, help="Output directory. Defaults to DATA_PATH/output_repo_id.")

    parser.add_argument("--no-concatenate-videos", action="store_true", help="Keep one video per source file instead of packing video shards.")
    parser.add_argument("--no-concatenate-data", action="store_true", help="Keep one parquet file per source file instead of packing data shards.")

    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print inputs without merging.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(f"This script requires LeRobot Dataset v3, found {CODEBASE_VERSION}")

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (DATA_PATH / args.output_repo_id).resolve()
    )
    if args.parent_roots:
        dataset_roots = find_dataset_roots(args.parent_roots, output_root)
    else:
        dataset_roots = args.dataset_roots
    dataset_roots = unique_roots(dataset_roots)
    validate_dataset_roots(dataset_roots, output_root)

    if output_root.is_symlink():
        raise ValueError(f"Output root must not be a symbolic link: {output_root}")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"Output root exists and is not a directory: {output_root}")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")

    LOGGER.info("Loading %d source datasets", len(dataset_roots))
    datasets = [LeRobotDataset(repo_id=root.name, root=root) for root in dataset_roots]
    for dataset in datasets:
        LOGGER.info(
            "Source %s: %s, episodes=%d, frames=%d",
            dataset.repo_id,
            dataset.root,
            dataset.meta.total_episodes,
            dataset.meta.total_frames,
        )
    LOGGER.info("Output: %s", output_root)

    if args.dry_run:
        LOGGER.info("Dry run complete")
        return

    if output_root.exists():
        shutil.rmtree(output_root)

    merged = merge_datasets(
        datasets=datasets,
        output_repo_id=args.output_repo_id,
        output_dir=output_root,
        concatenate_videos=not args.no_concatenate_videos,
        concatenate_data=not args.no_concatenate_data,
    )
    LOGGER.info(
        "Merged dataset complete: %s, episodes=%d, frames=%d",
        merged.root,
        merged.meta.total_episodes,
        merged.meta.total_frames,
    )


if __name__ == "__main__":
    main()
