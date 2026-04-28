"""Deterministic scene-level dataset splitting.

SIDD and MIDD contain multiple images per scene or sensor condition. This
module derives stable train/validation/test scene lists by sorting before
randomization and seeding Python's RNG with the configured seed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def scene_id_from_path(path: Path) -> str:
    """Infer a scene identifier from a dataset path.

    Args:
        path: Image path belonging to a scene folder or named scene file.

    Returns:
        Stable scene identifier string.

    Raises:
        ValueError: If the path has no usable stem.
    """

    if not path.stem:
        raise ValueError(f"Cannot infer scene id from path: {path}")
    parent = path.parent.name
    return parent if parent else path.stem


def split_scenes(
    scene_ids: list[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[str]]:
    """Split scene identifiers into deterministic train/val/test lists.

    Args:
        scene_ids: Scene identifiers.
        ratios: Train, validation, and test fractions.
        seed: RNG seed used after sorting.

    Returns:
        Dictionary with ``train``, ``val``, and ``test`` lists.

    Raises:
        ValueError: If ratios are invalid or no scenes are provided.
    """

    if not scene_ids:
        raise ValueError("Cannot split an empty scene list")
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must contain three values summing to 1.0")
    ordered = sorted(set(scene_ids))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    total = len(ordered)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    return {
        "train": sorted(ordered[:train_end]),
        "val": sorted(ordered[train_end:val_end]),
        "test": sorted(ordered[val_end:]),
    }


def save_splits(splits: dict[str, dict[str, list[str]]], path: Path) -> None:
    """Save dataset split metadata as JSON.

    Args:
        splits: Nested split dictionary keyed by dataset/domain.
        path: Output JSON path.

    Returns:
        None.

    Raises:
        OSError: If the output path cannot be written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2, sort_keys=True)


def load_splits(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load dataset split metadata from JSON.

    Args:
        path: Split JSON path.

    Returns:
        Nested split dictionary.

    Raises:
        FileNotFoundError: If the split file does not exist.
        ValueError: If the JSON is malformed.
    """

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Split JSON must contain an object at the top level")
    return data
