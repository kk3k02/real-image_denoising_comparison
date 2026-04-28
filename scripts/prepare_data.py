"""Prepare dataset split metadata and verify SIDD/MIDD integrity.

The script checks that expected paired files are present before training or
evaluation and writes deterministic scene-level splits to JSON. Dataset archives
must be placed under the paths configured in ``experiment_config.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import _discover_pairs
from data.splits import save_splits, split_scenes
from utils.config import Config, load_config
from utils.logging_setup import configure_logging
from utils.reproducibility import set_reproducibility

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None.

    Returns:
        Parsed argparse namespace.

    Raises:
        SystemExit: If arguments are invalid.
    """

    parser = argparse.ArgumentParser(description="Verify denoising datasets and build split JSON.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", choices=["srgb", "raw", "both"], default="both")
    return parser.parse_args()


def verify_dataset(root: Path, expected_pairs: int, label: str) -> list[str]:
    """Verify a paired dataset root and return scene ids.

    Args:
        root: Dataset root path.
        expected_pairs: Required number of noisy/clean pairs.
        label: Human-readable dataset label.

    Returns:
        Scene identifiers discovered from pair folders.

    Raises:
        FileNotFoundError: If the root does not exist.
        RuntimeError: If pair count differs from expectation.
    """

    pairs = _discover_pairs(root)
    if len(pairs) != expected_pairs:
        raise RuntimeError(
            f"{label}: expected {expected_pairs} noisy/clean pairs under {root}, found {len(pairs)}. "
            "Check dataset extraction and filename conventions."
        )
    LOGGER.info("%s integrity OK: %s pairs", label, len(pairs))
    return [pair.scene_id for pair in pairs]


def build_splits(config: Config, domain: str) -> dict[str, dict[str, list[str]]]:
    """Build deterministic split dictionary for selected domains.

    Args:
        config: Experiment configuration.
        domain: ``srgb``, ``raw``, or ``both``.

    Returns:
        Nested split dictionary.

    Raises:
        RuntimeError: If dataset verification fails.
    """

    splits: dict[str, dict[str, list[str]]] = {}
    if domain in {"srgb", "both"}:
        scenes = verify_dataset(config.paths.sidd_medium_root, config.data.expected_sidd_pairs, "SIDD Medium sRGB")
        splits["sidd_srgb"] = split_scenes(scenes, config.data.train_val_test_ratio, config.seed)
    if domain in {"raw", "both"}:
        sidd_raw_scenes = verify_dataset(
            config.paths.sidd_medium_raw_root,
            config.data.expected_sidd_pairs,
            "SIDD Medium RAW",
        )
        midd_scenes = verify_dataset(config.paths.midd_root, config.data.expected_midd_pairs, "MIDD RAW")
        splits["sidd_raw"] = split_scenes(sidd_raw_scenes, config.data.train_val_test_ratio, config.seed)
        splits["midd_raw"] = split_scenes(midd_scenes, config.data.train_val_test_ratio, config.seed)
    return splits


def main() -> None:
    """Run dataset verification and split generation.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If verification fails.
    """

    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.logging.level)
    set_reproducibility(config.seed)
    try:
        splits = build_splits(config, args.domain)
        save_splits(splits, config.paths.splits_json)
        LOGGER.info("Saved splits to %s", config.paths.splits_json)
    except Exception as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
