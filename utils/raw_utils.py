"""RAW metadata and normalization utilities.

RAW denoising experiments need sensor-specific black/white levels and optional
affine noise model parameters. This module uses ``rawpy`` when files are real
camera RAWs and falls back to sidecar metadata for converted arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_raw_metadata(raw_path: Path, sidecar_path: Path | None = None) -> dict[str, Any]:
    """Read black level, white level, and optional noise parameters.

    Args:
        raw_path: Path to a RAW file readable by ``rawpy`` or a NumPy array file.
        sidecar_path: Optional JSON file containing metadata overrides.

    Returns:
        Dictionary with ``black_level``, ``white_level``, ``beta1``, and ``beta2`` keys
        when those values can be found.

    Raises:
        FileNotFoundError: If ``raw_path`` or an explicitly provided sidecar is missing.
        RuntimeError: If metadata cannot be extracted from either source.
    """

    if not raw_path.exists():
        raise FileNotFoundError(f"RAW file not found: {raw_path}")
    metadata: dict[str, Any] = {}
    inferred_sidecar = raw_path.with_suffix(".json")
    metadata_path = sidecar_path or (inferred_sidecar if inferred_sidecar.exists() else None)
    if metadata_path is not None:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata sidecar not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata.update(json.load(handle))

    if "black_level" not in metadata or "white_level" not in metadata:
        try:
            import rawpy

            with rawpy.imread(str(raw_path)) as raw:
                black = raw.black_level_per_channel
                metadata["black_level"] = float(np.mean(black))
                metadata["white_level"] = float(raw.white_level)
        except Exception as exc:
            if "black_level" not in metadata or "white_level" not in metadata:
                raise RuntimeError(
                    f"Could not extract RAW metadata for {raw_path}. Provide a JSON sidecar."
                ) from exc
    metadata.setdefault("beta1", None)
    metadata.setdefault("beta2", None)
    return metadata


def affine_raw_sigma(signal: np.ndarray, beta1: float, beta2: float) -> np.ndarray:
    """Compute RAW noise sigma from the affine variance model.

    Args:
        signal: Normalized RAW signal in ``[0, 1]``.
        beta1: Shot-noise coefficient.
        beta2: Read-noise variance coefficient.

    Returns:
        Per-pixel sigma map as ``float32``.

    Raises:
        ValueError: If coefficients are negative.
    """

    if beta1 < 0 or beta2 < 0:
        raise ValueError("RAW noise model coefficients must be non-negative")
    variance = beta1 * np.clip(signal, 0.0, 1.0) + beta2
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def normalize_raw_mosaic(raw: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    """Normalize a Bayer mosaic using black and white levels.

    Args:
        raw: RAW mosaic array.
        black_level: Sensor black level.
        white_level: Sensor saturation white level.

    Returns:
        Normalized ``float32`` RAW mosaic clipped to ``[0, 1]``.

    Raises:
        ValueError: If ``white_level`` is not greater than ``black_level``.
    """

    if white_level <= black_level:
        raise ValueError("white_level must be greater than black_level")
    normalized = (raw.astype(np.float32) - float(black_level)) / float(white_level - black_level)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)
