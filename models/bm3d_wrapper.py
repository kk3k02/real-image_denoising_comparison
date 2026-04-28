"""BM3D wrapper with domain-specific noise estimation.

This module adapts the external ``bm3d`` package to the project evaluator. sRGB
noise is estimated with a Haar HH-band MAD estimator, while RAW noise can be
supplied from affine metadata-derived sigma maps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.raw_utils import affine_raw_sigma, read_raw_metadata


class BM3DDenoiser:
    """Classical BM3D denoiser with sRGB and RAW sigma estimation.

    Args:
        sigma_estimation: ``mad`` for sRGB or ``metadata`` for RAW.

    Example:
        >>> denoiser = BM3DDenoiser("mad")
        >>> denoiser.sigma_estimation
        'mad'
    """

    def __init__(self, sigma_estimation: str = "mad") -> None:
        self.sigma_estimation = sigma_estimation

    def estimate_sigma_mad(self, image: np.ndarray) -> float:
        """Estimate noise sigma from the Haar HH subband using MAD.

        Args:
            image: Normalized image array in ``[0, 1]``.

        Returns:
            Scalar noise standard deviation.

        Raises:
            ValueError: If the image is smaller than one Haar block.
        """

        if image.shape[0] < 2 or image.shape[1] < 2:
            raise ValueError("MAD sigma estimation requires at least a 2x2 image")
        gray = image.mean(axis=2) if image.ndim == 3 else image
        even_h = gray.shape[0] - gray.shape[0] % 2
        even_w = gray.shape[1] - gray.shape[1] % 2
        block = gray[:even_h, :even_w].astype(np.float32)
        hh = (
            block[0::2, 0::2]
            - block[0::2, 1::2]
            - block[1::2, 0::2]
            + block[1::2, 1::2]
        ) / 2.0
        return float(np.median(np.abs(hh)) / 0.6745)

    def estimate_sigma_raw(self, raw: np.ndarray, raw_path: Path) -> float:
        """Estimate scalar RAW sigma from affine noise metadata.

        Args:
            raw: Normalized RAW mosaic.
            raw_path: RAW file path used to find metadata or sidecar noise parameters.

        Returns:
            Mean sigma over the normalized mosaic.

        Raises:
            RuntimeError: If beta coefficients are unavailable.
        """

        metadata = read_raw_metadata(raw_path)
        beta1 = metadata.get("beta1")
        beta2 = metadata.get("beta2")
        if beta1 is None or beta2 is None:
            raise RuntimeError(f"Missing beta1/beta2 RAW noise parameters for {raw_path}")
        sigma_map = affine_raw_sigma(raw, float(beta1), float(beta2))
        return float(sigma_map.mean())

    def denoise(self, noisy: np.ndarray, domain: str, raw_path: Path | None = None) -> np.ndarray:
        """Run BM3D denoising on a normalized image.

        Args:
            noisy: Normalized noisy image.
            domain: ``srgb`` or ``raw``.
            raw_path: RAW file path required when ``domain`` is ``raw`` and metadata sigma is used.

        Returns:
            Denoised image clipped to ``[0, 1]``.

        Raises:
            ValueError: If ``domain`` is unsupported or RAW metadata is missing.
            ImportError: If the ``bm3d`` package is not installed.
        """

        try:
            import bm3d
        except ImportError as exc:
            raise ImportError("Install bm3d==4.0.0 to use BM3D denoising") from exc

        if domain == "srgb":
            sigma = self.estimate_sigma_mad(noisy)
        elif domain == "raw":
            if raw_path is None:
                raise ValueError("raw_path is required for RAW BM3D sigma estimation")
            sigma = self.estimate_sigma_raw(noisy, raw_path)
        else:
            raise ValueError(f"Unsupported domain: {domain}")
        denoised = bm3d.bm3d(noisy.astype(np.float32), sigma_psd=sigma)
        return np.clip(denoised.astype(np.float32), 0.0, 1.0)
