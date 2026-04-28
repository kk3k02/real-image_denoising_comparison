"""Image preprocessing for sRGB and RAW denoising experiments.

This module provides deterministic normalization, non-overlapping patch
extraction, training-only augmentation, and RAW Bayer handling used by dataset
classes and scripts.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.raw_utils import normalize_raw_mosaic, read_raw_metadata


def load_srgb_image(path: Path) -> np.ndarray:
    """Load an sRGB image as normalized RGB ``float32``.

    Args:
        path: Path to a PNG/JPEG/TIFF image.

    Returns:
        RGB image with shape ``H x W x 3`` and values in ``[0, 1]``.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If OpenCV cannot decode the file.
    """

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    max_value = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
    return np.clip(image.astype(np.float32) / float(max_value), 0.0, 1.0)


def load_raw_mosaic(path: Path) -> np.ndarray:
    """Load and normalize a RAW Bayer mosaic.

    Args:
        path: Path to a RAW camera file, ``.npy`` array, or image-like mosaic.

    Returns:
        Normalized RAW mosaic with shape ``H x W`` and values in ``[0, 1]``.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file cannot be read as RAW, NumPy, or an image.
    """

    if not path.exists():
        raise FileNotFoundError(f"RAW file not found: {path}")
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
        metadata = read_raw_metadata(path)
        return normalize_raw_mosaic(raw, metadata["black_level"], metadata["white_level"])
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw_handle:
            raw = raw_handle.raw_image_visible.copy()
            metadata = read_raw_metadata(path)
            return normalize_raw_mosaic(raw, metadata["black_level"], metadata["white_level"])
    except Exception:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not load RAW mosaic: {path}")
        if image.ndim == 3:
            image = image[..., 0]
        max_value = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
        return np.clip(image.astype(np.float32) / float(max_value), 0.0, 1.0)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an image array to a channel-first tensor.

    Args:
        image: ``H x W`` RAW mosaic or ``H x W x C`` sRGB array.

    Returns:
        ``C x H x W`` tensor.

    Raises:
        ValueError: If the input rank is unsupported.
    """

    if image.ndim == 2:
        return torch.from_numpy(image[None, ...].astype(np.float32))
    if image.ndim == 3:
        return torch.from_numpy(np.transpose(image, (2, 0, 1)).astype(np.float32))
    raise ValueError(f"Unsupported image rank: {image.ndim}")


def extract_non_overlapping_patches(image: np.ndarray, patch_size: int) -> list[np.ndarray]:
    """Extract non-overlapping square patches from an image.

    Args:
        image: Input image array.
        patch_size: Patch side length.

    Returns:
        List of patches in row-major order.

    Raises:
        ValueError: If ``patch_size`` is invalid or larger than the image.
    """

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    height, width = image.shape[:2]
    if height < patch_size or width < patch_size:
        raise ValueError(f"Image {height}x{width} is smaller than patch size {patch_size}")
    patches: list[np.ndarray] = []
    for top in range(0, height - patch_size + 1, patch_size):
        for left in range(0, width - patch_size + 1, patch_size):
            patches.append(image[top : top + patch_size, left : left + patch_size].copy())
    return patches


def augment_pair(noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply paired random flips and right-angle rotations for training.

    Args:
        noisy: Noisy image or patch.
        clean: Aligned clean target.

    Returns:
        Augmented noisy and clean arrays.

    Raises:
        ValueError: If input shapes differ.
    """

    if noisy.shape != clean.shape:
        raise ValueError(f"Paired augmentation requires matching shapes, got {noisy.shape} and {clean.shape}")
    if random.random() < 0.5:
        noisy = np.flip(noisy, axis=1)
        clean = np.flip(clean, axis=1)
    if random.random() < 0.5:
        noisy = np.flip(noisy, axis=0)
        clean = np.flip(clean, axis=0)
    rotations = random.randint(0, 3)
    if rotations:
        noisy = np.rot90(noisy, k=rotations, axes=(0, 1))
        clean = np.rot90(clean, k=rotations, axes=(0, 1))
    return np.ascontiguousarray(noisy), np.ascontiguousarray(clean)


def pad_to_factor(tensor: torch.Tensor, factor: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad a tensor spatially so height and width are divisible by ``factor``.

    Args:
        tensor: Tensor with shape ``C x H x W`` or ``N x C x H x W``.
        factor: Required divisibility factor.

    Returns:
        Padded tensor and ``(pad_h, pad_w)`` values.

    Raises:
        ValueError: If tensor rank or factor is invalid.
    """

    if factor <= 0:
        raise ValueError("factor must be positive")
    if tensor.ndim not in {3, 4}:
        raise ValueError("tensor must have rank 3 or 4")
    height, width = tensor.shape[-2:]
    pad_h = (factor - height % factor) % factor
    pad_w = (factor - width % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0)
    padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (pad_h, pad_w)


def crop_padding(tensor: torch.Tensor, padding: tuple[int, int]) -> torch.Tensor:
    """Remove spatial padding returned by ``pad_to_factor``.

    Args:
        tensor: Padded tensor.
        padding: ``(pad_h, pad_w)`` tuple.

    Returns:
        Cropped tensor.

    Raises:
        ValueError: If padding contains negative values.
    """

    pad_h, pad_w = padding
    if pad_h < 0 or pad_w < 0:
        raise ValueError("padding values must be non-negative")
    end_h = None if pad_h == 0 else -pad_h
    end_w = None if pad_w == 0 else -pad_w
    return tensor[..., :end_h, :end_w]
