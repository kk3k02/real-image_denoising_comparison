"""PSNR and SSIM metrics for CPU and GPU evaluation.

CPU metrics use scikit-image for conventional full-reference reporting. GPU
metrics use piqa modules so large evaluation runs can remain on CUDA.
"""

from __future__ import annotations

import numpy as np
import torch


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor image to NumPy layout expected by scikit-image.

    Args:
        tensor: Image tensor with shape ``C x H x W`` or ``1 x C x H x W``.

    Returns:
        NumPy image with ``H x W x C`` or ``H x W`` layout.

    Raises:
        ValueError: If tensor rank is unsupported.
    """

    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Only batch size 1 can be converted to a single NumPy image")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError("Expected C x H x W tensor")
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    if array.shape[0] == 1:
        return array[0]
    return np.transpose(array, (1, 2, 0))


def psnr_cpu(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Compute CPU PSNR using scikit-image.

    Args:
        prediction: Predicted image tensor.
        target: Clean target tensor.

    Returns:
        PSNR value in decibels.

    Raises:
        ValueError: If converted images have different shapes.
    """

    from skimage.metrics import peak_signal_noise_ratio

    pred_np = tensor_to_numpy_image(prediction)
    target_np = tensor_to_numpy_image(target)
    if pred_np.shape != target_np.shape:
        raise ValueError(f"PSNR shape mismatch: {pred_np.shape} vs {target_np.shape}")
    return float(peak_signal_noise_ratio(target_np, pred_np, data_range=1.0))


def ssim_cpu(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Compute CPU SSIM using scikit-image.

    Args:
        prediction: Predicted image tensor.
        target: Clean target tensor.

    Returns:
        SSIM value.

    Raises:
        ValueError: If converted images have different shapes.
    """

    from skimage.metrics import structural_similarity

    pred_np = tensor_to_numpy_image(prediction)
    target_np = tensor_to_numpy_image(target)
    if pred_np.shape != target_np.shape:
        raise ValueError(f"SSIM shape mismatch: {pred_np.shape} vs {target_np.shape}")
    channel_axis = -1 if pred_np.ndim == 3 else None
    return float(structural_similarity(target_np, pred_np, data_range=1.0, channel_axis=channel_axis))


def psnr_gpu(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute GPU PSNR using piqa.

    Args:
        prediction: Predicted batch tensor in ``[0, 1]``.
        target: Clean target batch tensor.

    Returns:
        Scalar PSNR tensor.

    Raises:
        ImportError: If piqa is unavailable.
    """

    from piqa import PSNR

    metric = PSNR().to(prediction.device)
    return metric(prediction.clamp(0.0, 1.0), target.clamp(0.0, 1.0))


def ssim_gpu(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute GPU SSIM using piqa.

    Args:
        prediction: Predicted batch tensor in ``[0, 1]``.
        target: Clean target batch tensor.

    Returns:
        Scalar SSIM tensor.

    Raises:
        ImportError: If piqa is unavailable.
    """

    from piqa import SSIM

    metric = SSIM().to(prediction.device)
    return metric(prediction.clamp(0.0, 1.0), target.clamp(0.0, 1.0))
