"""Loss functions for neural denoising fine-tuning.

DnCNN and FFDNet use image-domain MSE. CBDNet additionally applies the
asymmetric CNNE loss so underestimating real noise is penalized more strongly.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def mse_denoising_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE denoising loss.

    Args:
        prediction: Denoised prediction tensor.
        target: Clean target tensor.

    Returns:
        Scalar MSE loss.

    Raises:
        ValueError: If tensor shapes differ.
    """

    if prediction.shape != target.shape:
        raise ValueError(f"MSE loss shape mismatch: {prediction.shape} vs {target.shape}")
    return F.mse_loss(prediction, target)


def asymmetric_loss(sigma_pred: torch.Tensor, sigma_gt: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    """Compute CBDNet asymmetric noise-estimation loss.

    Args:
        sigma_pred: Predicted sigma map.
        sigma_gt: Target sigma map.
        alpha: Weight applied to overestimation; underestimation receives ``1 - alpha``.

    Returns:
        Scalar asymmetric loss.

    Raises:
        ValueError: If shapes differ or ``alpha`` is outside ``[0, 1]``.
    """

    if sigma_pred.shape != sigma_gt.shape:
        raise ValueError(f"Asymmetric loss shape mismatch: {sigma_pred.shape} vs {sigma_gt.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    diff = sigma_pred - sigma_gt
    weight = torch.where(diff < 0, 1 - alpha, alpha)
    return (weight * diff**2).mean()


def residual_sigma_target(noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Build a per-pixel sigma proxy from paired noisy and clean tensors.

    Args:
        noisy: Noisy input tensor.
        clean: Clean target tensor.

    Returns:
        Single-channel absolute residual map averaged over color channels.

    Raises:
        ValueError: If input shapes differ.
    """

    if noisy.shape != clean.shape:
        raise ValueError(f"Sigma target shape mismatch: {noisy.shape} vs {clean.shape}")
    return torch.mean(torch.abs(noisy - clean), dim=1, keepdim=True)
