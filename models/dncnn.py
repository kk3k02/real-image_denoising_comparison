"""DnCNN residual denoiser implementation.

The model follows the blind DnCNN layout: first Conv+ReLU, middle Conv+BN+ReLU
blocks, and a final Conv that predicts noise residuals subtracted from the
input image.
"""

from __future__ import annotations

import torch
from torch import nn


class DnCNN(nn.Module):
    """Blind DnCNN denoising network.

    Args:
        in_channels: Number of image channels for the selected domain.
        num_layers: Total convolutional layers; 17 gives 15 middle blocks.
        num_channels: Hidden feature width.

    Example:
        >>> model = DnCNN(in_channels=3, num_layers=17, num_channels=64)
        >>> y = model(torch.zeros(1, 3, 16, 16))
        >>> y.shape
        torch.Size([1, 3, 16, 16])
    """

    def __init__(self, in_channels: int, num_layers: int, num_channels: int) -> None:
        super().__init__()
        if num_layers < 3:
            raise ValueError("DnCNN requires at least three convolutional layers")
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(num_channels, in_channels, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise an input tensor by residual subtraction.

        Args:
            x: Noisy tensor with shape ``N x C x H x W``.

        Returns:
            Denoised tensor with the same shape.

        Raises:
            ValueError: If ``x`` is not a 4D tensor.
        """

        if x.ndim != 4:
            raise ValueError("DnCNN expects an N x C x H x W tensor")
        residual = self.net(x)
        return torch.clamp(x - residual, 0.0, 1.0)
