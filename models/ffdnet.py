"""FFDNet implementation with noise-level map input.

FFDNet downsamples the image with pixel unshuffle, concatenates a downsampled
noise map, processes features through Conv+BN+ReLU blocks, and reconstructs the
full-resolution denoised image with pixel shuffle.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from data.preprocessing import crop_padding, pad_to_factor


class FFDNet(nn.Module):
    """Fast flexible denoising network.

    Args:
        in_channels: Number of image channels for the domain.
        num_layers: Number of convolutional layers before pixel shuffle.
        num_channels: Hidden feature width.
        downscale_factor: Pixel-unshuffle factor.

    Example:
        >>> model = FFDNet(3, 15, 64, 2)
        >>> y = model(torch.zeros(1, 3, 16, 16), torch.zeros(1, 1, 16, 16))
        >>> y.shape
        torch.Size([1, 3, 16, 16])
    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        num_channels: int,
        downscale_factor: int,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("FFDNet requires at least two convolutional layers")
        if downscale_factor <= 1:
            raise ValueError("downscale_factor must be greater than 1")
        self.in_channels = in_channels
        self.downscale_factor = downscale_factor
        shuffled_channels = in_channels * downscale_factor * downscale_factor
        layers: list[nn.Module] = [
            nn.Conv2d(shuffled_channels + 1, num_channels, kernel_size=3, padding=1, bias=True),
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
        layers.append(
            nn.Conv2d(
                num_channels,
                shuffled_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, noise_map: torch.Tensor) -> torch.Tensor:
        """Denoise an input tensor conditioned on a noise map.

        Args:
            x: Noisy tensor with shape ``N x C x H x W``.
            noise_map: Per-pixel noise level map with shape ``N x 1 x H x W`` or scalar-like map.

        Returns:
            Denoised tensor with original spatial dimensions.

        Raises:
            ValueError: If tensor ranks or batch sizes are incompatible.
        """

        if x.ndim != 4 or noise_map.ndim != 4:
            raise ValueError("FFDNet expects 4D input and noise map tensors")
        if x.shape[0] != noise_map.shape[0]:
            raise ValueError("Input and noise map batch sizes must match")
        padded, padding = pad_to_factor(x, self.downscale_factor)
        if noise_map.shape[-2:] != x.shape[-2:]:
            noise_map = F.interpolate(noise_map, size=x.shape[-2:], mode="nearest")
        noise_padded, _ = pad_to_factor(noise_map, self.downscale_factor)
        shuffled = F.pixel_unshuffle(padded, self.downscale_factor)
        noise_down = F.avg_pool2d(noise_padded, kernel_size=self.downscale_factor, stride=self.downscale_factor)
        features = torch.cat([shuffled, noise_down], dim=1)
        restored = F.pixel_shuffle(self.net(features), self.downscale_factor)
        restored = crop_padding(restored, padding)
        return torch.clamp(restored, 0.0, 1.0)
