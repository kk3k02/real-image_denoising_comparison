"""CBDNet implementation for blind real-image denoising.

CBDNet combines CNNE, a five-layer fully convolutional noise estimator, with
CNND, a U-Net denoiser that consumes the noisy image concatenated with the
estimated sigma map.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CNNE(nn.Module):
    """CBDNet noise estimation subnetwork.

    Args:
        in_channels: Number of noisy image channels.
        num_layers: Number of Conv+ReLU layers; the final layer outputs sigma.
        hidden_channels: Hidden channel width.

    Example:
        >>> net = CNNE(3, 5, 32)
        >>> net(torch.zeros(1, 3, 8, 8)).shape
        torch.Size([1, 1, 8, 8])
    """

    def __init__(self, in_channels: int, num_layers: int, hidden_channels: int) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("CNNE requires at least two convolutional layers")
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = hidden_channels
        layers.append(nn.Conv2d(current_channels, 1, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate a non-negative per-pixel sigma map.

        Args:
            x: Noisy tensor with shape ``N x C x H x W``.

        Returns:
            Sigma map with shape ``N x 1 x H x W``.

        Raises:
            ValueError: If ``x`` is not a 4D tensor.
        """

        if x.ndim != 4:
            raise ValueError("CNNE expects an N x C x H x W tensor")
        return F.softplus(self.net(x))


class ConvBlock(nn.Module):
    """Two-layer convolutional block used in CNND.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.

    Example:
        >>> block = ConvBlock(4, 8)
        >>> block(torch.zeros(1, 4, 8, 8)).shape
        torch.Size([1, 8, 8, 8])
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two convolutions with ReLU activations.

        Args:
            x: Input feature tensor.

        Returns:
            Output feature tensor.

        Raises:
            ValueError: If ``x`` is not a 4D tensor.
        """

        if x.ndim != 4:
            raise ValueError("ConvBlock expects a 4D tensor")
        return self.net(x)


class CNND(nn.Module):
    """CBDNet denoising U-Net with skip connections.

    Args:
        in_channels: Image channels plus one sigma channel.
        out_channels: Number of denoised image channels.
        base_channels: Base U-Net width.

    Example:
        >>> net = CNND(4, 3, 32)
        >>> net(torch.zeros(1, 4, 32, 32)).shape
        torch.Size([1, 3, 32, 32])
    """

    def __init__(self, in_channels: int, out_channels: int, base_channels: int) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise concatenated noisy image and sigma map features.

        Args:
            x: Tensor with shape ``N x (C+1) x H x W``.

        Returns:
            Denoised tensor with shape ``N x C x H x W``.

        Raises:
            ValueError: If ``x`` is not 4D.
        """

        if x.ndim != 4:
            raise ValueError("CNND expects a 4D tensor")
        original_size = x.shape[-2:]
        pad_h = (8 - original_size[0] % 8) % 8
        pad_w = (8 - original_size[1] % 8) % 8
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else x
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        d3 = self.up3(bottleneck)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.out(d1)
        if pad_h or pad_w:
            out = out[..., : original_size[0], : original_size[1]]
        return torch.clamp(out, 0.0, 1.0)


class CBDNet(nn.Module):
    """CBDNet model combining CNNE and CNND.

    Args:
        in_channels: Number of image channels for the domain.
        cnne_layers: Number of CNNE convolutional layers.
        cnnd_channels: Base channel count for CNND.

    Example:
        >>> model = CBDNet(3, 5, 32)
        >>> clean, sigma = model(torch.zeros(1, 3, 16, 16))
        >>> clean.shape, sigma.shape
        (torch.Size([1, 3, 16, 16]), torch.Size([1, 1, 16, 16]))
    """

    def __init__(self, in_channels: int, cnne_layers: int, cnnd_channels: int) -> None:
        super().__init__()
        self.cnne = CNNE(in_channels, cnne_layers, cnnd_channels)
        self.cnnd = CNND(in_channels + 1, in_channels, cnnd_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate noise and denoise an input image.

        Args:
            x: Noisy tensor with shape ``N x C x H x W``.

        Returns:
            Tuple of denoised tensor and estimated sigma map.

        Raises:
            ValueError: If ``x`` is not a 4D tensor.
        """

        if x.ndim != 4:
            raise ValueError("CBDNet expects an N x C x H x W tensor")
        sigma = self.cnne(x)
        denoised = self.cnnd(torch.cat([x, sigma], dim=1))
        return denoised, sigma
