"""Model construction helpers for CLI scripts.

Scripts use this module to instantiate domain-specific neural networks from the
shared configuration dataclass without duplicating architecture parameters.
"""

from __future__ import annotations

from typing import Literal

from torch import nn

from models.cbdnet import CBDNet
from models.dncnn import DnCNN
from models.ffdnet import FFDNet
from utils.config import Config

ModelName = Literal["dncnn", "ffdnet", "cbdnet"]


def channels_for_domain(domain: str) -> int:
    """Return model input channels for an experiment domain.

    Args:
        domain: ``srgb`` or ``raw``.

    Returns:
        Number of channels consumed by domain-specific models.

    Raises:
        ValueError: If ``domain`` is unsupported.
    """

    if domain == "srgb":
        return 3
    if domain == "raw":
        return 1
    raise ValueError(f"Unsupported domain: {domain}")


def build_model(model_name: ModelName, domain: str, config: Config) -> nn.Module:
    """Construct a neural denoising model from config.

    Args:
        model_name: ``dncnn``, ``ffdnet``, or ``cbdnet``.
        domain: ``srgb`` or ``raw``.
        config: Experiment configuration.

    Returns:
        Instantiated PyTorch module.

    Raises:
        ValueError: If ``model_name`` is unsupported.
    """

    channels = channels_for_domain(domain)
    if model_name == "dncnn":
        return DnCNN(channels, config.models.dncnn.num_layers, config.models.dncnn.num_channels)
    if model_name == "ffdnet":
        return FFDNet(
            channels,
            config.models.ffdnet.num_layers,
            config.models.ffdnet.num_channels,
            config.models.ffdnet.downscale_factor,
        )
    if model_name == "cbdnet":
        return CBDNet(channels, config.models.cbdnet.cnne_layers, config.models.cbdnet.cnnd_channels)
    raise ValueError(f"Unsupported model: {model_name}")
