"""Logging and optional Weights & Biases initialization.

All scripts route runtime messages through the standard logging package. This
module keeps logger configuration and optional W&B startup in one place so the
YAML config controls runtime verbosity and experiment tracking.
"""

from __future__ import annotations

import logging
from typing import Any

from utils.config import Config


def configure_logging(level: str) -> None:
    """Configure process-wide Python logging.

    Args:
        level: Logging level name such as ``INFO`` or ``DEBUG``.

    Returns:
        None.

    Raises:
        ValueError: If the logging level name is unknown.
    """

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_wandb(config: Config, run_name: str | None = None) -> Any | None:
    """Initialize a W&B run when enabled in config.

    Args:
        config: Parsed experiment configuration.
        run_name: Optional human-readable run name.

    Returns:
        W&B run object when enabled, otherwise ``None``.

    Raises:
        ImportError: If W&B is enabled but the package is unavailable.
    """

    if not config.logging.wandb_enabled:
        return None
    import wandb

    return wandb.init(
        project=config.logging.wandb_project,
        name=run_name,
        config=config,
    )
