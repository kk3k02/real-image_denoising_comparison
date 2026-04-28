"""Reproducibility helpers for deterministic training and evaluation.

The pipeline uses this module at script startup so dataset splits, patch
sampling, model initialization, and CUDA kernels behave consistently across
Colab and Windows where deterministic implementations are available.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_reproducibility(seed: int) -> None:
    """Set random seeds and deterministic PyTorch behavior.

    Args:
        seed: Seed applied to Python, NumPy, PyTorch, and CUDA RNGs.

    Returns:
        None.

    Raises:
        ValueError: If ``seed`` is negative.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
