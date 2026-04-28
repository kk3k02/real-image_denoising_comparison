"""Inference timing utilities for CPU and GPU benchmarks.

The protocol follows the thesis requirements: CUDA warmup before timed runs
with synchronization around each measurement, and single-core CPU timing for
deployment-oriented comparisons.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch

from utils.config import BenchmarkingConfig


def time_torch_inference(
    forward_fn: Callable[[torch.Tensor], torch.Tensor | tuple[torch.Tensor, ...]],
    input_tensor: torch.Tensor,
    device: torch.device,
    config: BenchmarkingConfig,
) -> tuple[float, float]:
    """Measure wall-clock inference time for a PyTorch forward callable.

    Args:
        forward_fn: Callable accepting one input tensor.
        input_tensor: Input tensor before device transfer.
        device: Requested torch device.
        config: Benchmarking protocol settings.

    Returns:
        Mean and standard deviation in milliseconds.

    Raises:
        ValueError: If CUDA timing is requested without CUDA availability.
    """

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is False")
    x = input_tensor.to(device)
    timings: list[float] = []
    with torch.no_grad():
        if device.type == "cuda":
            for _ in range(config.gpu_warmup_runs):
                forward_fn(x)
            torch.cuda.synchronize()
            runs = config.gpu_timed_runs
            for _ in range(runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                forward_fn(x)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)
        else:
            old_threads = torch.get_num_threads()
            torch.set_num_threads(config.cpu_num_threads)
            try:
                for _ in range(config.cpu_timed_runs):
                    start = time.perf_counter()
                    forward_fn(x)
                    timings.append((time.perf_counter() - start) * 1000.0)
            finally:
                torch.set_num_threads(old_threads)
    return float(statistics.mean(timings)), float(statistics.stdev(timings) if len(timings) > 1 else 0.0)


def time_callable(callable_fn: Callable[[], object], runs: int) -> tuple[float, float]:
    """Measure a generic CPU callable.

    Args:
        callable_fn: Zero-argument callable to execute.
        runs: Number of timed executions.

    Returns:
        Mean and standard deviation in milliseconds.

    Raises:
        ValueError: If ``runs`` is not positive.
    """

    if runs <= 0:
        raise ValueError("runs must be positive")
    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        callable_fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.mean(timings)), float(statistics.stdev(timings) if len(timings) > 1 else 0.0)
