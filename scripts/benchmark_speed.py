"""Benchmark CPU and GPU inference speed for denoising methods.

The script uses synthetic tensors with configured spatial size so timing can be
run independently from dataset availability on Colab and Windows.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarking.timer import time_torch_inference
from training.trainer import forward_model
from utils.config import load_config
from utils.logging_setup import configure_logging
from utils.model_factory import build_model, channels_for_domain
from utils.reproducibility import set_reproducibility

LOGGER = logging.getLogger(__name__)
NEURAL_METHODS = ("dncnn", "ffdnet", "cbdnet")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None.

    Returns:
        Parsed argparse namespace.

    Raises:
        SystemExit: If arguments are invalid.
    """

    parser = argparse.ArgumentParser(description="Benchmark denoising inference speed.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", choices=["srgb", "raw", "both"], default="both")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def selected_domains(domain: str) -> list[str]:
    """Expand a domain selector.

    Args:
        domain: ``srgb``, ``raw``, or ``both``.

    Returns:
        List of concrete domains.

    Raises:
        ValueError: If the selector is unsupported.
    """

    if domain == "both":
        return ["srgb", "raw"]
    if domain in {"srgb", "raw"}:
        return [domain]
    raise ValueError(f"Unsupported domain selector: {domain}")


def main() -> None:
    """Run synthetic neural inference benchmark.

    Args:
        None.

    Returns:
        None.

    Raises:
        ValueError: If CUDA is requested but unavailable.
    """

    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.logging.level)
    set_reproducibility(config.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    rows = []
    height, width = config.benchmarking.test_image_size
    for domain in selected_domains(args.domain):
        channels = channels_for_domain(domain)
        x = torch.rand(1, channels, height, width)
        for method in NEURAL_METHODS:
            model = build_model(method, domain, config).to(device).eval()

            def forward(input_tensor: torch.Tensor, model: torch.nn.Module = model, method: str = method):
                prediction, _ = forward_model(model, method, input_tensor, None)
                return prediction

            mean_ms, std_ms = time_torch_inference(forward, x, device, config.benchmarking)
            rows.append({"method": method, "domain": domain, "device": device.type, "mean_ms": mean_ms, "std_ms": std_ms})
            LOGGER.info("%s %s %s: %.3f +- %.3f ms", method, domain, device.type, mean_ms, std_ms)
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    output = config.paths.results_dir / f"benchmark_{device.type}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    LOGGER.info("Saved %s", output)


if __name__ == "__main__":
    main()
