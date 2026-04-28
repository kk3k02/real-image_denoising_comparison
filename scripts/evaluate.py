"""Run full denoising evaluation and write CSV results.

The script evaluates BM3D and fine-tuned neural checkpoints on SIDD test splits
and, for RAW, on MIDD test splits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import MIDDDataset, SIDDDataset
from data.splits import load_splits
from evaluation.evaluator import Evaluator, save_summary
from models.bm3d_wrapper import BM3DDenoiser
from utils.config import Config, load_config
from utils.logging_setup import configure_logging
from utils.model_factory import build_model
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

    parser = argparse.ArgumentParser(description="Evaluate denoising methods.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domain", choices=["srgb", "raw", "both"], default="both")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
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


def make_test_loaders(config: Config, domain: str) -> list[tuple[str, DataLoader]]:
    """Create test DataLoaders for a domain.

    Args:
        config: Experiment configuration.
        domain: ``srgb`` or ``raw``.

    Returns:
        List of ``(dataset_name, loader)`` pairs.

    Raises:
        FileNotFoundError: If split JSON is missing.
    """

    splits = load_splits(config.paths.splits_json)
    if domain == "srgb":
        dataset = SIDDDataset(
            config.paths.sidd_medium_root,
            "srgb",
            "test",
            config.data.patch_size,
            scenes=splits["sidd_srgb"]["test"],
        )
        return [("sidd", DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.training.num_workers))]
    sidd_raw = SIDDDataset(
        config.paths.sidd_medium_raw_root,
        "raw",
        "test",
        config.data.patch_size,
        scenes=splits["sidd_raw"]["test"],
    )
    midd_raw = MIDDDataset(
        config.paths.midd_root,
        "raw",
        "test",
        config.data.patch_size,
        scenes=splits["midd_raw"]["test"],
    )
    return [
        ("sidd", DataLoader(sidd_raw, batch_size=1, shuffle=False, num_workers=config.training.num_workers)),
        ("midd", DataLoader(midd_raw, batch_size=1, shuffle=False, num_workers=config.training.num_workers)),
    ]


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a model checkpoint.

    Args:
        model: Model instance.
        checkpoint_path: Checkpoint file path.
        device: Torch device.

    Returns:
        Model with loaded weights.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main() -> None:
    """Run evaluation for selected domains.

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
    evaluator = Evaluator(config.paths.results_dir, device)
    all_results = []
    for domain in selected_domains(args.domain):
        loaders = make_test_loaders(config, domain)
        for dataset_name, loader in loaders:
            all_results.append(
                evaluator.evaluate_bm3d(
                    loader,
                    BM3DDenoiser(config.models.bm3d.sigma_estimation),
                    dataset_name,
                    domain,
                )
            )
            for method in NEURAL_METHODS:
                model = build_model(method, domain, config)
                checkpoint = config.paths.checkpoints_dir / f"{method}_{domain}.pt"
                model = load_checkpoint(model, checkpoint, device)
                all_results.append(evaluator.evaluate_model(loader, model, method, dataset_name, domain))
    save_summary(all_results, config.paths.results_dir / "summary.csv")
    LOGGER.info("Saved summary CSV")


if __name__ == "__main__":
    main()
