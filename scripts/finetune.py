"""Fine-tune DnCNN, FFDNet, and CBDNet on SIDD Medium splits.

The script keeps sRGB and RAW domains separate by constructing one dataset and
one model per domain/method combination.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import SIDDDataset
from data.splits import load_splits
from training.trainer import fit
from utils.config import Config, load_config
from utils.logging_setup import configure_logging, init_wandb
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

    parser = argparse.ArgumentParser(description="Fine-tune neural denoising models.")
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


def make_sidd_loaders(config: Config, domain: str) -> tuple[DataLoader, DataLoader]:
    """Create SIDD train and validation DataLoaders.

    Args:
        config: Experiment configuration.
        domain: ``srgb`` or ``raw``.

    Returns:
        Training and validation loaders.

    Raises:
        FileNotFoundError: If split JSON is missing.
    """

    splits = load_splits(config.paths.splits_json)
    key = f"sidd_{domain}"
    root = config.paths.sidd_medium_root if domain == "srgb" else config.paths.sidd_medium_raw_root
    train_dataset = SIDDDataset(
        root,
        domain,
        "train",
        config.data.patch_size,
        scenes=splits[key]["train"],
        augment=True,
    )
    val_dataset = SIDDDataset(
        root,
        domain,
        "val",
        config.data.patch_size,
        scenes=splits[key]["val"],
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def main() -> None:
    """Run fine-tuning for all selected neural models.

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
    for domain in selected_domains(args.domain):
        train_loader, val_loader = make_sidd_loaders(config, domain)
        for method in NEURAL_METHODS:
            model = build_model(method, domain, config)
            wandb_run = init_wandb(config, run_name=f"{method}-{domain}")
            checkpoint = config.paths.checkpoints_dir / f"{method}_{domain}.pt"
            LOGGER.info("Fine-tuning %s on SIDD %s", method, domain)
            fit(model, method, train_loader, val_loader, config, device, checkpoint, wandb_run)
            if wandb_run is not None:
                wandb_run.finish()


if __name__ == "__main__":
    main()
