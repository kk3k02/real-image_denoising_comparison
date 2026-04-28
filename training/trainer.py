"""Fine-tuning loop for neural denoising models.

The trainer handles optimizer setup, scheduler stepping, early stopping,
checkpointing, and optional W&B logging while preserving model-specific forward
contracts for DnCNN, FFDNet, and CBDNet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import asymmetric_loss, mse_denoising_loss, residual_sigma_target
from utils.config import Config

LOGGER = logging.getLogger(__name__)
ModelName = Literal["dncnn", "ffdnet", "cbdnet"]


class EarlyStopping:
    """Track validation loss and stop after patience is exceeded.

    Args:
        patience: Number of unimproved epochs to tolerate.

    Example:
        >>> stopper = EarlyStopping(2)
        >>> stopper.update(1.0)
        True
    """

    def __init__(self, patience: int) -> None:
        if patience <= 0:
            raise ValueError("patience must be positive")
        self.patience = patience
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def update(self, val_loss: float) -> bool:
        """Update state with a validation loss.

        Args:
            val_loss: Current validation loss.

        Returns:
            ``True`` when the loss improved.

        Raises:
            ValueError: If ``val_loss`` is negative.
        """

        if val_loss < 0:
            raise ValueError("val_loss must be non-negative")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    def should_stop(self) -> bool:
        """Report whether training should stop.

        Args:
            None.

        Returns:
            ``True`` when patience has been exceeded.

        Raises:
            None.
        """

        return self.bad_epochs >= self.patience


def _noise_map_from_pair(noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Create a batch-level noise map for FFDNet conditioning.

    Args:
        noisy: Noisy batch.
        clean: Clean batch.

    Returns:
        Single-channel sigma map broadcast over each image.

    Raises:
        ValueError: If shapes differ.
    """

    if noisy.shape != clean.shape:
        raise ValueError("Noisy and clean tensors must have the same shape")
    sigma = torch.sqrt(torch.mean((noisy - clean) ** 2, dim=(1, 2, 3), keepdim=True))
    return sigma.expand(noisy.shape[0], 1, noisy.shape[-2], noisy.shape[-1])


def forward_model(
    model: nn.Module,
    model_name: ModelName,
    noisy: torch.Tensor,
    clean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a model with its method-specific inputs.

    Args:
        model: Neural denoising model.
        model_name: One of ``dncnn``, ``ffdnet``, or ``cbdnet``.
        noisy: Noisy input batch.
        clean: Optional clean target used to estimate FFDNet training sigma.

    Returns:
        Tuple of denoised prediction and optional sigma prediction.

    Raises:
        ValueError: If ``model_name`` is unsupported or FFDNet lacks a clean target.
    """

    if model_name == "dncnn":
        return model(noisy), None
    if model_name == "ffdnet":
        if clean is None:
            sigma = torch.std(noisy, dim=(1, 2, 3), keepdim=True).expand(
                noisy.shape[0], 1, noisy.shape[-2], noisy.shape[-1]
            )
        else:
            sigma = _noise_map_from_pair(noisy, clean)
        return model(noisy, sigma), None
    if model_name == "cbdnet":
        prediction, sigma_pred = model(noisy)
        return prediction, sigma_pred
    raise ValueError(f"Unsupported model name: {model_name}")


def train_one_epoch(
    model: nn.Module,
    model_name: ModelName,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
) -> float:
    """Train a neural denoiser for one epoch.

    Args:
        model: Denoising model.
        model_name: Model identifier.
        loader: Training DataLoader.
        optimizer: Optimizer.
        device: Torch device.
        alpha: CBDNet asymmetric loss weight.

    Returns:
        Mean training loss.

    Raises:
        ValueError: If the loader is empty.
    """

    if len(loader) == 0:
        raise ValueError("Training loader is empty")
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=f"train-{model_name}", leave=False):
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction, sigma_pred = forward_model(model, model_name, noisy, clean)
        loss = mse_denoising_loss(prediction, clean)
        if model_name == "cbdnet" and sigma_pred is not None:
            sigma_gt = residual_sigma_target(noisy, clean)
            loss = loss + asymmetric_loss(sigma_pred, sigma_gt, alpha)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / len(loader)


def validate(
    model: nn.Module,
    model_name: ModelName,
    loader: DataLoader,
    device: torch.device,
    alpha: float,
) -> float:
    """Evaluate validation loss.

    Args:
        model: Denoising model.
        model_name: Model identifier.
        loader: Validation DataLoader.
        device: Torch device.
        alpha: CBDNet asymmetric loss weight.

    Returns:
        Mean validation loss.

    Raises:
        ValueError: If the loader is empty.
    """

    if len(loader) == 0:
        raise ValueError("Validation loader is empty")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"val-{model_name}", leave=False):
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            prediction, sigma_pred = forward_model(model, model_name, noisy, clean)
            loss = mse_denoising_loss(prediction, clean)
            if model_name == "cbdnet" and sigma_pred is not None:
                loss = loss + asymmetric_loss(sigma_pred, residual_sigma_target(noisy, clean), alpha)
            total_loss += float(loss.detach().cpu())
    return total_loss / len(loader)


def fit(
    model: nn.Module,
    model_name: ModelName,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
    checkpoint_path: Path,
    wandb_run: Any | None = None,
) -> Path:
    """Fine-tune a model and save the best validation checkpoint.

    Args:
        model: Denoising model.
        model_name: Model identifier.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Experiment configuration.
        device: Torch device.
        checkpoint_path: Output checkpoint path.
        wandb_run: Optional W&B run object.

    Returns:
        Path to the best checkpoint.

    Raises:
        ValueError: If scheduler type is unsupported.
    """

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    if config.training.lr_scheduler == "cosine":
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.num_epochs
        )
    elif config.training.lr_scheduler in {"none", ""}:
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {config.training.lr_scheduler}")
    stopper = EarlyStopping(config.training.early_stopping_patience)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    alpha = config.models.cbdnet.asymmetric_alpha
    for epoch in range(config.training.num_epochs):
        train_loss = train_one_epoch(model, model_name, train_loader, optimizer, device, alpha)
        val_loss = validate(model, model_name, val_loader, device, alpha)
        if scheduler is not None:
            scheduler.step()
        improved = stopper.update(val_loss)
        LOGGER.info(
            "epoch=%s model=%s train_loss=%.6f val_loss=%.6f improved=%s",
            epoch + 1,
            model_name,
            train_loss,
            val_loss,
            improved,
        )
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if improved:
            torch.save({"model_state_dict": model.state_dict(), "config": config, "model_name": model_name}, checkpoint_path)
        if stopper.should_stop():
            LOGGER.info("Early stopping after epoch %s", epoch + 1)
            break
    return checkpoint_path
