"""Evaluation runner for all denoising methods and datasets.

The evaluator executes BM3D and neural methods on full test images, records
per-image PSNR/SSIM and wall-clock inference time, and writes CSV files for
thesis tables and downstream notebook analysis.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import psnr_cpu, ssim_cpu, tensor_to_numpy_image
from models.bm3d_wrapper import BM3DDenoiser
from training.trainer import forward_model

LOGGER = logging.getLogger(__name__)
MethodName = Literal["bm3d", "dncnn", "ffdnet", "cbdnet"]


class Evaluator:
    """Run denoising evaluation and save per-image results.

    Args:
        results_dir: Directory where CSV files are written.
        device: Torch device for neural methods.

    Example:
        >>> evaluator = Evaluator(Path("results"), torch.device("cpu"))
        >>> evaluator.results_dir.name
        'results'
    """

    def __init__(self, results_dir: Path, device: torch.device) -> None:
        self.results_dir = results_dir
        self.device = device
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_bm3d(
        self,
        loader: DataLoader,
        denoiser: BM3DDenoiser,
        dataset_name: str,
        domain: str,
    ) -> pd.DataFrame:
        """Evaluate BM3D on a dataset loader.

        Args:
            loader: Test DataLoader with batch size one.
            denoiser: BM3D wrapper.
            dataset_name: Dataset label for CSV grouping.
            domain: ``srgb`` or ``raw``.

        Returns:
            Per-image result dataframe.

        Raises:
            ValueError: If loader batches are not single images.
        """

        rows: list[dict[str, float | str]] = []
        for batch in tqdm(loader, desc=f"eval-bm3d-{dataset_name}-{domain}"):
            noisy_tensor = batch["noisy"]
            clean_tensor = batch["clean"]
            if noisy_tensor.shape[0] != 1:
                raise ValueError("BM3D evaluation expects batch_size=1")
            noisy_np = tensor_to_numpy_image(noisy_tensor)
            start = time.perf_counter()
            denoised_np = denoiser.denoise(noisy_np, domain)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            denoised_tensor = torch.from_numpy(
                denoised_np[None, None, ...] if denoised_np.ndim == 2 else np.transpose(denoised_np, (2, 0, 1))[None, ...]
            ).float()
            rows.append(
                {
                    "image_id": str(batch["image_id"][0]),
                    "psnr": psnr_cpu(denoised_tensor, clean_tensor),
                    "ssim": ssim_cpu(denoised_tensor, clean_tensor),
                    "inference_time_ms": elapsed_ms,
                    "method": "bm3d",
                    "domain": domain,
                    "dataset": dataset_name,
                }
            )
        return self._save_results(pd.DataFrame(rows), "bm3d", domain, dataset_name)

    def evaluate_model(
        self,
        loader: DataLoader,
        model: nn.Module,
        method: Literal["dncnn", "ffdnet", "cbdnet"],
        dataset_name: str,
        domain: str,
    ) -> pd.DataFrame:
        """Evaluate a neural denoising model.

        Args:
            loader: Test DataLoader with batch size one.
            model: Neural denoising model.
            method: Model identifier.
            dataset_name: Dataset label.
            domain: ``srgb`` or ``raw``.

        Returns:
            Per-image result dataframe.

        Raises:
            ValueError: If loader batches are not single images.
        """

        model.to(self.device)
        model.eval()
        rows: list[dict[str, float | str]] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"eval-{method}-{dataset_name}-{domain}"):
                noisy = batch["noisy"].to(self.device)
                clean = batch["clean"].to(self.device)
                if noisy.shape[0] != 1:
                    raise ValueError("Evaluation expects batch_size=1")
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                prediction, _ = forward_model(model, method, noisy, None)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                rows.append(
                    {
                        "image_id": str(batch["image_id"][0]),
                        "psnr": psnr_cpu(prediction.cpu(), clean.cpu()),
                        "ssim": ssim_cpu(prediction.cpu(), clean.cpu()),
                        "inference_time_ms": elapsed_ms,
                        "method": method,
                        "domain": domain,
                        "dataset": dataset_name,
                    }
                )
        return self._save_results(pd.DataFrame(rows), method, domain, dataset_name)

    def _save_results(self, dataframe: pd.DataFrame, method: str, domain: str, dataset_name: str) -> pd.DataFrame:
        """Save per-image CSV results.

        Args:
            dataframe: Result dataframe.
            method: Method label.
            domain: Domain label.
            dataset_name: Dataset label.

        Returns:
            The same dataframe for caller aggregation.

        Raises:
            ValueError: If required columns are missing.
        """

        required = {"image_id", "psnr", "ssim", "inference_time_ms"}
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(f"Missing result columns: {sorted(missing)}")
        output = self.results_dir / f"{method}_{domain}_{dataset_name}.csv"
        dataframe[["image_id", "psnr", "ssim", "inference_time_ms"]].to_csv(output, index=False)
        LOGGER.info("Saved %s", output)
        return dataframe


def save_summary(dataframes: list[pd.DataFrame], output_path: Path) -> pd.DataFrame:
    """Save aggregate mean/std metrics by method, domain, and dataset.

    Args:
        dataframes: Per-image result dataframes.
        output_path: Output summary CSV.

    Returns:
        Summary dataframe.

    Raises:
        ValueError: If no dataframes are provided.
    """

    if not dataframes:
        raise ValueError("No result dataframes to summarize")
    combined = pd.concat(dataframes, ignore_index=True)
    summary = (
        combined.groupby(["method", "domain", "dataset"], as_index=False)
        .agg(
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            inference_time_ms_mean=("inference_time_ms", "mean"),
            inference_time_ms_std=("inference_time_ms", "std"),
        )
        .fillna(0.0)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary
