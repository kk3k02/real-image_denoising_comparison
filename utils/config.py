"""Configuration dataclasses and YAML loader for the denoising pipeline.

This module centralizes all experiment parameters in a typed ``Config`` object.
Pipeline modules receive this object instead of hardcoding constants, which
keeps scripts reproducible across Colab and Windows runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


Domain = Literal["srgb", "raw"]


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths used by all pipeline stages.

    Args:
        sidd_medium_root: Root folder containing SIDD sRGB pairs.
        sidd_medium_raw_root: Root folder containing SIDD RAW pairs.
        midd_root: Root folder containing MIDD RAW pairs.
        splits_json: JSON file storing deterministic scene-level splits.
        checkpoints_dir: Directory for model checkpoints.
        results_dir: Directory for CSV outputs.

    Example:
        >>> cfg = load_config(Path("config/experiment_config.yaml"))
        >>> cfg.paths.results_dir
        PosixPath('results')
    """

    sidd_medium_root: Path
    sidd_medium_raw_root: Path
    midd_root: Path
    splits_json: Path
    checkpoints_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class DataConfig:
    """Dataset configuration shared by loaders and split builders.

    Args:
        patch_size: Non-overlapping patch size used during fine-tuning.
        train_val_test_ratio: Scene-level train/validation/test fractions.
        domains: Supported image domains.
        expected_sidd_pairs: Expected number of SIDD Medium noisy/clean pairs.
        expected_midd_pairs: Expected number of MIDD noisy/clean pairs.

    Example:
        >>> cfg = load_config(Path("config/experiment_config.yaml"))
        >>> "srgb" in cfg.data.domains
        True
    """

    patch_size: int
    train_val_test_ratio: tuple[float, float, float]
    domains: tuple[str, ...]
    expected_sidd_pairs: int
    expected_midd_pairs: int


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters used by the fine-tuning loop.

    Args:
        batch_size: Mini-batch size.
        num_epochs: Maximum number of fine-tuning epochs.
        learning_rate: Initial optimizer learning rate.
        lr_scheduler: Scheduler name.
        early_stopping_patience: Epochs without validation improvement before stopping.
        num_workers: DataLoader worker count.

    Example:
        >>> cfg = load_config(Path("config/experiment_config.yaml"))
        >>> cfg.training.batch_size
        32
    """

    batch_size: int
    num_epochs: int
    learning_rate: float
    lr_scheduler: str
    early_stopping_patience: int
    num_workers: int


@dataclass(frozen=True)
class DnCNNConfig:
    """DnCNN architecture parameters.

    Args:
        num_layers: Total convolutional layers including first and last.
        num_channels: Hidden feature width.
        blind: Whether the model is trained without explicit sigma input.

    Example:
        >>> cfg = load_config(Path("config/experiment_config.yaml"))
        >>> cfg.models.dncnn.blind
        True
    """

    num_layers: int
    num_channels: int
    blind: bool


@dataclass(frozen=True)
class FFDNetConfig:
    """FFDNet architecture parameters.

    Args:
        num_layers: Number of convolutional blocks before reconstruction.
        num_channels: Hidden feature width.
        downscale_factor: Pixel-unshuffle factor.

    Example:
        >>> cfg = load_config(Path("config/experiment_config.yaml"))
        >>> cfg.models.ffdnet.downscale_factor
        2
    """

    num_layers: int
    num_channels: int
    downscale_factor: int


@dataclass(frozen=True)
class CBDNetConfig:
    """CBDNet architecture and loss parameters.

    Args:
        cnne_layers: Number of convolutional layers in the noise estimator.
        cnnd_channels: Base channel width in the denoising U-Net.
        asymmetric_alpha: Asymmetric loss weight.

    Example:
        >>> cfg = load_config(Path("config/experiment_config.yaml"))
        >>> cfg.models.cbdnet.asymmetric_alpha
        0.3
    """

    cnne_layers: int
    cnnd_channels: int
    asymmetric_alpha: float


@dataclass(frozen=True)
class BM3DConfig:
    """BM3D wrapper parameters.

    Args:
        sigma_estimation: Sigma source selector for BM3D.

    Example:
        >>> cfg = load_config(Path("config/experiment_config.yaml"))
        >>> cfg.models.bm3d.sigma_estimation
        'mad'
    """

    sigma_estimation: str


@dataclass(frozen=True)
class ModelsConfig:
    """Container for all model-specific configuration blocks.

    Args:
        dncnn: DnCNN parameters.
        ffdnet: FFDNet parameters.
        cbdnet: CBDNet parameters.
        bm3d: BM3D parameters.

    Example:
        >>> cfg = load_config(Path("config/experiment_config.yaml"))
        >>> cfg.models.dncnn.num_layers
        17
    """

    dncnn: DnCNNConfig
    ffdnet: FFDNetConfig
    cbdnet: CBDNetConfig
    bm3d: BM3DConfig


@dataclass(frozen=True)
class BenchmarkingConfig:
    """Timing protocol settings for CPU and GPU inference.

    Args:
        gpu_warmup_runs: CUDA warmup iterations.
        gpu_timed_runs: CUDA timed iterations.
        cpu_timed_runs: CPU timed iterations.
        cpu_num_threads: CPU thread count for single-core measurements.
        test_image_size: Synthetic input size for benchmark scripts.

    Example:
        >>> cfg = load_config(Path("config/experiment_config.yaml"))
        >>> cfg.benchmarking.gpu_timed_runs
        50
    """

    gpu_warmup_runs: int
    gpu_timed_runs: int
    cpu_timed_runs: int
    cpu_num_threads: int
    test_image_size: tuple[int, int]


@dataclass(frozen=True)
class LoggingConfig:
    """Runtime logging and optional Weights & Biases settings.

    Args:
        level: Python logging level name.
        wandb_project: W&B project name.
        wandb_enabled: Whether to initialize W&B.

    Example:
        >>> cfg = load_config(Path("config/experiment_config.yaml"))
        >>> cfg.logging.level
        'INFO'
    """

    level: str
    wandb_project: str
    wandb_enabled: bool


@dataclass(frozen=True)
class Config:
    """Top-level experiment configuration.

    Args:
        seed: Global reproducibility seed.
        paths: Filesystem path block.
        data: Dataset parameter block.
        training: Fine-tuning parameter block.
        models: Model parameter block.
        benchmarking: Timing protocol block.
        logging: Logging block.

    Example:
        >>> cfg = load_config(Path("config/experiment_config.yaml"))
        >>> cfg.seed
        42
    """

    seed: int
    paths: PathsConfig
    data: DataConfig
    training: TrainingConfig
    models: ModelsConfig
    benchmarking: BenchmarkingConfig
    logging: LoggingConfig


def load_config(path: Path) -> Config:
    """Load an experiment YAML file into typed dataclasses.

    Args:
        path: Path to ``experiment_config.yaml``.

    Returns:
        Parsed immutable configuration object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required fields are missing or ratios are invalid.
    """

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    required = {"seed", "paths", "data", "training", "models", "benchmarking", "logging"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"Config is missing required sections: {sorted(missing)}")

    ratios = tuple(float(x) for x in raw["data"]["train_val_test_ratio"])
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("data.train_val_test_ratio must contain three values summing to 1.0")

    paths = PathsConfig(**{key: Path(value) for key, value in raw["paths"].items()})
    data = DataConfig(
        patch_size=int(raw["data"]["patch_size"]),
        train_val_test_ratio=ratios,
        domains=tuple(raw["data"]["domains"]),
        expected_sidd_pairs=int(raw["data"]["expected_sidd_pairs"]),
        expected_midd_pairs=int(raw["data"]["expected_midd_pairs"]),
    )
    training = TrainingConfig(**raw["training"])
    models = ModelsConfig(
        dncnn=DnCNNConfig(**raw["models"]["dncnn"]),
        ffdnet=FFDNetConfig(**raw["models"]["ffdnet"]),
        cbdnet=CBDNetConfig(**raw["models"]["cbdnet"]),
        bm3d=BM3DConfig(**raw["models"]["bm3d"]),
    )
    benchmarking = BenchmarkingConfig(
        gpu_warmup_runs=int(raw["benchmarking"]["gpu_warmup_runs"]),
        gpu_timed_runs=int(raw["benchmarking"]["gpu_timed_runs"]),
        cpu_timed_runs=int(raw["benchmarking"]["cpu_timed_runs"]),
        cpu_num_threads=int(raw["benchmarking"]["cpu_num_threads"]),
        test_image_size=tuple(int(x) for x in raw["benchmarking"]["test_image_size"]),
    )
    logging_config = LoggingConfig(**raw["logging"])
    return Config(
        seed=int(raw["seed"]),
        paths=paths,
        data=data,
        training=training,
        models=models,
        benchmarking=benchmarking,
        logging=logging_config,
    )
