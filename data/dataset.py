"""PyTorch datasets for SIDD Medium and MIDD noisy/clean image pairs.

The datasets enforce domain separation through an explicit ``domain`` argument.
Training mode expands each full-resolution pair into non-overlapping patches,
while validation and test modes return full images for metric computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
from torch.utils.data import Dataset

from data.preprocessing import (
    augment_pair,
    extract_non_overlapping_patches,
    load_raw_mosaic,
    load_srgb_image,
    to_tensor,
)

Domain = Literal["srgb", "raw"]
Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class ImagePair:
    """Noisy/clean image pair discovered on disk.

    Args:
        noisy_path: Path to the noisy observation.
        clean_path: Path to the aligned clean target.
        image_id: Stable identifier used in result CSVs.
        scene_id: Scene identifier used for deterministic splits.

    Example:
        >>> pair = ImagePair(Path("NOISY.png"), Path("GT.png"), "img1", "scene1")
        >>> pair.image_id
        'img1'
    """

    noisy_path: Path
    clean_path: Path
    image_id: str
    scene_id: str


class PairedDenoisingDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Base dataset for aligned noisy/clean denoising pairs.

    Args:
        pairs: Aligned noisy/clean pairs.
        domain: ``srgb`` or ``raw``.
        split: Dataset split name.
        patch_size: Non-overlapping patch size for training.
        augment: Whether to apply paired training augmentation.

    Example:
        >>> ds = PairedDenoisingDataset([], "srgb", "test", 128, False)
        >>> len(ds)
        0
    """

    def __init__(
        self,
        pairs: list[ImagePair],
        domain: Domain,
        split: Split,
        patch_size: int,
        augment: bool,
    ) -> None:
        if domain not in {"srgb", "raw"}:
            raise ValueError(f"Unsupported domain: {domain}")
        self.pairs = pairs
        self.domain = domain
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and split == "train"
        self.loader: Callable[[Path], object] = load_srgb_image if domain == "srgb" else load_raw_mosaic
        self.index: list[tuple[int, int | None]] = self._build_index()

    def _build_index(self) -> list[tuple[int, int | None]]:
        """Build pair/patch lookup entries.

        Args:
            None.

        Returns:
            List of ``(pair_index, patch_index)`` entries.

        Raises:
            ValueError: If patch extraction fails for training data.
        """

        if self.split != "train":
            return [(idx, None) for idx in range(len(self.pairs))]
        entries: list[tuple[int, int | None]] = []
        for idx, pair in enumerate(self.pairs):
            noisy = self.loader(pair.noisy_path)
            patch_count = len(extract_non_overlapping_patches(noisy, self.patch_size))
            entries.extend((idx, patch_idx) for patch_idx in range(patch_count))
        return entries

    def __len__(self) -> int:
        """Return dataset length.

        Args:
            None.

        Returns:
            Number of full images or training patches.

        Raises:
            None.
        """

        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        """Load one denoising sample.

        Args:
            item: Dataset index.

        Returns:
            Dictionary with noisy tensor, clean tensor, image id, scene id, and domain.

        Raises:
            IndexError: If ``item`` is outside the dataset range.
            ValueError: If paired images have incompatible shapes.
        """

        if item < 0 or item >= len(self.index):
            raise IndexError(item)
        pair_idx, patch_idx = self.index[item]
        pair = self.pairs[pair_idx]
        noisy = self.loader(pair.noisy_path)
        clean = self.loader(pair.clean_path)
        if noisy.shape != clean.shape:
            raise ValueError(f"Pair shape mismatch for {pair.image_id}: {noisy.shape} vs {clean.shape}")
        if patch_idx is not None:
            noisy = extract_non_overlapping_patches(noisy, self.patch_size)[patch_idx]
            clean = extract_non_overlapping_patches(clean, self.patch_size)[patch_idx]
        if self.augment:
            noisy, clean = augment_pair(noisy, clean)
        return {
            "noisy": to_tensor(noisy),
            "clean": to_tensor(clean),
            "image_id": pair.image_id if patch_idx is None else f"{pair.image_id}_patch_{patch_idx:05d}",
            "scene_id": pair.scene_id,
            "domain": self.domain,
        }


def _discover_pairs(root: Path, scene_filter: set[str] | None = None) -> list[ImagePair]:
    """Discover noisy/clean pairs under a dataset root.

    Args:
        root: Dataset root.
        scene_filter: Optional set of scene ids to keep.

    Returns:
        Sorted image pair list.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        ValueError: If no pairs are discovered.
    """

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    noisy_markers = ("NOISY", "Noisy", "noisy")
    clean_markers = ("GT", "CLEAN", "Clean", "clean", "GROUNDTRUTH", "GroundTruth")
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and any(marker in path.name for marker in noisy_markers)
    )
    pairs: list[ImagePair] = []
    for noisy_path in candidates:
        clean_path: Path | None = None
        for noisy_marker in noisy_markers:
            if noisy_marker not in noisy_path.name:
                continue
            for clean_marker in clean_markers:
                candidate = noisy_path.with_name(noisy_path.name.replace(noisy_marker, clean_marker))
                if candidate.exists():
                    clean_path = candidate
                    break
            if clean_path is not None:
                break
        if clean_path is None:
            continue
        scene_id = noisy_path.parent.name
        if scene_filter is not None and scene_id not in scene_filter:
            continue
        image_id = noisy_path.relative_to(root).with_suffix("").as_posix().replace("/", "__")
        pairs.append(ImagePair(noisy_path, clean_path, image_id, scene_id))
    if not pairs:
        raise ValueError(f"No noisy/clean pairs found under {root}")
    return pairs


class SIDDDataset(PairedDenoisingDataset):
    """SIDD Medium paired denoising dataset.

    Args:
        root: SIDD sRGB or RAW root depending on ``domain``.
        domain: ``srgb`` or ``raw``.
        split: Split name.
        patch_size: Training patch size.
        scenes: Optional scene ids selected by deterministic split JSON.
        augment: Whether to apply training augmentation.

    Example:
        >>> ds = SIDDDataset(Path("data/raw/SIDD_Medium_Srgb"), "srgb", "train", 128, [])
        >>> isinstance(ds.domain, str)
        True
    """

    def __init__(
        self,
        root: Path,
        domain: Domain,
        split: Split,
        patch_size: int,
        scenes: list[str] | None = None,
        augment: bool = False,
    ) -> None:
        pairs = _discover_pairs(root, set(scenes) if scenes else None)
        super().__init__(pairs=pairs, domain=domain, split=split, patch_size=patch_size, augment=augment)


class MIDDDataset(PairedDenoisingDataset):
    """MIDD RAW paired denoising dataset.

    Args:
        root: MIDD RAW root.
        domain: Must be ``raw`` because MIDD is RAW-only.
        split: Split name.
        patch_size: Training patch size.
        scenes: Optional scene/sensor ids selected by split JSON.
        augment: Whether to apply training augmentation.

    Example:
        >>> MIDDDataset(Path("missing"), "srgb", "test", 128)
        Traceback (most recent call last):
        ...
        ValueError: MIDD supports only the raw domain
    """

    def __init__(
        self,
        root: Path,
        domain: Domain,
        split: Split,
        patch_size: int,
        scenes: list[str] | None = None,
        augment: bool = False,
    ) -> None:
        if domain != "raw":
            raise ValueError("MIDD supports only the raw domain")
        pairs = _discover_pairs(root, set(scenes) if scenes else None)
        super().__init__(pairs=pairs, domain=domain, split=split, patch_size=patch_size, augment=augment)
