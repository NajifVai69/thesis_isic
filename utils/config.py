"""Config + path resolution for the ISIC-2019 pipeline.

Everything that touches the filesystem goes through here so the rest of the
codebase stays platform-agnostic. The dataset path lives in configs/default.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class Paths:
    root: Path
    work_dir: Path
    groundtruth_csv: Path
    metadata_csv: Path
    flat_input_dir: Path

    # Artifacts produced by preprocessing
    @property
    def phash_csv(self) -> Path:
        return self.work_dir / "phash.csv"

    @property
    def dedup_keep_csv(self) -> Path:
        return self.work_dir / "dedup_keep.csv"

    @property
    def split_csv(self) -> Path:
        return self.work_dir / "split.csv"

    @property
    def memmap_images(self) -> Path:
        return self.work_dir / "images.uint8.npy"

    @property
    def memmap_labels(self) -> Path:
        return self.work_dir / "labels.int64.npy"

    @property
    def memmap_meta(self) -> Path:
        return self.work_dir / "meta.npz"

    @property
    def memmap_index_csv(self) -> Path:
        return self.work_dir / "memmap_index.csv"


def load_config(config_path: str | Path = "configs/default.yaml") -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_paths(cfg: dict) -> Paths:
    root = Path(cfg["data"]["root"])
    work_dir = Path(cfg["data"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    return Paths(
        root=root,
        work_dir=work_dir,
        groundtruth_csv=root / cfg["data"]["groundtruth_csv"],
        metadata_csv=root / cfg["data"]["metadata_csv"],
        flat_input_dir=root / cfg["data"]["flat_input_dir"],
    )


def resolve_image_path(image_id: str, class_name: str, paths: Paths) -> Optional[Path]:
    """Return the on-disk path for an image given its id and class.

    Tries the per-class folder layout first (matches the Kaggle mirror in the
    screenshot), then falls back to the flat-folder layout. Tries .jpg, .jpeg,
    .png in that order. Returns None if nothing is found — caller decides what
    to do (we log and skip; we do NOT silently insert blanks).
    """
    for ext in (".jpg", ".jpeg", ".png", ".JPG"):
        # per-class layout
        p = paths.root / class_name / f"{image_id}{ext}"
        if p.exists():
            return p
        # flat layout
        p = paths.flat_input_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None
