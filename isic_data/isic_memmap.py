"""PyTorch Dataset backed by the pre-decoded uint8 memmapped array.

Why a memmap Dataset?  On Windows, num_workers=0 is mandatory (DataLoader
multiprocessing deadlocks with CUDA + certain image libs). Pre-decoding all
images once into a contiguous (N,H,W,3) uint8 .npy array means __getitem__
reduces to a single pointer read — the OS page-cache keeps the whole array
warm after the first epoch so there is effectively no I/O overhead.

Usage:
    from datasets.isic_memmap import ISICDataset, build_dataloaders
    train_ds, val_ds, test_ds = ISICDataset.create_splits("configs/default.yaml")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds, batch_size=256
    )
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as tv2

from utils.config import load_config, resolve_paths

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── Transforms ────────────────────────────────────────────────────────────────

def build_train_transform(image_size: int = 224) -> tv2.Compose:
    """Per-sample augmentation applied in __getitem__ (CPU, single-threaded).

    Order rationale:
      - Spatial ops (flip, crop) first on uint8 → minimal memory overhead.
      - ColorJitter + RandAugment on float32 → all ops internally use float.
      - Normalize last.

    Dermoscopy notes:
      - hue=0 in ColorJitter: dermoscopy pigment is diagnostically meaningful;
        we do not shift hue.
      - scale=(0.8, 1.0) in RandomResizedCrop: avoids cropping too much of the
        lesion while still providing spatial variation.
    """
    return tv2.Compose([
        tv2.ToImage(),                                          # numpy HWC → uint8 CHW tensor
        tv2.RandomHorizontalFlip(p=0.5),
        tv2.RandomVerticalFlip(p=0.5),
        tv2.RandomResizedCrop(size=image_size, scale=(0.80, 1.0), antialias=True),
        tv2.ToDtype(torch.float32, scale=True),                 # uint8 → float32 [0, 1]
        tv2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
        tv2.RandAugment(num_ops=2, magnitude=9),
        tv2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform() -> tv2.Compose:
    """Deterministic transform for val / test / TTA views."""
    return tv2.Compose([
        tv2.ToImage(),
        tv2.ToDtype(torch.float32, scale=True),
        tv2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class ISICDataset(Dataset):
    """Reads images and labels from the pre-built uint8 memmapped array.

    Metadata (age, sex, anatomical site) is also available for models that use
    the cross-attention head. When `use_metadata=False` (default for baselines),
    __getitem__ returns just (image_tensor, label).

    Args:
        images_path:     path to images.uint8.npy
        labels_path:     path to labels.int64.npy
        meta_path:       path to meta.npz
        index_csv:       path to memmap_index.csv  (row index + split column)
        split:           one of "train" | "val" | "test"
        transform:       callable applied to each uint8 HWC numpy array
        use_metadata:    if True, returns (image, label, meta_dict)
        age_stats:       (mean, std) for z-scoring age; if None and split=="train",
                         computed from this split and cached as age_stats.json
                         in the same directory as index_csv.
    """

    def __init__(
        self,
        images_path: Path,
        labels_path: Path,
        meta_path: Path,
        index_csv: Path,
        split: str,
        transform=None,
        use_metadata: bool = False,
        age_stats: Optional[tuple[float, float]] = None,
    ):
        assert split in ("train", "val", "test"), f"unknown split {split!r}"
        self.split = split
        self.transform = transform
        self.use_metadata = use_metadata

        # Load (mmap) the full array — the slice we keep is determined below.
        self._images = np.load(images_path, mmap_mode="r")  # (N, H, W, 3) uint8
        self._labels = np.load(labels_path, mmap_mode="r")  # (N,) int64
        meta_npz = np.load(meta_path, allow_pickle=True)
        self._age       = meta_npz["age"]        # (N,) float32, NaN for missing
        self._sex_idx   = meta_npz["sex_idx"]    # (N,) int16, -1 for missing
        self._site_idx  = meta_npz["site_idx"]   # (N,) int16, -1 for missing

        # Resolve which memmap rows belong to this split
        idx_df = pd.read_csv(index_csv)
        mask = idx_df["split"] == split
        self._rows   = idx_df.loc[mask, "memmap_row"].to_numpy(dtype=np.int64)
        self._labels_split = self._labels[self._rows]

        # Age normalisation (z-score); fit on train, reuse for val/test
        age_stats_path = Path(index_csv).parent / "age_stats.json"
        if age_stats is not None:
            self.age_mean, self.age_std = age_stats
        elif age_stats_path.exists():
            saved = json.loads(age_stats_path.read_text())
            self.age_mean, self.age_std = saved["mean"], saved["std"]
        else:
            # Compute from THIS split — should be called on train split first.
            train_ages = self._age[self._rows]
            valid = train_ages[~np.isnan(train_ages)]
            self.age_mean = float(valid.mean()) if len(valid) else 50.0
            self.age_std  = float(valid.std())  if len(valid) else 20.0
            if self.age_std < 1e-6:
                self.age_std = 1.0
            age_stats_path.write_text(json.dumps({"mean": self.age_mean, "std": self.age_std}))

        # Class counts (useful externally for CB-Focal weight computation)
        self._class_counts = np.bincount(self._labels_split, minlength=8)

    # -- convenience constructors -----------------------------------------

    @classmethod
    def create_splits(
        cls,
        config_path: str | Path = "configs/default.yaml",
        use_metadata: bool = False,
    ) -> tuple["ISICDataset", "ISICDataset", "ISICDataset"]:
        """Build train/val/test datasets from a config file."""
        cfg = load_config(config_path)
        paths = resolve_paths(cfg)
        size = int(cfg["preprocessing"]["image_size"])

        common = dict(
            images_path=paths.memmap_images,
            labels_path=paths.memmap_labels,
            meta_path=paths.memmap_meta,
            index_csv=paths.memmap_index_csv,
            use_metadata=use_metadata,
        )
        # Build train first so age_stats.json is written before val/test need it.
        train_ds = cls(**common, split="train", transform=build_train_transform(size))
        age_stats = (train_ds.age_mean, train_ds.age_std)
        val_ds   = cls(**common, split="val",  transform=build_eval_transform(), age_stats=age_stats)
        test_ds  = cls(**common, split="test", transform=build_eval_transform(), age_stats=age_stats)
        return train_ds, val_ds, test_ds

    # -- dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int):
        row = int(self._rows[i])
        img_np = np.array(self._images[row])          # copy out of mmap → (H,W,3) uint8
        label  = int(self._labels[row])

        if self.transform is not None:
            img = self.transform(img_np)              # → float32 CHW tensor
        else:
            img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        if not self.use_metadata:
            return img, label

        # Normalise age; replace NaN with 0 after z-scoring (treated as "mean")
        age_raw = float(self._age[row])
        if np.isnan(age_raw):
            age_norm = 0.0
            age_missing = 1
        else:
            age_norm = (age_raw - self.age_mean) / self.age_std
            age_missing = 0

        meta = {
            "age":          torch.tensor(age_norm,                  dtype=torch.float32),
            "age_missing":  torch.tensor(age_missing,               dtype=torch.long),
            "sex_idx":      torch.tensor(int(self._sex_idx[row]),   dtype=torch.long),
            "site_idx":     torch.tensor(int(self._site_idx[row]),  dtype=torch.long),
        }
        return img, label, meta

    @property
    def class_counts(self) -> np.ndarray:
        """(8,) array: number of training examples per class, in label order."""
        return self._class_counts.copy()


# ── Sampler helper ────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: ISICDataset) -> WeightedRandomSampler:
    """WeightedRandomSampler with per-class weight = 1 / sqrt(class_count).

    Why sqrt instead of 1/n (inverse frequency)?
      - Pure inverse frequency would over-correct on ISIC: it gives a 53× ratio
        (NV vs DF) that collapses to the minority classes, same failure mode as
        β=0.9999 in CB-Focal but via sampling.
      - sqrt compression gives a ~7× ratio — comparable to our CB-Focal β=0.999 —
        and stacks multiplicatively with CB-Focal for a soft double boost without
        over-correction.
      - Combined effect: minority classes appear more often AND carry higher loss
        weight when they do appear.  Empirically ~1–2% BMA gain on ISIC-like
        data with severe imbalance.

    Args:
        dataset: an ISICDataset (must have _labels_split attribute).

    Returns:
        WeightedRandomSampler that samples len(dataset) indices with replacement.
    """
    counts = dataset.class_counts.clip(1).astype(np.float64)     # (C,) avoid div-by-zero
    weights_per_class = 1.0 / np.sqrt(counts)                    # sqrt-inverse weighting
    sample_weights = weights_per_class[dataset._labels_split]    # (N,) per-sample
    return WeightedRandomSampler(
        weights    = torch.from_numpy(sample_weights).float(),
        num_samples= len(dataset),
        replacement= True,
    )


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    train_ds: ISICDataset,
    val_ds:   ISICDataset,
    test_ds:  ISICDataset,
    batch_size: int = 256,
    use_weighted_sampler: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Windows-safe DataLoaders: num_workers=0, pin_memory=True.

    pin_memory=True is still beneficial even with num_workers=0 — it allocates
    the output tensors in pinned (page-locked) memory so the GPU DMA transfer
    overlaps with CPU work.

    Args:
        use_weighted_sampler: if True, replaces shuffle=True with a
            WeightedRandomSampler (1/sqrt(class_count) weights). Combines with
            CB-Focal for a mild double boost on minority classes.
    """
    _common = dict(num_workers=0, pin_memory=True, persistent_workers=False)
    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, **_common)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, **_common)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **_common)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **_common)
    return train_loader, val_loader, test_loader