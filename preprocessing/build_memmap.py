"""Pre-decode every kept image to a single uint8 memmap on disk.

Why: on Windows we must run with `num_workers=0` (multiprocessing
DataLoader hangs are a real and well-known issue). To compensate we
eliminate JPEG-decode-from-disk-per-batch entirely by decoding once
into a contiguous (N, H, W, 3) uint8 array we mmap at training time.

  - At 224x224x3 uint8 and ~25k images: ~3.6 GB on disk and in page cache.
    The user has 64 GB RAM, so the whole tensor stays resident after the
    first epoch and per-batch I/O collapses to a memcpy.
  - Color constancy is applied here, so training reads the corrected image
    directly with no per-batch CPU cost.
  - We also apply a 'short-side resize then center-crop' policy: aspect
    ratios in ISIC vary, and a naive resize would distort lesions.

Outputs written to work_dir:
  - images.uint8.npy       : (N, H, W, 3) uint8 memmap (header-prefixed .npy)
  - labels.int64.npy       : (N,) int64
  - meta.npz               : age_approx (float32, NaN for missing),
                             sex_idx, site_idx (int16, -1 for missing),
                             source_idx (int8)
  - memmap_index.csv       : the row order of the memmap with image_id, split, etc.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.color_constancy import shades_of_gray
from utils.config import load_config, resolve_paths, resolve_image_path

# Stable category orderings for the categorical metadata fields. We keep these
# explicit so train/val/test all use the same int encoding.
SEX_CATS = ["female", "male"]
SITE_CATS = [
    "anterior torso",
    "head/neck",
    "lateral torso",
    "lower extremity",
    "oral/genital",
    "palms/soles",
    "posterior torso",
    "upper extremity",
]
SOURCE_CATS = ["HAM", "BCN", "MSK", "UNK"]


def _encode_categorical(value, cats: list[str]) -> int:
    """Return index of value in cats, or -1 for missing/unknown."""
    if value is None:
        return -1
    if isinstance(value, float) and np.isnan(value):
        return -1
    s = str(value).strip().lower()
    for i, c in enumerate(cats):
        if s == c.lower():
            return i
    return -1


def _load_and_preprocess(path: Path, size: int, do_cc: bool, cc_p: int) -> np.ndarray:
    """Read an image, apply color constancy, short-side-resize + center-crop to
    (size, size, 3) uint8 RGB. Uses cv2 for speed; converts BGR->RGB explicitly.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cv2 failed to decode {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if do_cc:
        rgb = shades_of_gray(rgb, p=cc_p)

    h, w = rgb.shape[:2]
    short = min(h, w)
    scale = size / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    # INTER_AREA for downscale, INTER_LINEAR for upscale.
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)

    # center crop
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    out = resized[y0:y0 + size, x0:x0 + size]
    assert out.shape == (size, size, 3), f"got shape {out.shape}"
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    size = int(cfg["preprocessing"]["image_size"])
    cc_mode = str(cfg["preprocessing"]["color_constancy"])
    cc_p = int(cfg["preprocessing"]["cc_minkowski_p"])
    do_cc = cc_mode == "shades_of_gray"
    if cc_mode not in {"shades_of_gray", "none"}:
        raise ValueError(f"unknown color_constancy: {cc_mode}")

    print(f"[info] loading split from {paths.split_csv}")
    split = pd.read_csv(paths.split_csv)
    n = len(split)
    print(f"[info] memmap will hold {n} images at {size}x{size} uint8 ({n * size * size * 3 / 1e9:.2f} GB)")

    # Allocate the memmap via np.lib.format so it's a real .npy file we can
    # mmap directly at training time with np.load(..., mmap_mode='r').
    images_path = paths.memmap_images
    labels_path = paths.memmap_labels

    # np.lib.format.open_memmap writes the .npy header + zeros the file.
    images = np.lib.format.open_memmap(
        images_path, mode="w+", dtype=np.uint8, shape=(n, size, size, 3)
    )
    labels = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int64, shape=(n,)
    )

    age = np.full(n, np.nan, dtype=np.float32)
    sex_idx = np.full(n, -1, dtype=np.int16)
    site_idx = np.full(n, -1, dtype=np.int16)
    source_idx = np.full(n, -1, dtype=np.int8)

    # Pull metadata once for fast lookup
    meta_df = pd.read_csv(paths.metadata_csv)
    meta_df["image"] = meta_df["image"].astype(str)
    meta_lookup = meta_df.set_index("image").to_dict(orient="index")

    n_failed = 0
    for i in tqdm(range(n), desc="memmap"):
        row = split.iloc[i]
        image_id = str(row["image"])
        class_name = str(row["class"])
        try:
            path = resolve_image_path(image_id, class_name, paths)
            if path is None:
                raise FileNotFoundError(f"image not found for {image_id}")
            arr = _load_and_preprocess(path, size=size, do_cc=do_cc, cc_p=cc_p)
        except Exception as e:
            print(f"[warn] {image_id}: {e}", file=sys.stderr)
            arr = np.zeros((size, size, 3), dtype=np.uint8)  # placeholder
            n_failed += 1

        images[i] = arr
        labels[i] = int(row["label"])

        m = meta_lookup.get(image_id, {})
        a = m.get("age_approx", np.nan)
        try:
            age[i] = float(a) if a is not None and not (isinstance(a, float) and np.isnan(a)) else np.nan
        except (TypeError, ValueError):
            age[i] = np.nan
        sex_idx[i] = _encode_categorical(m.get("sex"), SEX_CATS)
        site_idx[i] = _encode_categorical(m.get("anatom_site_general"), SITE_CATS)
        source_idx[i] = _encode_categorical(row.get("source"), SOURCE_CATS)

    # Flush memmaps to disk
    images.flush()
    labels.flush()
    del images, labels

    np.savez(
        paths.memmap_meta,
        age=age,
        sex_idx=sex_idx,
        site_idx=site_idx,
        source_idx=source_idx,
        sex_cats=np.array(SEX_CATS),
        site_cats=np.array(SITE_CATS),
        source_cats=np.array(SOURCE_CATS),
    )

    # Write index so we know which row in the memmap corresponds to which split etc.
    idx = split.reset_index(drop=True).copy()
    idx["memmap_row"] = np.arange(n, dtype=np.int64)
    idx.to_csv(paths.memmap_index_csv, index=False)

    print(f"\n[ok]   wrote {paths.memmap_images}")
    print(f"[ok]   wrote {paths.memmap_labels}")
    print(f"[ok]   wrote {paths.memmap_meta}")
    print(f"[ok]   wrote {paths.memmap_index_csv}")
    if n_failed:
        print(f"[warn] {n_failed} images failed to decode and were stored as black placeholders.")


if __name__ == "__main__":
    main()
