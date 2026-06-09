"""pHash-based near-duplicate detection across ISIC-2019.

ISIC-2019 is a union of HAM10000 + BCN_20000 + MSK. There are known near-
duplicates (same lesion under slightly different crops/lighting). Letting one
copy land in train and another in test inflates scores by 1-3% BMA — exactly
the kind of "novelty" that gets a paper rejected.

This script:
  1. Computes a perceptual hash (pHash) for every training image.
  2. Groups images whose pairwise Hamming distance is <= threshold.
  3. Within each duplicate group, keeps the one image with the lexicographically
     smallest image-id (deterministic), drops the rest.
  4. Writes phash.csv (image_id, source, hash_hex) and dedup_keep.csv
     (image_id, kept: bool, group_id).

Usage:
    python -m preprocessing.dedup_phash --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Make repo root importable when run as `python -m preprocessing.dedup_phash`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, resolve_paths, resolve_image_path
from utils.labels import load_groundtruth


def _phash_uint64(img: Image.Image, hash_size: int) -> int:
    h = imagehash.phash(img, hash_size=hash_size)
    # imagehash stores hash.hash as a 2D bool array of size hash_size x hash_size.
    bits = h.hash.flatten().astype(np.uint8)
    if bits.size > 64:
        raise ValueError("hash_size > 8 would exceed uint64; reduce hash_size")
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def compute_phashes(gt: pd.DataFrame, paths, hash_size: int) -> pd.DataFrame:
    """Compute pHash for every image in the groundtruth table.

    Skips (with a warning) any image we cannot locate on disk; logs the count.
    Returns a DataFrame with columns [image, class, hash].
    """
    hashes = np.zeros(len(gt), dtype=np.uint64)
    found = np.ones(len(gt), dtype=bool)

    # "class" is a Python keyword and pandas itertuples renames it; use plain
    # column access via .iat in a zipped loop instead — fast enough at N=25k.
    image_ids = gt["image"].to_numpy()
    class_names = gt["class"].to_numpy()
    for i in tqdm(range(len(gt)), desc="pHash"):
        path = resolve_image_path(str(image_ids[i]), str(class_names[i]), paths)
        if path is None:
            found[i] = False
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                hashes[i] = _phash_uint64(im, hash_size)
        except Exception as e:
            print(f"[warn] failed to hash {image_ids[i]}: {e}", file=sys.stderr)
            found[i] = False

    n_missing = int((~found).sum())
    if n_missing:
        print(f"[warn] {n_missing}/{len(gt)} images not found or unreadable.", file=sys.stderr)

    out = gt.loc[found].copy()
    out["hash"] = hashes[found]
    return out.reset_index(drop=True)


def _popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorized 64-bit popcount via the bit-trick (no per-element Python)."""
    x = x.astype(np.uint64, copy=True)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0f0f0f0f0f0f0f0f)
    h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return ((x * h01) >> np.uint64(56)).astype(np.int32)


def find_duplicate_groups(hashes: np.ndarray, threshold: int) -> np.ndarray:
    """Union-find grouping of images by Hamming-distance(<=threshold).

    Computes pairwise hamming in chunks to keep memory bounded (~hundreds of MB
    at N=25k). Returns a (N,) int array of group ids in [0..G).
    """
    n = hashes.shape[0]
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # union by smaller index to make group representative deterministic
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    chunk = 1024
    H = hashes.astype(np.uint64)
    for start in tqdm(range(0, n, chunk), desc="dedup-pairwise"):
        end = min(start + chunk, n)
        block = H[start:end, None] ^ H[None, :]               # (chunk, N) uint64
        dist = _popcount64(block)                              # (chunk, N) int32
        # only consider j > i to avoid double work
        for ii in range(end - start):
            i_global = start + ii
            # mask to upper triangle
            row = dist[ii]
            row[: i_global + 1] = 100  # exclude self + already-handled
            matches = np.flatnonzero(row <= threshold)
            for j in matches:
                union(i_global, int(j))

    # path-compress all
    groups = np.array([find(i) for i in range(n)], dtype=np.int32)
    # remap to dense 0..G-1
    _, dense = np.unique(groups, return_inverse=True)
    return dense.astype(np.int32)


def select_keepers(df: pd.DataFrame) -> pd.DataFrame:
    """Within each group, keep the row with the lexicographically smallest image
    id. Deterministic and independent of input order."""
    df = df.copy()
    df["kept"] = False
    keepers = df.sort_values(["group_id", "image"]).groupby("group_id").head(1).index
    df.loc[keepers, "kept"] = True
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)

    print(f"[info] loading groundtruth from {paths.groundtruth_csv}")
    gt = load_groundtruth(paths.groundtruth_csv)
    print(f"[info] {len(gt)} labeled images across {gt['class'].nunique()} classes")

    print("[info] computing pHash for every image")
    hash_size = int(cfg["preprocessing"]["phash"]["hash_size"])
    df = compute_phashes(gt, paths, hash_size=hash_size)

    threshold = int(cfg["preprocessing"]["phash"]["hamming_threshold"])
    print(f"[info] grouping near-duplicates at hamming<={threshold}")
    df["group_id"] = find_duplicate_groups(df["hash"].to_numpy(dtype=np.uint64), threshold)

    df = select_keepers(df)

    # write phash.csv and dedup_keep.csv
    df_out = df[["image", "class", "label", "hash", "group_id", "kept"]].copy()
    df_out["hash_hex"] = df_out["hash"].map(lambda x: f"{int(x):016x}")
    df_out = df_out.drop(columns=["hash"])
    df_out.to_csv(paths.phash_csv, index=False)

    keep = df_out[["image", "class", "label", "group_id", "kept"]].copy()
    keep.to_csv(paths.dedup_keep_csv, index=False)

    n_groups = int(df["group_id"].nunique())
    n_kept = int(df["kept"].sum())
    n_dropped = int(len(df) - n_kept)
    print(f"[ok]   {len(df)} hashed -> {n_groups} groups -> {n_kept} kept ({n_dropped} duplicates dropped)")
    print(f"[ok]   wrote {paths.phash_csv}")
    print(f"[ok]   wrote {paths.dedup_keep_csv}")


if __name__ == "__main__":
    main()
