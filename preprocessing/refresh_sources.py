"""Re-label source column in split.csv and meta.npz without re-decoding images.

Run this once after patching utils/labels.py::infer_source().
It does NOT touch images.uint8.npy or labels.int64.npy.

Usage:
    python -m preprocessing.refresh_sources --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, resolve_paths
from utils.labels import load_metadata
from preprocessing.build_memmap import SOURCE_CATS, _encode_categorical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)

    # ── 1. Re-compute source in split.csv ─────────────────────────────────────
    print(f"[info] loading split from {paths.split_csv}")
    split = pd.read_csv(paths.split_csv)

    print(f"[info] loading metadata from {paths.metadata_csv}")
    meta = load_metadata(paths.metadata_csv)   # uses updated infer_source()

    # Merge new source column into split, drop old one
    split = split.drop(columns=["source"], errors="ignore")
    split = split.merge(meta[["image", "source"]], on="image", how="left")
    split["source"] = split["source"].fillna("MSK")  # conservative fallback

    print("\n[info] updated source distribution per split:")
    pivot = split.pivot_table(
        index="source", columns="split", values="image",
        aggfunc="count", fill_value=0, observed=False
    )
    print(pivot.to_string())

    split.to_csv(paths.split_csv, index=False)
    print(f"\n[ok]   wrote {paths.split_csv}")

    # ── 2. Re-encode source_idx in meta.npz ───────────────────────────────────
    if not paths.memmap_meta.exists():
        print("[warn] meta.npz not found — skipping npz update. Run build_memmap.py first.")
        return

    print(f"\n[info] updating source_idx in {paths.memmap_meta}")
    data = dict(np.load(paths.memmap_meta, allow_pickle=True))

    # Rebuild source_idx aligned to memmap_index row order
    idx = pd.read_csv(paths.memmap_index_csv)
    # Ensure we only have one source column after merge.
    idx = idx.drop(columns=["source"], errors="ignore")
    idx = idx.merge(split[["image", "source"]], on="image", how="left")
    idx["source"] = idx["source"].fillna("MSK")

    source_idx = np.array(
        [_encode_categorical(s, SOURCE_CATS) for s in idx["source"]],
        dtype=np.int8,
    )

    data["source_idx"] = source_idx
    data["source_cats"] = np.array(SOURCE_CATS)
    np.savez(paths.memmap_meta, **data)
    print(f"[ok]   wrote {paths.memmap_meta}")

    print("\n[ok] source refresh complete — images.uint8.npy untouched.")


if __name__ == "__main__":
    main()