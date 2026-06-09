"""Stratified train / val / test split.

Stratification:
  - On class label (so every class is represented in every split).
  - On source dataset (HAM / BCN / MSK) when enabled, so we never train on one
    source and test on another only.

We also respect HAM10000's lesion_id when available: multiple dermoscopy images
can share a lesion_id (same lesion, different acquisition). All images of the
same lesion go to the same split — otherwise a near-duplicate of a training
image can land in test.

Inputs:  dedup_keep.csv (from dedup_phash.py) and the metadata CSV.
Output:  split.csv with columns [image, label, class, source, lesion_id, split]
         where split ∈ {train, val, test}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, resolve_paths
from utils.labels import load_metadata


def _grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: str,
    sizes: tuple[float, float, float],
    seed: int,
) -> np.ndarray:
    """Assign each *group* to train/val/test, stratified by the group's stratify
    value (mode within the group). Returns a (len(df),) array of {0,1,2}
    (train, val, test) aligned to df's row order.
    """
    train_frac, val_frac, test_frac = sizes
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    # Reduce to one row per group with its stratify key (modal value in group).
    grp = (
        df.groupby(group_col)[stratify_col]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )

    # First peel off test (test_frac of groups), then split remainder into train+val.
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(grp[[group_col]], grp[stratify_col]))

    relative_val = val_frac / (train_frac + val_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
    sub = grp.iloc[trainval_idx].reset_index(drop=True)
    train_rel, val_rel = next(sss2.split(sub[[group_col]], sub[stratify_col]))

    train_groups = set(sub.iloc[train_rel][group_col].tolist())
    val_groups = set(sub.iloc[val_rel][group_col].tolist())
    test_groups = set(grp.iloc[test_idx][group_col].tolist())

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    out = np.empty(len(df), dtype=np.int8)
    g = df[group_col].to_numpy()
    for i in range(len(df)):
        gi = g[i]
        if gi in train_groups:
            out[i] = 0
        elif gi in val_groups:
            out[i] = 1
        elif gi in test_groups:
            out[i] = 2
        else:
            raise RuntimeError(f"group {gi!r} not assigned")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    seed = int(cfg["seed"])

    print(f"[info] loading dedup keep list from {paths.dedup_keep_csv}")
    keep = pd.read_csv(paths.dedup_keep_csv)
    keep = keep[keep["kept"]].copy()
    print(f"[info] {len(keep)} kept images after dedup")

    print(f"[info] loading metadata from {paths.metadata_csv}")
    meta = load_metadata(paths.metadata_csv)
    df = keep.merge(meta, on="image", how="left")

    # Build the grouping key:
    #   - if lesion_id is present, use it (so all images of the same lesion stay together)
    #   - otherwise the image itself is its own group
    if "lesion_id" in df.columns:
        df["group_key"] = df["lesion_id"].where(df["lesion_id"].notna(), df["image"])
    else:
        df["group_key"] = df["image"]

    # Build the stratify key:
    #   class only, or class + source.
    if bool(cfg["preprocessing"]["split"]["stratify_on_source"]):
        df["strat_key"] = df["class"].astype(str) + "|" + df["source"].astype(str)
    else:
        df["strat_key"] = df["class"].astype(str)

    sizes = (
        float(cfg["preprocessing"]["split"]["train"]),
        float(cfg["preprocessing"]["split"]["val"]),
        float(cfg["preprocessing"]["split"]["test"]),
    )
    print(f"[info] splitting train/val/test = {sizes}")
    code = _grouped_stratified_split(df, "group_key", "strat_key", sizes, seed)
    df["split"] = pd.Categorical.from_codes(code, categories=["train", "val", "test"])

    # Sanity report: class counts per split
    print("\n[info] class distribution per split:")
    pivot = df.pivot_table(index="class", columns="split", values="image", aggfunc="count", fill_value=0, observed=False)
    print(pivot.to_string())

    print("\n[info] source distribution per split:")
    pivot_src = df.pivot_table(index="source", columns="split", values="image", aggfunc="count", fill_value=0, observed=False)
    print(pivot_src.to_string())

    out = df[["image", "label", "class", "source", "lesion_id", "split"]].copy()
    out.to_csv(paths.split_csv, index=False)
    print(f"\n[ok]   wrote {paths.split_csv}")


if __name__ == "__main__":
    main()
