"""Diagnostic: show what's actually in the image_id and lesion_id columns.

Run once to figure out what naming conventions the Kaggle mirror uses, then we
patch utils/labels.py::infer_source() to be correct.

Usage:
    python -m preprocessing.inspect_sources --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, resolve_paths
from utils.labels import load_groundtruth, infer_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)

    gt = load_groundtruth(paths.groundtruth_csv)
    meta = pd.read_csv(paths.metadata_csv)
    meta["image"] = meta["image"].astype(str)
    df = gt.merge(meta, on="image", how="left")
    print(f"[info] {len(df)} images total\n")

    # 1. image_id prefix distribution
    print("=" * 70)
    print("IMAGE_ID PREFIXES (everything before the first digit or underscore)")
    print("=" * 70)
    def img_prefix(s: str) -> str:
        m = re.match(r"^([A-Za-z_]+)", str(s))
        return m.group(1) if m else "(none)"
    img_pref = df["image"].map(img_prefix).value_counts()
    print(img_pref.to_string())

    # 2. lesion_id prefix distribution (if column exists)
    print("\n" + "=" * 70)
    print("LESION_ID PREFIXES")
    print("=" * 70)
    if "lesion_id" in df.columns:
        non_null = df["lesion_id"].notna()
        print(f"lesion_id present:  {int(non_null.sum())}/{len(df)} ({100*non_null.mean():.1f}%)")
        if non_null.any():
            les_pref = df.loc[non_null, "lesion_id"].astype(str).map(img_prefix).value_counts()
            print(les_pref.to_string())
    else:
        print("(no lesion_id column)")

    # 3. Cross-tabulate image_id prefix x lesion_id prefix
    print("\n" + "=" * 70)
    print("IMAGE_PREFIX x LESION_PREFIX (top combos)")
    print("=" * 70)
    if "lesion_id" in df.columns:
        df["_ip"] = df["image"].map(img_prefix)
        df["_lp"] = df["lesion_id"].astype(str).map(lambda s: img_prefix(s) if s != "nan" else "(missing)")
        ct = pd.crosstab(df["_ip"], df["_lp"])
        print(ct.to_string())

    # 4. What did my current infer_source() do?
    print("\n" + "=" * 70)
    print("CURRENT infer_source() OUTPUT")
    print("=" * 70)
    df["src_current"] = df["image"].map(infer_source)
    print(df["src_current"].value_counts().to_string())

    # 5. Sample IDs from each bucket so we can see actual filenames
    print("\n" + "=" * 70)
    print("SAMPLE IMAGE IDs per current bucket  (10 per group)")
    print("=" * 70)
    for src, g in df.groupby("src_current"):
        sample = g.sample(min(10, len(g)), random_state=0)["image"].tolist()
        print(f"\n[{src}] ({len(g)} images)")
        for s in sample:
            print(f"  {s}")

    # 6. Numeric range of ISIC_ IDs — useful for refining range heuristics
    print("\n" + "=" * 70)
    print("ISIC_xxx NUMERIC RANGE")
    print("=" * 70)
    isic_mask = df["image"].str.startswith("ISIC_")
    if isic_mask.any():
        nums = (
            df.loc[isic_mask, "image"]
            .str.replace("ISIC_", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
            .dropna()
            .astype(int)
        )
        # bucket into 5000-id-wide bins to visualize the distribution
        bins = list(range(0, int(nums.max()) + 5001, 5000))
        binned = pd.cut(nums, bins=bins, right=False)
        print(binned.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()