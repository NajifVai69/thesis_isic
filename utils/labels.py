"""Label and source-dataset inference from ISIC-2019 metadata.

ISIC-2019 = HAM10000 + BCN_20000 + MSK. We use the lesion_id column as the
primary source signal (it preserves the original dataset prefix: HAM_xxxxx,
BCN_xxxxx, MSK), falling back to image-ID numeric range for the 2,084 images
that have no lesion_id.

Confirmed from the Kaggle mirror (salviohexia/isic-2019-skin-lesion-images):
  - All 25,331 images have ISIC_xxxxxxx image IDs (mirror renamed everything).
  - HAM10000: lesion_id starts with "HAM_"; numeric range 24306–34320.
  - BCN_20000: lesion_id starts with "BCN_"; numeric IDs >= 50000.
  - MSK: lesion_id == "MSK" (819) or no lesion_id with IDs < 24306 (including
    ISIC_0xxxxxxx_downsampled variants from MSK-4).
"""
from __future__ import annotations

import math

import pandas as pd

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_groundtruth(csv_path) -> pd.DataFrame:
    """Return a DataFrame with columns [image, class, label].

    ISIC_2019_Training_GroundTruth.csv is one-hot over [MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK].
    UNK is the out-of-distribution class only used at test time on the challenge
    server, so the labeled training set has zero rows with UNK=1. We assert that.
    """
    df = pd.read_csv(csv_path)
    expected = {"image"} | set(CLASSES) | {"UNK"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Groundtruth CSV missing columns: {missing}")

    if (df["UNK"] == 1).any():
        n_unk = int((df["UNK"] == 1).sum())
        raise ValueError(
            f"Found {n_unk} rows with UNK=1 in the training groundtruth. "
            f"The labeled training set should have none — check your CSV."
        )

    onehot = df[CLASSES].to_numpy()
    if not ((onehot.sum(axis=1) == 1).all()):
        bad = int((onehot.sum(axis=1) != 1).sum())
        raise ValueError(f"{bad} rows are not exactly one-hot across the 8 classes.")

    out = pd.DataFrame({
        "image": df["image"].astype(str),
        "label": onehot.argmax(axis=1),
    })
    out["class"] = out["label"].map(lambda i: CLASSES[i])
    return out


def infer_source(image_id: str, lesion_id=None) -> str:
    """Map an ISIC-2019 image to its source dataset.

    Args:
        image_id:  e.g. 'ISIC_0034320' or 'ISIC_0014157_downsampled'
        lesion_id: from the metadata CSV, e.g. 'HAM_0000118', 'BCN_0000004',
                   'MSK', or NaN/None when absent.

    Returns:
        'HAM' | 'BCN' | 'MSK'

    Strategy:
        1. lesion_id prefix is authoritative when present.
        2. Fallback to image-id numeric range for the 2,084 no-lesion-id images:
             - 24306 <= num <= 34320   → HAM
             - num >= 50000            → BCN
             - otherwise (low IDs + _downsampled MSK-4 images) → MSK
    """
    # --- Primary: lesion_id ---
    if lesion_id is not None:
        if not (isinstance(lesion_id, float) and math.isnan(lesion_id)):
            lid = str(lesion_id).strip()
            if lid.startswith("HAM_"):
                return "HAM"
            if lid.startswith("BCN_"):
                return "BCN"
            if lid.upper().startswith("MSK"):
                return "MSK"

    # --- Fallback: image_id numeric heuristic ---
    if not image_id.startswith("ISIC_"):
        return "MSK"  # unexpected — treat conservatively
    # Strip "ISIC_" then take only digits before any further underscore
    # e.g. "0014157_downsampled" → "0014157"
    tail = image_id[5:]  # drop "ISIC_"
    numeric_str = tail.split("_")[0]
    try:
        num = int(numeric_str)
    except ValueError:
        return "MSK"
    if 24306 <= num <= 34320:
        return "HAM"
    if num >= 50000:
        return "BCN"
    return "MSK"


def load_metadata(csv_path) -> pd.DataFrame:
    """Return a DataFrame with columns
       [image, age_approx, sex, anatom_site_general, lesion_id, source].

    source is inferred from lesion_id first, numeric image-id range as fallback.
    Missing age/sex/site values are kept as NaN — handled by a learned missing
    token in the metadata cross-attention head.
    """
    df = pd.read_csv(csv_path)
    df["image"] = df["image"].astype(str)

    if "lesion_id" in df.columns:
        df["source"] = [
            infer_source(img, lid)
            for img, lid in zip(df["image"], df["lesion_id"])
        ]
    else:
        df["source"] = df["image"].map(lambda x: infer_source(x))

    keep = ["image", "age_approx", "sex", "anatom_site_general", "lesion_id", "source"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()