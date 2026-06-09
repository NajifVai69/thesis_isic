"""Shades-of-Gray color constancy (Finlayson & Trezzi, 2004).

Standard preprocessing for dermoscopy classification — corrects for scanner /
illuminant color casts that differ across HAM10000, BCN_20000, and MSK. We use
the Minkowski p-norm form with p=6 by default, which is what most ISIC papers
report.
"""
from __future__ import annotations

import numpy as np


def shades_of_gray(img_rgb_uint8: np.ndarray, p: int = 6) -> np.ndarray:
    """Apply Shades-of-Gray color constancy.

    Args:
        img_rgb_uint8: (H, W, 3) uint8 RGB image.
        p:             Minkowski norm order. p=1 is grey-world, p=inf is max-RGB,
                       p=6 is the canonical Shades-of-Gray.

    Returns:
        (H, W, 3) uint8 RGB image, corrected.
    """
    if img_rgb_uint8.dtype != np.uint8 or img_rgb_uint8.ndim != 3 or img_rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3) uint8 RGB, got dtype={img_rgb_uint8.dtype} shape={img_rgb_uint8.shape}")

    img = img_rgb_uint8.astype(np.float32)
    # Per-channel Minkowski norm of the pixel intensities.
    # norm_c = ( mean_pixels( I_c ** p ) ) ** (1/p)
    norm = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p)  # shape (3,)
    gray = float(norm.mean())
    scale = gray / (norm + 1e-8)
    out = img * scale[None, None, :]
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)
