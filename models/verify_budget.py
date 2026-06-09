"""Verify model budgets before committing to training runs.

Run on the GPU machine (where timm + torch are installed):

    python -m models.verify_budget

Prints, for every variant in HYBRID_VARIANTS and DEKAN_VARIANTS:
    - total / pretrained-stem / from-scratch parameter counts
    - output shape on a (2, 3, 224, 224) input
    - GMAC count via fvcore (skipped with a note if fvcore is missing)
    - the AdamW param groups that train.py will use

Budget targets:
    hybrid_full / hybrid_* variants  : < 6 M params, < 1 GMAC  (lightweight flagship)
    dekan_full  / dekan_*  variants  : 10–15 M params           (accuracy-oriented tier)

Use this to confirm the numbers and tune trunk hyperparameters before training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dekan import DEKAN_VARIANTS, create_dekan
from models.hybrid import HYBRID_VARIANTS, create_hybrid


def _fmt(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:5.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:5.1f} K"
    return f"{n:>6d}"


def _count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _fake_meta(B: int, device: str = "cpu") -> dict:
    """Realistic metadata batch with some missing values."""
    return {
        "age":         torch.tensor([0.5, -0.2, 1.1, 0.0][:B], device=device),
        "age_missing": torch.tensor([0,    1,   0,   0][:B], device=device),
        "sex_idx":     torch.tensor([0,   -1,   1,   0][:B], device=device),
        "site_idx":    torch.tensor([3,   -1,   5,   7][:B], device=device),
    }


def _try_fvcore_macs(model: torch.nn.Module, sample_args: tuple) -> str:
    """Compute MACs using fvcore. Returns a formatted string or skip note."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return "fvcore not installed — skip (pip install fvcore)"
    model.eval()
    try:
        flops = FlopCountAnalysis(model, sample_args)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
        return f"{total / 1e9:5.3f} GMAC"
    except Exception as e:
        return f"fvcore failed ({type(e).__name__}: {e})"


def _stem_param_count(model: torch.nn.Module) -> int:
    """Count parameters in CNN stem(s), handling single and dual backbone models."""
    n = 0
    # Hybrid: single stem attribute
    if hasattr(model, "stem"):
        n += _count(model.stem)
    # DEKAN: two optional stems
    if hasattr(model, "densenet_stem"):
        n += _count(model.densenet_stem)
    if hasattr(model, "effnet_stem"):
        n += _count(model.effnet_stem)
    return n


def _print_table(
    variants: list[str],
    factory,
    x: torch.Tensor,
    meta: dict,
    device: str,
    needs_meta_fn,
    budget_note: str,
) -> None:
    W = 100
    print("=" * W)
    print(f"{'variant':22s}  {'total':>9s}  {'stems':>9s}  {'scratch':>9s}  "
          f"{'GMAC':>11s}  {'out shape':>12s}")
    print("-" * W)

    for name in sorted(variants):
        model = factory(name, pretrained=False).to(device)
        model.eval()

        n_total = _count(model)
        n_stems = _stem_param_count(model)
        n_scr   = n_total - n_stems

        use_meta = needs_meta_fn(name)
        with torch.no_grad():
            if use_meta:
                y = model(x, meta=meta)
                sample_args = (x, meta)
            else:
                y = model(x)
                sample_args = (x,)
        out_shape = tuple(y.shape)
        macs = _try_fvcore_macs(model, sample_args)

        pg = model.get_param_groups(3e-4, 0.3)
        pg_desc = ", ".join(
            f"{g.get('name', '?')}="
            f"{sum(p.numel() for p in g['params']) / 1e6:.2f}M"
            f"@{g['lr']:.0e}"
            for g in pg
        )

        print(f"{name:22s}  {_fmt(n_total):>9s}  {_fmt(n_stems):>9s}  {_fmt(n_scr):>9s}  "
              f"{macs:>11s}  {str(out_shape):>12s}")
        print(f"{'':22s}  groups: {pg_desc}")

    print()
    print(budget_note)


def main():
    from models.baseline import needs_metadata

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, H, W = 2, 3, 224, 224
    x    = torch.randn(B, C, H, W, device=device)
    meta = _fake_meta(B, device=device)

    print(f"\nDevice: {device}  |  Input: {tuple(x.shape)}\n")

    print("-- Lightweight hybrid variants (< 6M / < 1 GMAC target) " + "-" * 32)
    _print_table(
        sorted(HYBRID_VARIANTS), create_hybrid, x, meta, device,
        lambda name: name == "hybrid_full",
        "Target: < 6.00 M params  |  < 1.000 GMAC  (hybrid_full is the headline model)",
    )

    print()
    print("-- DEKAN flagship variants (~10-15 M target) " + "-" * 46)
    _print_table(
        sorted(DEKAN_VARIANTS), create_dekan, x, meta, device,
        needs_metadata,
        "Target: 10–15 M params  (separate accuracy-oriented tier; dekan_full is the headline model)",
    )


if __name__ == "__main__":
    main()