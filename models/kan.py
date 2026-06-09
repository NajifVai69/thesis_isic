"""Vendored efficient-KAN for ISIC-2019 classification.

Kolmogorov–Arnold Network (KAN) linear layer based on the efficient-KAN
formulation by Liu et al. (2024):
  "KAN: Kolmogorov-Arnold Networks" — arXiv 2404.19756

Each KANLinear replaces a standard nn.Linear with two branches:
  1. Base branch:   w_base · silu(x)          — handles out-of-range values robustly
  2. Spline branch: B-spline basis expansion with learned coefficients

AMP safety: the B-spline basis is always computed in fp32 regardless of the
outer autocast context. The result is cast back to the caller's dtype before
returning so the rest of the network sees a consistent dtype.

Why vendored instead of `pip install efficient_kan`?
  - Keeps the dependency tree minimal (consistent with this codebase's hand-rolled style)
  - Adds the fp32-pinning that the upstream package does not have
  - ~150 lines total
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """One KAN layer: base residual (SiLU + linear) + B-spline correction.

    Args:
        in_features:   Input dimension.
        out_features:  Output dimension.
        grid_size:     Number of B-spline grid intervals. More → more expressive,
                       more parameters. 5 is a good default for a classifier head.
        spline_order:  B-spline polynomial degree. 3 = cubic splines (C² smooth).
        scale_noise:   Std used to initialise the spline weights.
        grid_range:    [lo, hi] range the uniform grid spans.  The grid is extended
                       by `spline_order` cells on each side, so inputs modestly
                       outside this range are still covered.  [-1, 1] works well
                       when the input comes from a LayerNorm-normalised vector.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        grid_size:    int         = 5,
        spline_order: int         = 3,
        scale_noise:  float       = 0.1,
        grid_range:   list[float] = None,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = [-1.0, 1.0]

        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        # Build uniform grid extended by spline_order on each side.
        # Total points: grid_size + 2 * spline_order + 1.
        k   = spline_order
        lo, hi = float(grid_range[0]), float(grid_range[1])
        h   = (hi - lo) / grid_size
        pts = torch.arange(-k, grid_size + k + 1, dtype=torch.float32) * h + lo
        # Shape (in_features, G) — one independent grid per input dimension.
        grid = pts.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)   # always fp32

        n_basis = grid_size + spline_order   # basis functions per input dim

        self.base_weight   = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features * n_basis)
        )

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, 0.0, scale_noise / max(math.sqrt(in_features), 1))

    # ── B-spline basis (always fp32, AMP-safe) ─────────────────────────────

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis functions via the De Boor recursion.

        Always operates in fp32 regardless of the outer autocast context.

        Args:
            x: (B, in_features) — arbitrary values (values near grid_range work best).
        Returns:
            (B, in_features, grid_size + spline_order) fp32 tensor.
        """
        # Force fp32 for numerical stability; grid is already fp32.
        x32  = x.detach().float()                                # (B, in_f)
        grid = self.grid.float()                                 # (in_f, G)
        G    = grid.shape[1]                                     # grid_size + 2k + 1
        k    = self.spline_order

        x32 = x32.unsqueeze(-1)                                  # (B, in_f, 1)

        # Order-0: indicator for each of the G-1 grid intervals.
        bases = ((x32 >= grid[:, :-1]) & (x32 < grid[:, 1:])).float()  # (B, in_f, G-1)

        # De Boor recursion: lift from order 0 to order k.
        # After p iterations we have (G-p-1) basis functions.
        for p in range(1, k + 1):
            n = G - p - 1                         # number of output basis functions
            # Grid slices (all shape (in_f, n)):
            t_i   = grid[:, :n]                   # t_i
            t_ip  = grid[:, p : n + p]            # t_{i+p}
            t_ip1 = grid[:, p + 1 : n + p + 1]   # t_{i+p+1}
            t_i1  = grid[:, 1 : n + 1]            # t_{i+1}

            # Left and right blending factors (shape (B, in_f, n) via broadcasting)
            left  = (x32 - t_i)   / (t_ip  - t_i   + 1e-8)
            right = (t_ip1 - x32) / (t_ip1 - t_i1  + 1e-8)

            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        # Final shape: (B, in_f, grid_size + spline_order)
        return bases.contiguous()

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_features) — arbitrary leading dimensions.
        Returns:
            (..., out_features) in the same dtype as x.
        """
        orig_dtype = x.dtype
        leading    = x.shape[:-1]
        x_flat     = x.reshape(-1, self.in_features)             # (B, in_f)

        # ── Base branch: w_base · silu(x) ──────────────────────────────────
        base_out = F.linear(F.silu(x_flat), self.base_weight)    # (B, out_f)

        # ── Spline branch: always fp32, cast output back ────────────────────
        spline_basis = self._b_splines(x_flat)                   # (B, in_f, n_basis) fp32
        spline_flat  = spline_basis.reshape(x_flat.shape[0], -1) # (B, in_f * n_basis) fp32
        spline_out   = F.linear(
            spline_flat, self.spline_weight.float()
        ).to(orig_dtype)                                          # (B, out_f)

        return (base_out + spline_out).reshape(*leading, self.out_features)


class KAN(nn.Module):
    """Stacked KANLinear layers with inter-layer LayerNorm.

    Args:
        dims:         Width of each layer, e.g. [256, 64, 8] → two KAN layers.
        grid_size:    Shared grid size (default 5).
        spline_order: Shared spline order (default 3 = cubic).
        grid_range:   Grid range passed to every KANLinear.

    Example::
        head = KAN([256, 8])                     # single KAN layer
        head = KAN([256, 64, 8])                 # two KAN layers with hidden 64
    """

    def __init__(
        self,
        dims:         list[int],
        grid_size:    int         = 5,
        spline_order: int         = 3,
        grid_range:   list[float] = None,
    ):
        super().__init__()
        assert len(dims) >= 2, "KAN needs at least [in_dim, out_dim]"
        if grid_range is None:
            grid_range = [-1.0, 1.0]

        self.layers = nn.ModuleList([
            KANLinear(
                dims[i], dims[i + 1],
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
            for i in range(len(dims) - 1)
        ])
        # LayerNorm between consecutive KAN layers (identity after the last)
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i + 1]) if i < len(dims) - 2 else nn.Identity()
            for i in range(len(dims) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = norm(layer(x))
        return x
