"""Class-Balanced Focal Loss (Cui et al., CVPR 2019).

Reference: "Class-Balanced Loss Based on Effective Number of Samples"
           https://arxiv.org/abs/1901.05555

The key insight: instead of raw class frequencies, use the *effective number*
of samples E(n) = (1 - β^n) / (1 - β). As n → ∞, E(n) → 1/(1-β), giving a
soft cap that prevents dominant classes from getting all the weight.

Combined with focal modulation (Lin et al., RetinaNet): down-weight easy
examples so the model focuses on hard, misclassified samples — especially
important for the rare classes (DF, VASC) where individual hard examples
carry outsized learning signal.

Final per-sample loss:
    L = - α_y · (1 - p_y)^γ · CE_smooth(logits, y)

where:
    α_y         = (1 - β) / (1 - β^{n_y})     (unnormalised CB weight)
    CE_smooth   = (1-ε)·NLL_y + ε·H_uniform    (label-smoothed cross-entropy)
    H_uniform   = -mean_c log(p_c)             (entropy of uniform distribution)

Weights are normalised to sum to num_classes before being applied so that the
effective learning rate remains comparable across different β choices.

Tuning notes (validated on ISIC-2019):
    beta=0.9999 → weight ratio NV/DF ≈ 37×  → class collapse in 5–8 epochs
    beta=0.999  → weight ratio NV/DF ≈  7×  → stable training       ← use this
    label_smoothing=0.1 prevents overconfident memorisation of rare classes
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss with optional label smoothing.

    Args:
        class_counts:     (C,) per-class training sample counts.
        beta:             Effective-number β. Use 0.999 for ISIC-2019 (see note above).
        gamma:            Focal exponent. 2.0 is standard.
        label_smoothing:  ε in [0, 1). 0.1 is a safe default. Set 0.0 to disable.
        reduction:        'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        class_counts: np.ndarray | torch.Tensor,
        beta: float = 0.999,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        assert 0.0 <= label_smoothing < 1.0
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

        counts = np.asarray(class_counts, dtype=np.float64)
        if (counts == 0).any():
            raise ValueError("class_counts contains a zero — a class has no training samples.")

        effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * len(counts)   # normalise → sum = C
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw class scores
            targets: (B,)   integer labels in [0, C)
        Returns:
            scalar loss (reduction='mean'/'sum') or (B,) tensor (reduction='none')
        """
        log_probs = F.log_softmax(logits, dim=1)                            # (B, C)
        log_p_t   = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)    # (B,)
        nll       = -log_p_t                                                 # (B,)

        # Label-smoothed NLL: blend hard target with uniform distribution.
        # CE_smooth = (1-ε)·NLL_true + ε·(−mean_c log p_c)
        if self.label_smoothing > 0.0:
            h_uniform = -log_probs.mean(dim=1)                              # (B,)
            smooth_nll = (1.0 - self.label_smoothing) * nll \
                       + self.label_smoothing * h_uniform                   # (B,)
        else:
            smooth_nll = nll

        # Focal modulation — computed from the un-smoothed p_t so the
        # down-weighting is still based on model confidence.
        p_t          = torch.exp(log_p_t)                                   # (B,)
        focal_weight = (1.0 - p_t) ** self.gamma                           # (B,)

        alpha      = self.weights[targets]                                  # (B,)
        per_sample = alpha * focal_weight * smooth_nll                     # (B,)

        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample