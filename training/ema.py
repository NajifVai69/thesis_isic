"""Exponential Moving Average (EMA) of model weights.

Keeps a shadow copy of the model whose parameters are a running average of the
live model's parameters. At inference time, the EMA model typically has 0.5–1%
higher BMA than the live model — essentially a free ensemble of late-training
checkpoints.

Usage:
    ema = ModelEMA(model, decay=0.9998)

    # Inside training loop, after optimizer.step():
    ema.update(model)

    # For validation / checkpointing:
    with ema.apply(model):
        val_metrics = evaluate(model, val_loader)   # model params temporarily replaced
    # After the 'with' block, model reverts to its original parameters.
"""
from __future__ import annotations

import contextlib
from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    """Shadow-copy EMA of a model's parameters and buffers.

    The EMA lives on the same device as the source model.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        """
        Args:
            model: the live training model.
            decay: EMA decay factor. 0.9998 ≈ "average of last 5000 steps"
                   which is appropriate for 60 epochs × ~94 steps/epoch ≈ 5640 steps.
        """
        self.decay = decay
        # Deep-copy the model so the shadow starts identical to the live model.
        # Wrap in no_grad so the copy does not participate in autograd.
        with torch.no_grad():
            self.shadow = deepcopy(model)
        self.shadow.eval()
        # Freeze shadow params — they are updated manually, not by an optimizer.
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow params: shadow_p ← decay * shadow_p + (1-decay) * live_p."""
        d = self.decay
        for s_param, l_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            s_param.copy_(d * s_param + (1.0 - d) * l_param.data)

        # Also sync non-parameter buffers (e.g. BN running stats).
        for s_buf, l_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(l_buf)

    @contextlib.contextmanager
    def apply(self, model: nn.Module):
        """Context manager: temporarily swap model params ↔ EMA params.

        Useful for running validation with EMA weights without needing a
        separate forward pass through the shadow model.

        Example:
            with ema.apply(model):
                run_validation(model)
        """
        # Save current live params
        original = [p.data.clone() for p in model.parameters()]
        # Replace with EMA params
        for m_param, s_param in zip(model.parameters(), self.shadow.parameters()):
            m_param.data.copy_(s_param.data)
        try:
            yield
        finally:
            # Restore live params
            for m_param, orig in zip(model.parameters(), original):
                m_param.data.copy_(orig)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)