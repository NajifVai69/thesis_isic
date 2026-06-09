"""Training utilities: seeding, checkpointing, logging, parameter counting."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility across Python / NumPy / PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN determinism — slight speed cost but required for exact reproduction.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    ema_state_dict: dict | None,
    optimizer_state: dict,
    scheduler_state: dict,
    scaler_state: dict,
    metrics: dict,
) -> None:
    """Save a complete checkpoint so training can be resumed or evaluated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":          epoch,
        "model":          model.state_dict(),
        "ema":            ema_state_dict,
        "optimizer":      optimizer_state,
        "scheduler":      scheduler_state,
        "scaler":         scaler_state,
        "metrics":        {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
    }, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    ema=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device: str = "cpu",
) -> dict:
    """Load checkpoint into model (and optionally optimiser / scheduler).

    Returns the metrics dict from the saved checkpoint.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("metrics", {})


# ── Logging ───────────────────────────────────────────────────────────────────

class CSVLogger:
    """Append-mode CSV logger for per-epoch training metrics."""

    def __init__(self, path: Path, columns: list[str]):
        self.path = path
        self.columns = columns
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(",".join(columns) + "\n", encoding="utf-8")

    def log(self, values: dict) -> None:
        row = ",".join(str(values.get(c, "")) for c in self.columns)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(row + "\n")


# ── Parameter counting ────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict[str, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ── Result directory helper ───────────────────────────────────────────────────

def result_dir(model_name: str, seed: int, base: str = "results") -> Path:
    """Canonical path:  results/{model_name}/seed{seed}/"""
    return Path(base) / model_name / f"seed{seed}"


def save_metrics_json(path: Path, metrics: dict) -> None:
    """Save scalar metrics to JSON, skipping non-serialisable values."""
    out = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")