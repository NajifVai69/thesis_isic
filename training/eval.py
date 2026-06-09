"""Evaluation script: test-set metrics, TTA, confusion matrix, loss curves.

Usage:
    # Evaluate best checkpoint on test set
    python -m training.eval --model resnet18 --seed 42

    # Evaluate without TTA
    python -m training.eval --model resnet18 --seed 42 --no_tta

Outputs written to results/{model_name}/seed{seed}/eval/:
    test_metrics.json       scalar metrics (BMA, accuracy, macro-F1, macro-AUC)
    test_metrics_tta.json   same with test-time augmentation
    confusion_matrix.png    normalized confusion matrix heatmap
    per_class_metrics.csv   per-class recall / precision / F1
    curves.png              train + val loss and BMA over epochs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isic_data.isic_memmap import ISICDataset, build_dataloaders
from models.baseline import create_model
from models.baseline import needs_metadata
from training.ema import ModelEMA
from training.metrics import ISICMetrics, format_results
from training.utils import count_parameters, load_checkpoint, result_dir, set_seed
from utils.labels import CLASSES

NUM_CLASSES = len(CLASSES)


# ── TTA helpers ───────────────────────────────────────────────────────────────

def _tta_views(imgs: torch.Tensor) -> torch.Tensor:
    """Generate 8 geometric views of a (B, C, H, W) batch.

    Views: original, hflip, vflip, hflip+vflip,
           rot90, rot90+hflip, rot270, rot270+hflip.

    Returns (B*8, C, H, W) — all 8 views stacked so a single forward pass
    handles the whole batch.
    """
    views = [
        imgs,
        imgs.flip(-1),                              # hflip
        imgs.flip(-2),                              # vflip
        imgs.flip(-1).flip(-2),                     # hflip + vflip
        imgs.rot90(1, dims=(-2, -1)),               # rot 90
        imgs.rot90(1, dims=(-2, -1)).flip(-1),      # rot 90 + hflip
        imgs.rot90(2, dims=(-2, -1)),               # rot 180
        imgs.rot90(3, dims=(-2, -1)),               # rot 270
    ]
    return torch.cat(views, dim=0)                  # (B*8, C, H, W)


@torch.no_grad()
def predict_tta(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    n_views: int = 8,
    use_amp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TTA inference over a DataLoader.

    Returns:
        probs:   (N, C) averaged softmax probabilities
        targets: (N,)  ground-truth labels
    """
    model.eval()
    all_probs   = []
    all_targets = []

    for batch in loader:
        if len(batch) == 3:
            imgs, labels, meta = batch
            meta = {k: v.to(device, non_blocking=True) for k, v in meta.items()}
        else:
            imgs, labels = batch
            meta = None
        imgs   = imgs.to(device, non_blocking=True)
        imgs   = imgs.to(memory_format=torch.channels_last)
        B      = imgs.shape[0]
        views  = _tta_views(imgs)                          # (B*8, C, H, W)
        # Replicate metadata along the batch dim to match the 8 stacked views.
        # `_tta_views` stacks as [view0_B, view1_B, …, view7_B], so we tile
        # the metadata by 8: meta[i] aligns with sample i in every view block.
        if meta is not None:
            meta_v = {k: v.repeat(8, *([1] * max(v.ndim - 1, 0))) for k, v in meta.items()}
        else:
            meta_v = None

        with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(views, meta=meta_v) if meta_v is not None else model(views)  # (B*8, C)

        probs = F.softmax(logits, dim=1)                   # (B*8, C)
        probs = probs.view(n_views, B, NUM_CLASSES).mean(0)  # (B, C)  ← average views

        # Correction for wrong stacking order above: we stacked views along
        # dim=0 (not dim=1), so reshape is (n_views, B, C), not (B, n_views, C).
        # The .view(n_views, B, ...) above is correct.

        all_probs.append(probs.cpu())
        all_targets.append(labels)

    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


@torch.no_grad()
def predict_standard(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard single-pass inference. Returns (probs, targets)."""
    model.eval()
    all_probs   = []
    all_targets = []

    for batch in loader:
        if len(batch) == 3:
            imgs, labels, meta = batch
            meta = {k: v.to(device, non_blocking=True) for k, v in meta.items()}
        else:
            imgs, labels = batch
            meta = None
        imgs = imgs.to(device, non_blocking=True)
        imgs = imgs.to(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(imgs, meta=meta) if meta is not None else model(imgs)
        all_probs.append(F.softmax(logits, dim=1).cpu())
        all_targets.append(labels)

    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_confusion_matrix(conf: np.ndarray, save_path: Path) -> None:
    """Plot a normalised confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        conf,
        annot=True, fmt=".2f",
        xticklabels=CLASSES, yticklabels=CLASSES,
        cmap="Blues", vmin=0.0, vmax=1.0,
        ax=ax, cbar=True,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (normalised by true class)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[ok]   saved {save_path}")


def plot_curves(log_csv: Path, save_path: Path) -> None:
    """Plot train / val loss and BMA curves from the training log."""
    if not log_csv.exists():
        print(f"[warn] log CSV not found: {log_csv}")
        return

    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"],   label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss"); axes[0].legend()

    # BMA curve
    axes[1].plot(df["epoch"], df["train_bma"], label="train")
    axes[1].plot(df["epoch"], df["val_bma"],   label="val")
    best_epoch = df.loc[df["val_bma"].idxmax(), "epoch"]
    best_bma   = df["val_bma"].max()
    axes[1].axvline(best_epoch, color="red", linestyle="--", alpha=0.5,
                    label=f"best val BMA={best_bma:.4f}")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("BMA")
    axes[1].set_title("Balanced Multi-class Accuracy"); axes[1].legend()

    plt.suptitle(f"{save_path.parents[1].name}  seed={save_path.parents[0].name}")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[ok]   saved {save_path}")


# ── Metric computation from probability predictions ───────────────────────────

def compute_metrics_from_probs(probs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Wraps ISICMetrics for use with pre-computed probabilities."""
    m = ISICMetrics()
    # ISICMetrics.update() accepts logits; raw probs work too since it applies
    # softmax internally — but probs already summing to 1 means softmax is
    # near-identity (no-op numerically).  Pass as-is.
    m.update(probs, targets)
    return m.compute()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",        type=str,  required=True)
    ap.add_argument("--seed",         type=int,  default=42)
    ap.add_argument("--config",       type=str,  default="configs/default.yaml")
    ap.add_argument("--train_config", type=str,  default="configs/training.yaml")
    ap.add_argument("--checkpoint",   type=str,  default=None,
                    help="Override path to checkpoint. Default: best.pth in result dir.")
    ap.add_argument("--no_tta",       action="store_true",
                    help="Skip test-time augmentation.")
    ap.add_argument("--batch_size",   type=int,  default=64,
                    help="Batch size for inference (TTA uses 8× this).")
    return ap.parse_args()


def main():
    args = parse_args()

    with open(args.train_config, encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    model_name = args.model
    seed       = args.seed
    use_amp    = bool(train_cfg["mixed_precision"])
    use_tta    = not args.no_tta

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}  model={model_name}  seed={seed}  TTA={use_tta}")

    rdir     = result_dir(model_name, seed)
    eval_dir = rdir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model = create_model(model_name, pretrained=False)
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)

    ema = ModelEMA(model, decay=0.9998)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else rdir / "checkpoints" / "best.pth"
    print(f"[info] loading checkpoint {ckpt_path}")
    load_checkpoint(ckpt_path, model, ema=ema, device=str(device))

    # Use EMA weights for evaluation (better generalisation)
    with ema.apply(model):
        _eval_with_model(
            model, args, train_cfg, device, eval_dir, use_tta, use_amp, rdir
        )


def _eval_with_model(model, args, train_cfg, device, eval_dir, use_tta, use_amp, rdir):
    # ── Test DataLoader ──────────────────────────────────────────────────
    _, _, test_ds = ISICDataset.create_splits(
        config_path=args.config, use_metadata=needs_metadata(args.model),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    print(f"[info] test set: {len(test_ds)} images")

    # ── Standard evaluation ──────────────────────────────────────────────
    print("[info] standard evaluation …")
    probs, targets = predict_standard(model, test_loader, device, use_amp)
    results = compute_metrics_from_probs(probs, targets)
    print("\nStandard eval:")
    print(format_results(results))

    out = {k: v for k, v in results.items() if isinstance(v, float)}
    (eval_dir / "test_metrics.json").write_text(json.dumps(out, indent=2))
    results["per_class"].to_csv(eval_dir / "per_class_metrics.csv", index=False)
    np.save(eval_dir / "conf_matrix.npy", results["conf_matrix"])        # raw array for figures.py
    plot_confusion_matrix(results["conf_matrix"], eval_dir / "confusion_matrix.png")

    # ── TTA evaluation ────────────────────────────────────────────────────
    if use_tta:
        print("\n[info] TTA evaluation …")
        probs_tta, _ = predict_tta(
            model, test_loader, device,
            n_views=int(train_cfg["tta"]["n_views"]),
            use_amp=use_amp,
        )
        results_tta = compute_metrics_from_probs(probs_tta, targets)
        print("\nTTA eval:")
        print(format_results(results_tta))

        out_tta = {k: v for k, v in results_tta.items() if isinstance(v, float)}
        (eval_dir / "test_metrics_tta.json").write_text(json.dumps(out_tta, indent=2))
        results_tta["per_class"].to_csv(eval_dir / "per_class_metrics_tta.csv", index=False)
        np.save(eval_dir / "conf_matrix_tta.npy", results_tta["conf_matrix"])
        plot_confusion_matrix(
            results_tta["conf_matrix"], eval_dir / "confusion_matrix_tta.png"
        )

    # ── Training curves ───────────────────────────────────────────────────
    plot_curves(rdir / "logs" / "train_log.csv", eval_dir / "curves.png")

    print(f"\n[done] eval outputs in {eval_dir.resolve()}")


if __name__ == "__main__":
    main()