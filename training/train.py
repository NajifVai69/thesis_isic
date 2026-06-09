"""Main training script.

Usage:
    python -m training.train --model resnet18 --seed 42
    python -m training.train --model mobilenetv2_100 --seed 42 --batch_size 256
    python -m training.train --model mobilevit_s --batch_size 128 --grad_accum_steps 2

Results are written to results/{model_name}/seed{seed}/:
    checkpoints/best.pth    (best val-BMA checkpoint)
    checkpoints/last.pth
    logs/train_log.csv      (epoch, train_loss, val_loss, train_bma, val_bma, lr)
    metrics_best_val.json   (scalar metrics at best-val checkpoint)
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isic_data.isic_memmap import ISICDataset, build_dataloaders
from losses.cb_focal import ClassBalancedFocalLoss
from models.baseline import create_model
from models.baseline import needs_metadata
from training.ema import ModelEMA
from training.metrics import ISICMetrics, format_results
from training.utils import (
    CSVLogger,
    count_parameters,
    load_checkpoint,
    result_dir,
    save_checkpoint,
    save_metrics_json,
    set_seed,
)


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Train an ISIC-2019 classification model.")
    ap.add_argument("--model",            type=str,   default=None)
    ap.add_argument("--seed",             type=int,   default=None)
    ap.add_argument("--epochs",           type=int,   default=None)
    ap.add_argument("--batch_size",       type=int,   default=None)
    ap.add_argument("--grad_accum_steps", type=int,   default=None)
    ap.add_argument("--lr",               type=float, default=None)
    ap.add_argument("--config",           type=str,   default="configs/default.yaml")
    ap.add_argument("--train_config",     type=str,   default="configs/training.yaml")
    ap.add_argument("--resume",           type=str,   default=None,
                    help="Path to checkpoint to resume from.")
    return ap.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _merge_args(cfg: dict, args) -> dict:
    """CLI args override yaml config values."""
    t = cfg["training"]
    if args.model        is not None: cfg["model"]              = args.model
    if args.seed         is not None: cfg["seed"]               = args.seed
    if args.epochs       is not None: t["epochs"]               = args.epochs
    if args.batch_size   is not None: t["batch_size"]           = args.batch_size
    if args.grad_accum_steps is not None: t["grad_accum_steps"] = args.grad_accum_steps
    if args.lr           is not None: cfg["optimizer"]["lr"]    = args.lr
    return cfg


def _run_epoch(
    model:        nn.Module,
    loader,
    criterion:    nn.Module,
    optimizer,
    scaler,
    device:       torch.device,
    grad_accum:   int,
    ema:          ModelEMA | None,
    train:        bool,
    metrics:      ISICMetrics,
    mixup_alpha:  float = 0.0,
    mixup_prob:   float = 0.5,
) -> float:
    """One full pass over a DataLoader.  Returns mean loss.

    Mixup (Zhang et al., ICLR 2018):
      - Applied only during training, to a random fraction (mixup_prob) of batches.
      - lam ~ Beta(alpha, alpha); mixed_x = lam*x_i + (1-lam)*x_j.
      - Loss = lam*CE(y_i) + (1-lam)*CE(y_j) — both terms use CB-Focal weights.
      - Metrics are computed on the *original* labels (not mixed) so BMA reflects
        true class coverage rather than soft-target accuracy.
      - We intentionally do NOT apply mixup at gamma-weighted focal level; the
        focal weight is computed per-term inside the criterion, which handles it.
    """
    model.train(train)
    total_loss = 0.0
    n_batches  = 0
    if train and optimizer is not None:
        optimizer.zero_grad()

    no_grad_ctx = torch.no_grad() if not train else contextlib.nullcontext()
    for step, batch in enumerate(loader):
        # Dataset returns (img, label) without metadata and (img, label, meta)
        # with metadata. Hybrid_full needs the meta dict; baselines never do.
        if len(batch) == 3:
            imgs, labels, meta = batch
            meta = {k: v.to(device, non_blocking=True) for k, v in meta.items()}
        else:
            imgs, labels = batch
            meta = None

        # ── Move to device + channels-last layout ──────────────────────────
        imgs   = imgs.to(device, non_blocking=True)
        imgs   = imgs.to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        # ── Mixup (training only) ───────────────────────────────────────────
        do_mixup   = train and mixup_alpha > 0.0 and np.random.rand() < mixup_prob
        labels_b   = None
        lam        = 1.0
        if do_mixup:
            lam    = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx    = torch.randperm(imgs.size(0), device=device)
            imgs   = lam * imgs + (1.0 - lam) * imgs[idx]
            labels_b = labels[idx]
            # Note: metadata is NOT mixed — using sample i's metadata with a
            # blended image is acceptable since metadata is a weak side-channel.

        # ── Forward + loss ─────────────────────────────────────────────────
        with no_grad_ctx, torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
            logits = model(imgs, meta=meta) if meta is not None else model(imgs)
            if do_mixup:
                loss = lam * criterion(logits, labels) \
                     + (1.0 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)
            if grad_accum > 1:
                loss = loss / grad_accum

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

        total_loss += loss.item() * (grad_accum if grad_accum > 1 else 1)
        n_batches  += 1

        # Metrics — always use original labels (not mixed) so BMA is meaningful
        with torch.no_grad():
            metrics.update(logits, labels)

    return total_loss / max(n_batches, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load configs ────────────────────────────────────────────────────────
    with open(args.train_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _merge_args(cfg, args)

    model_name  = cfg["model"]
    seed        = int(cfg["seed"])
    epochs      = int(cfg["training"]["epochs"])
    batch_size  = int(cfg["training"]["batch_size"])
    grad_accum  = int(cfg["training"]["grad_accum_steps"])
    lr          = float(cfg["optimizer"]["lr"])
    wd          = float(cfg["optimizer"]["weight_decay"])
    warmup_ep   = int(cfg["scheduler"]["warmup_epochs"])
    min_lr      = float(cfg["scheduler"]["min_lr"])
    ema_decay   = float(cfg["ema"]["decay"])
    use_ema     = bool(cfg["ema"]["enabled"])
    use_amp     = bool(cfg["mixed_precision"])
    use_cl      = bool(cfg["channels_last"])
    use_compile = bool(cfg["compile"])
    beta             = float(cfg["loss"]["beta"])
    gamma            = float(cfg["loss"]["gamma"])
    label_smoothing  = float(cfg["loss"].get("label_smoothing", 0.0))
    backbone_lr_scale = float(cfg["optimizer"].get("backbone_lr_scale", 1.0))
    use_sampler  = bool(cfg.get("sampler", {}).get("weighted", False))
    mixup_alpha  = float(cfg.get("mixup", {}).get("alpha", 0.0))
    mixup_prob   = float(cfg.get("mixup", {}).get("prob", 0.5))
    use_mixup    = bool(cfg.get("mixup", {}).get("enabled", False)) and mixup_alpha > 0.0

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}  |  model: {model_name}  |  seed: {seed}")

    # ── Result directory ────────────────────────────────────────────────────
    rdir = result_dir(model_name, seed)
    (rdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (rdir / "logs").mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(
        rdir / "logs" / "train_log.csv",
        ["epoch", "train_loss", "val_loss", "train_bma", "val_bma", "lr"],
    )
    # Save the merged config for reproducibility
    (rdir / "config.yaml").write_text(yaml.dump(cfg), encoding="utf-8")

    # ── Datasets ────────────────────────────────────────────────────────────
    use_meta = needs_metadata(model_name)
    if use_meta:
        print(f"[info] {model_name!r} consumes metadata — enabling meta channel in dataset")

    print("[info] loading datasets …")
    train_ds, val_ds, _ = ISICDataset.create_splits(
        config_path=args.config, use_metadata=use_meta
    )
    train_loader, val_loader, _ = build_dataloaders(
        train_ds, val_ds, val_ds,     # test_ds unused during training
        batch_size=batch_size,
        use_weighted_sampler=use_sampler,
    )
    if use_sampler:
        print("[info] using WeightedRandomSampler (1/√n class weights)")
    print(
        f"[info] train={len(train_ds)}  val={len(val_ds)}  "
        f"batches/epoch={len(train_loader)}"
    )

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"[info] creating model {model_name!r} …")
    model = create_model(model_name, pretrained=True)
    model = model.to(device)
    if use_cl:
        model = model.to(memory_format=torch.channels_last)
    if use_compile:
        print("[info] torch.compile(model) …")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[warn] torch.compile failed ({e}), continuing without.")

    params = count_parameters(model)
    print(f"[info] params: {params['total']:,} total  {params['trainable']:,} trainable")

    # ── Loss ────────────────────────────────────────────────────────────────
    criterion = ClassBalancedFocalLoss(
        class_counts=train_ds.class_counts,
        beta=beta,
        gamma=gamma,
        label_smoothing=label_smoothing,
    ).to(device)
    print(
        f"[info] CB-Focal β={beta}  γ={gamma}  "
        f"class weights: {criterion.weights.cpu().numpy().round(3)}"
    )
    if use_mixup:
        print(f"[info] Mixup enabled: α={mixup_alpha}  prob={mixup_prob}")

    # ── Optimiser — differential LR ─────────────────────────────────────────
    # Pretrained parts get lr * backbone_lr_scale; from-scratch parts get lr.
    # Hybrid models implement a model-specific `get_param_groups()` that knows
    # which parts are pretrained (just the CNN stem); baselines fall back to
    # splitting on `get_classifier()` (everything except the final FC is
    # pretrained for timm-loaded baselines).
    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(lr, backbone_lr_scale)
        names = [g.get("name", "?") for g in param_groups]
        lrs   = [f"{g['lr']:.2e}" for g in param_groups]
        print(f"[info] param groups: {dict(zip(names, lrs))}")
    else:
        try:
            head_module = model.get_classifier()
            head_ids    = {id(p) for p in head_module.parameters()}
            backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
            head_params     = list(head_module.parameters())
            param_groups = [
                {"params": backbone_params, "lr": lr * backbone_lr_scale, "name": "backbone"},
                {"params": head_params,     "lr": lr,                     "name": "head"},
            ]
            print(f"[info] differential LR: backbone={lr*backbone_lr_scale:.2e}  head={lr:.2e}")
        except Exception:
            # Last-resort: single LR if model exposes neither hook.
            param_groups = model.parameters()
            print(f"[warn] no differential-LR hook — using single LR {lr:.2e}")

    optimizer = torch.optim.AdamW(
        param_groups, lr=lr, weight_decay=wd,
        betas=tuple(cfg["optimizer"]["betas"]),
    )

    # ── Scheduler: linear warmup → cosine decay ─────────────────────────────
    warmup_sched  = LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_ep
    )
    cosine_sched  = CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_ep, 1), eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_ep],
    )

    # ── AMP ─────────────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler(device="cuda") if use_amp and device.type == "cuda" else None

    # ── EMA ─────────────────────────────────────────────────────────────────
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_bma = 0.0
    if args.resume:
        print(f"[info] resuming from {args.resume}")
        saved_metrics = load_checkpoint(
            Path(args.resume), model, ema, optimizer, scheduler, scaler, device=str(device)
        )
        start_epoch  = saved_metrics.get("epoch", 0) + 1
        best_val_bma = saved_metrics.get("val_bma", 0.0)

    # ── Training loop ────────────────────────────────────────────────────────
    train_metrics = ISICMetrics()
    val_metrics   = ISICMetrics()

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # -- Train --
        train_metrics.reset()
        train_loss = _run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, grad_accum, ema, train=True, metrics=train_metrics,
            mixup_alpha=mixup_alpha if use_mixup else 0.0,
            mixup_prob=mixup_prob,
        )
        train_res = train_metrics.compute()

        # -- Validate (with EMA weights if available) --
        val_metrics.reset()
        if ema is not None:
            with ema.apply(model):
                val_loss = _run_epoch(
                    model, val_loader, criterion, None, None,
                    device, 1, None, train=False, metrics=val_metrics,
                )
        else:
            val_loss = _run_epoch(
                model, val_loader, criterion, None, None,
                device, 1, None, train=False, metrics=val_metrics,
            )
        val_res = val_metrics.compute()

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"epoch {epoch+1:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_bma={train_res['bma']:.4f}  val_bma={val_res['bma']:.4f}  "
            f"lr={current_lr:.2e}  {elapsed:.0f}s"
        )
        logger.log({
            "epoch":     epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "train_bma":  round(train_res["bma"], 6),
            "val_bma":    round(val_res["bma"], 6),
            "lr":         round(current_lr, 8),
        })

        # -- Checkpoint --
        ckpt_kwargs = dict(
            model=model,
            ema_state_dict=ema.state_dict() if ema else None,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict() if scaler else {},
        )
        # Always save last
        save_checkpoint(
            rdir / "checkpoints" / "last.pth",
            epoch=epoch, metrics={"epoch": epoch, "val_bma": val_res["bma"]},
            **ckpt_kwargs,
        )
        # Save best by val BMA
        if val_res["bma"] > best_val_bma:
            best_val_bma = val_res["bma"]
            save_checkpoint(
                rdir / "checkpoints" / "best.pth",
                epoch=epoch, metrics={"epoch": epoch, "val_bma": val_res["bma"]},
                **ckpt_kwargs,
            )
            save_metrics_json(rdir / "metrics_best_val.json", {
                **{f"val_{k}": v for k, v in val_res.items() if isinstance(v, float)},
                "epoch": epoch + 1,
            })
            print(f"  ✓ new best val_bma={best_val_bma:.4f}  saved best.pth")

    print(f"\n[done] best val BMA = {best_val_bma:.4f}")
    print(f"       results in {rdir.resolve()}")


if __name__ == "__main__":
    main()