"""Publication-quality figures for the ISIC-2019 hybrid CNN-ViT paper.

Reads pre-computed eval outputs from results/{model}/seed{seed}/eval/ and
generates all figures needed for the paper.  Run after eval.py completes.

Usage:
    # Generate all figures (uses seed 42 by default)
    python -m analysis.figures

    # Specify seed
    python -m analysis.figures --seed 42

Outputs written to figures/:
    01_training_curves.pdf/png      loss + BMA over epochs (hybrid_full)
    02_model_comparison.pdf/png     grouped bar chart: BMA, F1, AUC
    03_confusion_matrix.pdf/png     publication-quality confusion matrix (TTA)
    04_per_class_recall.pdf/png     per-class recall: ours vs best baseline
    05_efficiency_scatter.pdf/png   params vs BMA (efficiency claim)
    06_ablation.pdf/png             ablation bar chart
    07_dekan_ablation.pdf/png       DEKAN ablation bar chart
    08_confusion_matrix_dekan.pdf/png  publication-quality confusion matrix for DEKAN (TTA)
    09_training_curves_dekan.pdf/png  loss + BMA over epochs (dekan_full)
    summary_table.csv               full results table for LaTeX

Run `python -m training.eval --model <name> --seed 42` for each model first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.labels import CLASSES

# ── Paths & constants ─────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

# Display names in figures / table
MODEL_LABELS: dict[str, str] = {
    "resnet18":            "ResNet-18",
    "mobilenetv2_100":     "MobileNetV2",
    "efficientnet_b0":     "EfficientNet-B0",
    "mobilevit_s":         "MobileViT-S",
    "efficientformer_l1":  "EfficientFormer-L1",
    "hybrid_full":         "Hybrid (CNN+ViT+Meta)",
    "hybrid_no_meta":      "Hybrid w/o Metadata",
    "hybrid_cnn_only":     "CNN Only",
    "hybrid_vit_only":     "ViT Only",
    "dekan_full":          "DEKAN (Dual+KAN+Meta)",
    "dekan_no_meta":       "DEKAN w/o Metadata",
    "dekan_linear":        "DEKAN w/ Linear Head",
    "dekan_densenet_only": "DEKAN DenseNet Only",
    "dekan_effnet_only":   "DEKAN EffNet Only",
}

# Ordered list for the comparison figure (baselines first, then proposed)
BASELINE_MODELS = [
    "resnet18", "mobilenetv2_100", "efficientnet_b0",
    "mobilevit_s", "efficientformer_l1",
]
PROPOSED_MODEL       = "hybrid_full"
DEKAN_MODEL          = "dekan_full"
ABLATION_MODELS      = ["hybrid_cnn_only", "hybrid_vit_only", "hybrid_no_meta", "hybrid_full"]
DEKAN_ABLATION_MODELS = [
    "dekan_densenet_only", "dekan_effnet_only",
    "dekan_no_meta", "dekan_linear", "dekan_full",
]

# Known efficiency stats from verify_budget + timm model cards
# params in millions, gmac from fvcore
EFFICIENCY: dict[str, dict] = {
    "resnet18":            {"params": 11.69, "gmac": 1.810},
    "mobilenetv2_100":     {"params": 3.50,  "gmac": 0.300},
    "efficientnet_b0":     {"params": 5.29,  "gmac": 0.390},
    "mobilevit_s":         {"params": 5.60,  "gmac": 1.000},
    "efficientformer_l1":  {"params": 12.27, "gmac": 1.300},
    "hybrid_full":         {"params": 3.98,  "gmac": 0.631},
    "hybrid_no_meta":      {"params": 3.68,  "gmac": 0.630},
    "hybrid_cnn_only":     {"params": 1.81,  "gmac": 0.444},
    "hybrid_vit_only":     {"params": 1.97,  "gmac": 0.760},
    # DEKAN — verified via models.verify_budget on 2026-06-09
    "dekan_full":          {"params": 16.45, "gmac": 6.633},
    "dekan_no_meta":       {"params": 15.92, "gmac": 6.632},
    "dekan_linear":        {"params": 16.43, "gmac": 6.633},
    "dekan_densenet_only": {"params": 12.20, "gmac": 6.047},
    "dekan_effnet_only":   {"params": 10.59, "gmac": 1.155},
}

# Colors
C_BASELINE = "#4C72B0"   # blue  — baselines
C_PROPOSED = "#DD3333"   # red   — hybrid_full (lightweight proposed)
C_DEKAN    = "#2CA02C"   # green — dekan_full (flagship proposed)
C_ABLATION = "#888888"   # gray  — ablation variants
C_TTA      = "#FFA040"   # orange accent for TTA bars

# ── Publication style ─────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    9,
        "legend.framealpha":  0.9,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linewidth":     0.5,
    })


def savefig(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path)
    print(f"[saved] {FIGURES_DIR / name}.pdf/png")


# ── Data loading helpers ───────────────────────────────────────────────────────

def load_metrics(model: str, seed: int, tta: bool = False) -> dict | None:
    """Load test_metrics.json or test_metrics_tta.json. Returns None if missing."""
    fname = "test_metrics_tta.json" if tta else "test_metrics.json"
    path  = RESULTS_DIR / model / f"seed{seed}" / "eval" / fname
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_per_class(model: str, seed: int, tta: bool = False) -> pd.DataFrame | None:
    fname = "per_class_metrics_tta.csv" if tta else "per_class_metrics.csv"
    path  = RESULTS_DIR / model / f"seed{seed}" / "eval" / fname
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_conf_matrix(model: str, seed: int, tta: bool = True) -> np.ndarray | None:
    fname = "conf_matrix_tta.npy" if tta else "conf_matrix.npy"
    path  = RESULTS_DIR / model / f"seed{seed}" / "eval" / fname
    # fall back to non-TTA if TTA not available
    if not path.exists() and tta:
        path = RESULTS_DIR / model / f"seed{seed}" / "eval" / "conf_matrix.npy"
    if not path.exists():
        return None
    return np.load(path)


def load_train_log(model: str, seed: int) -> pd.DataFrame | None:
    path = RESULTS_DIR / model / f"seed{seed}" / "logs" / "train_log.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Resumed runs can leave stale rows from earlier attempts with overlapping
    # epoch numbers; keep the last (most recent) row per epoch and re-sort.
    df = df.drop_duplicates(subset="epoch", keep="last").sort_values("epoch")
    return df.reset_index(drop=True)


# ── Figure 1: Training curves ─────────────────────────────────────────────────

def fig_training_curves(seed: int, model: str = "hybrid_full", out_name: str = "01_training_curves") -> None:
    df = load_train_log(model, seed)
    if df is None:
        print(f"[skip] training curves — no log found for {model} seed{seed}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    label = MODEL_LABELS.get(model, model)

    # Loss
    ax = axes[0]
    line_train, = ax.plot(df["epoch"], df["train_loss"], color="#4C72B0", label="Train", linewidth=1.5)
    line_val,   = ax.plot(df["epoch"], df["val_loss"],   color="#DD3333", label="Val",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CB-Focal Loss")
    ax.set_title("Training & Validation Loss")

    # BMA
    ax = axes[1]
    ax.plot(df["epoch"], df["train_bma"], color="#4C72B0", linewidth=1.5)
    ax.plot(df["epoch"], df["val_bma"],   color="#DD3333", linewidth=1.5)
    best_ep  = df.loc[df["val_bma"].idxmax(), "epoch"]
    best_bma = df["val_bma"].max()
    line_best = ax.axvline(best_ep, color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
                            label=f"Best val BMA={best_bma:.4f} (ep {int(best_ep)})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced Multi-class Accuracy (BMA)")
    ax.set_title("Balanced Multi-class Accuracy")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle(f"{label}  —  Training Curves (seed {seed})", fontsize=12, fontweight="bold")
    fig.legend(handles=[line_train, line_val, line_best], loc="lower center",
               ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    savefig(fig, out_name)
    plt.close(fig)


# ── Figure 2: Model comparison bar chart ─────────────────────────────────────

def fig_model_comparison(seed: int) -> None:
    all_models = BASELINE_MODELS + [PROPOSED_MODEL, DEKAN_MODEL]
    rows = []

    for m in all_models:
        base = load_metrics(m, seed, tta=False)
        tta  = load_metrics(m, seed, tta=True)
        if base is None:
            print(f"[skip]  {m} — no eval results found")
            continue
        # Secondary metrics (F1/AUC) are reported at the TTA setting to match the
        # paper's main table; fall back to no-TTA only if a TTA file is missing.
        sec = tta if tta else base
        rows.append({
            "model":    m,
            "label":    MODEL_LABELS.get(m, m),
            "bma":      base.get("bma", 0),
            "f1":       sec.get("macro_f1", 0),
            "auc":      sec.get("macro_auc", 0),
            "bma_tta":  tta.get("bma", 0) if tta else 0,
            "f1_tta":   tta.get("macro_f1", 0) if tta else 0,
        })

    if not rows:
        print("[skip] model comparison — no eval results found")
        return

    df = pd.DataFrame(rows)
    n  = len(df)
    x  = np.arange(n)
    w  = 0.26

    def _bar_color(m: str) -> str:
        if m == PROPOSED_MODEL: return C_PROPOSED
        if m == DEKAN_MODEL:    return C_DEKAN
        return C_BASELINE
    colors = [_bar_color(m) for m in df["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))

    # ── Left: BMA (no TTA) + BMA (TTA) ─────────────────────────────────────
    ax = axes[0]
    bars1 = ax.bar(x - w/2, df["bma"],     width=w, color=colors, alpha=0.9, label="BMA (no TTA)")
    if df["bma_tta"].any():
        bars2 = ax.bar(x + w/2, df["bma_tta"], width=w, color=colors, alpha=0.55,
                       label="BMA (+TTA)", edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Balanced Multi-class Accuracy (BMA)")
    ax.set_title("BMA Comparison")
    ax.set_ylim(0, min(1.0, df[["bma","bma_tta"]].max().max() * 1.22))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Legend patches (collected for a shared legend below the figure)
    baseline_patch = mpatches.Patch(color=C_BASELINE, alpha=0.9, label="Baseline")
    proposed_patch = mpatches.Patch(color=C_PROPOSED, alpha=0.9, label="Ours (Hybrid, lightweight)")
    dekan_patch    = mpatches.Patch(color=C_DEKAN,    alpha=0.9, label="Ours (DEKAN, flagship)")

    # ── Right: Macro-F1 and Macro-AUC ───────────────────────────────────────
    ax = axes[1]
    ax.bar(x - w/2, df["f1"],  width=w, color=colors, alpha=0.9,  label="Macro-F1")
    ax.bar(x + w/2, df["auc"], width=w, color=colors, alpha=0.55, label="Macro-AUC",
           edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Macro-F1 and Macro-AUC (with TTA)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    f1_patch  = mpatches.Patch(color="steelblue", alpha=0.9,  label="Macro-F1 / BMA (no TTA)")
    auc_patch = mpatches.Patch(color="steelblue", alpha=0.55, label="Macro-AUC / BMA (+TTA)")

    fig.suptitle("Model Comparison on ISIC-2019 Test Set", fontsize=12, fontweight="bold")
    fig.legend(handles=[baseline_patch, proposed_patch, dekan_patch, f1_patch, auc_patch],
               loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.13, 1, 0.94])
    savefig(fig, "02_model_comparison")
    plt.close(fig)


# ── Figure 3: Confusion matrix ────────────────────────────────────────────────

def fig_confusion_matrix(seed: int, model: str = "hybrid_full") -> None:
    cm = load_conf_matrix(model, seed, tta=True)
    if cm is None:
        print(f"[skip] confusion matrix — no conf_matrix.npy found for {model}")
        return

    label = MODEL_LABELS.get(model, model)
    fig, ax = plt.subplots(figsize=(8, 7))

    # Mask the diagonal to make off-diagonal errors more visible
    mask_diag = np.eye(len(CLASSES), dtype=bool)

    # Background heatmap for off-diagonal
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        vmin=0.0, vmax=1.0,
        ax=ax,
        cbar=True,
        linewidths=0.4,
        linecolor="white",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )

    # Annotate every cell
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm[i, j]
            color = "white" if val > 0.5 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight)

    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class",      fontsize=11)
    ax.set_title(f"Confusion Matrix (normalised by true class)\n{label}  —  seed {seed}  +TTA",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "03_confusion_matrix")
    plt.close(fig)


# ── Figure 4: Per-class recall comparison ────────────────────────────────────

def fig_per_class_recall(seed: int,
                          proposed: str = "hybrid_full",
                          baseline: str = "efficientnet_b0") -> None:
    """Horizontal bar chart comparing per-class recall of proposed vs best baseline."""
    pc_prop = load_per_class(proposed, seed, tta=True)
    if pc_prop is None:
        pc_prop = load_per_class(proposed, seed)
    pc_base = load_per_class(baseline, seed, tta=True)
    if pc_base is None:
        pc_base = load_per_class(baseline, seed)

    if pc_prop is None:
        print(f"[skip] per-class recall — no results for {proposed}")
        return

    # Sort classes by recall of the proposed model (ascending) for visual clarity
    classes = list(pc_prop["class"])
    prop_recall = list(pc_prop["recall"])
    base_recall = list(pc_base["recall"]) if pc_base is not None else [0] * len(classes)

    # Sort by proposed model recall
    order = np.argsort(prop_recall)
    classes    = [classes[i] for i in order]
    prop_recall = [prop_recall[i] for i in order]
    base_recall = [base_recall[i] for i in order]

    y  = np.arange(len(classes))
    h  = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    b1 = ax.barh(y + h/2, prop_recall, height=h, color=C_PROPOSED, alpha=0.85,
                 label=MODEL_LABELS.get(proposed, proposed))
    if pc_base is not None:
        b2 = ax.barh(y - h/2, base_recall, height=h, color=C_BASELINE, alpha=0.85,
                     label=MODEL_LABELS.get(baseline, baseline))

    # Value labels
    for bar in b1:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", fontsize=8, color=C_PROPOSED, fontweight="bold")
    if pc_base is not None:
        for bar in b2:
            w = bar.get_width()
            ax.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{w:.3f}", va="center", fontsize=8, color=C_BASELINE)

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Recall (sensitivity per class)")
    ax.set_title("Per-Class Recall Comparison", fontweight="bold")
    ax.set_xlim(0, 1.08)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="0.5 threshold")
    ax.legend(loc="lower right")

    plt.tight_layout()
    savefig(fig, "04_per_class_recall")
    plt.close(fig)


# ── Figure 5: Efficiency scatter ──────────────────────────────────────────────

def fig_efficiency_scatter(seed: int) -> None:
    """Params (M) vs BMA with GMAC as bubble size — the efficiency claim figure."""
    points = []
    for model in BASELINE_MODELS + [PROPOSED_MODEL, DEKAN_MODEL]:
        m = load_metrics(model, seed, tta=True) or load_metrics(model, seed)
        if m is None:
            continue
        eff = EFFICIENCY.get(model, {})
        points.append({
            "model":  model,
            "label":  MODEL_LABELS.get(model, model),
            "params": eff.get("params", 0),
            "gmac":   eff.get("gmac", 0),
            "bma":    m.get("bma", 0),
            "is_proposed": model == PROPOSED_MODEL,
            "is_dekan":    model == DEKAN_MODEL,
        })

    if not points:
        print("[skip] efficiency scatter — no eval results found")
        return

    df = pd.DataFrame(points)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for _, row in df.iterrows():
        if row["is_proposed"]:   color, zorder = C_PROPOSED, 5
        elif row["is_dekan"]:    color, zorder = C_DEKAN,    5
        else:                    color, zorder = C_BASELINE, 3
        size = max(80, row["gmac"] * 120)   # bubble area ∝ GMACs (scaled for readability)
        ax.scatter(row["params"], row["bma"],
                   s=size, c=color, alpha=0.80,
                   edgecolors="white", linewidths=0.8, zorder=zorder)
        is_hero = row["is_proposed"] or row["is_dekan"]
        xoff = 0.20 if is_hero else 0.10
        yoff = 0.004 if is_hero else -0.013
        ax.annotate(
            row["label"],
            xy=(row["params"], row["bma"]),
            xytext=(row["params"] + xoff, row["bma"] + yoff),
            fontsize=8.5,
            fontweight="bold" if is_hero else "normal",
            color=color,
        )

    # Budget line
    ax.axvline(6.0, color="red", linestyle=":", linewidth=1.0, alpha=0.5, label="6M param budget")

    ax.set_xlabel("Parameters (M)", fontsize=11)
    ax.set_ylabel("Balanced Multi-class Accuracy (BMA)", fontsize=11)
    ax.set_title("Efficiency vs Accuracy Trade-off\n(bubble size ∝ GMACs)", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Legend for colors
    base_patch   = mpatches.Patch(color=C_BASELINE, alpha=0.80, label="Baseline")
    prop_patch   = mpatches.Patch(color=C_PROPOSED, alpha=0.80, label="Ours (Hybrid, lightweight)")
    dekan_patch  = mpatches.Patch(color=C_DEKAN,    alpha=0.80, label="Ours (DEKAN, flagship)")
    budget_line  = plt.Line2D([0], [0], color="red", linestyle=":", label="6M param budget")
    ax.legend(handles=[base_patch, prop_patch, dekan_patch, budget_line], loc="lower right")

    plt.tight_layout()
    savefig(fig, "05_efficiency_scatter")
    plt.close(fig)


# ── Figure 6: Ablation bar chart ──────────────────────────────────────────────

def fig_ablation(seed: int) -> None:
    rows = []
    for m in ABLATION_MODELS:
        base = load_metrics(m, seed, tta=False)
        tta  = load_metrics(m, seed, tta=True)
        if base is None:
            print(f"[skip ablation]  {m}")
            continue
        rows.append({
            "model":   m,
            "label":   MODEL_LABELS.get(m, m),
            "bma":     base.get("bma", 0),
            "f1":      base.get("macro_f1", 0),
            "bma_tta": tta.get("bma", 0) if tta else 0,
        })

    if not rows:
        print("[skip] ablation — no eval results")
        return

    df = pd.DataFrame(rows)
    n  = len(df)
    x  = np.arange(n)
    w  = 0.30

    colors = [C_PROPOSED if m == PROPOSED_MODEL else C_ABLATION for m in df["model"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars1 = ax.bar(x - w/2, df["bma"],     width=w, color=colors, alpha=0.90, label="BMA")
    bars2 = ax.bar(x + w/2, df["bma_tta"], width=w, color=colors, alpha=0.50,
                   label="BMA + TTA", edgecolor="gray", linewidth=0.5)

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=0, ha="center")
    ax.set_ylabel("Balanced Multi-class Accuracy (BMA)")
    ax.set_title("Ablation Study — Impact of Each Component", fontweight="bold")
    top = df[["bma","bma_tta"]].max().max()
    ax.set_ylim(0, min(1.0, top * 1.18))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ablation_patch = mpatches.Patch(color=C_ABLATION, alpha=0.9, label="Ablation variant")
    proposed_patch = mpatches.Patch(color=C_PROPOSED, alpha=0.9, label="Full model (ours)")
    tta_patch      = mpatches.Patch(color="gray",     alpha=0.5, label="+ TTA")
    ax.legend(handles=[ablation_patch, proposed_patch, tta_patch])

    plt.tight_layout()
    savefig(fig, "06_ablation")
    plt.close(fig)


# ── Figure 7: DEKAN ablation bar chart ───────────────────────────────────────

def fig_dekan_ablation(seed: int) -> None:
    rows = []
    for m in DEKAN_ABLATION_MODELS:
        base = load_metrics(m, seed, tta=False)
        tta  = load_metrics(m, seed, tta=True)
        if base is None:
            print(f"[skip dekan ablation]  {m}")
            continue
        rows.append({
            "model":   m,
            "label":   MODEL_LABELS.get(m, m),
            "bma":     base.get("bma", 0),
            "f1":      base.get("macro_f1", 0),
            "bma_tta": tta.get("bma", 0) if tta else 0,
        })

    if not rows:
        print("[skip] dekan ablation — no eval results")
        return

    df = pd.DataFrame(rows)
    n  = len(df)
    x  = np.arange(n)
    w  = 0.30

    colors = [C_DEKAN if m == DEKAN_MODEL else C_ABLATION for m in df["model"]]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars1 = ax.bar(x - w/2, df["bma"],     width=w, color=colors, alpha=0.90, label="BMA")
    bars2 = ax.bar(x + w/2, df["bma_tta"], width=w, color=colors, alpha=0.50,
                   label="BMA + TTA", edgecolor="gray", linewidth=0.5)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=15, ha="right")
    ax.set_ylabel("Balanced Multi-class Accuracy (BMA)")
    ax.set_title("DEKAN Ablation Study — Dual-Backbone Fusion, Metadata & KAN", fontweight="bold")
    top = df[["bma","bma_tta"]].max().max()
    ax.set_ylim(0, min(1.0, top * 1.18))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ablation_patch = mpatches.Patch(color=C_ABLATION, alpha=0.9, label="Ablation variant")
    dekan_patch    = mpatches.Patch(color=C_DEKAN,    alpha=0.9, label="Full DEKAN (ours)")
    tta_patch      = mpatches.Patch(color="gray",     alpha=0.5, label="+ TTA")
    ax.legend(handles=[ablation_patch, dekan_patch, tta_patch])

    plt.tight_layout()
    savefig(fig, "07_dekan_ablation")
    plt.close(fig)


# ── Figure 8: DEKAN confusion matrix ─────────────────────────────────────────

def fig_confusion_matrix_dekan(seed: int) -> None:
    cm = load_conf_matrix("dekan_full", seed, tta=True)
    if cm is None:
        print("[skip] DEKAN confusion matrix — no conf_matrix_tta.npy for dekan_full")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=False, cmap="Greens", vmin=0.0, vmax=1.0,
        ax=ax, cbar=True, linewidths=0.4, linecolor="white",
        xticklabels=CLASSES, yticklabels=CLASSES,
    )
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm[i, j]
            color = "white" if val > 0.5 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class",      fontsize=11)
    ax.set_title(
        "Confusion Matrix (normalised by true class)\nDEKAN  —  seed 42  +TTA",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, "08_confusion_matrix_dekan")
    plt.close(fig)


# ── Figure A1: Appendix — baseline confusion matrices grid ───────────────────

def fig_appendix_confusion_matrices(seed: int) -> None:
    """2×3 grid of confusion matrices for the five baseline models."""
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))

    for idx, model in enumerate(BASELINE_MODELS):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        cm = load_conf_matrix(model, seed, tta=True)
        if cm is None:
            ax.set_visible(False)
            continue
        sns.heatmap(
            cm, annot=False, cmap="Blues", vmin=0.0, vmax=1.0,
            ax=ax, cbar=False, linewidths=0.3, linecolor="white",
            xticklabels=CLASSES, yticklabels=CLASSES,
        )
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                val = cm[i, j]
                color = "white" if val > 0.5 else "black"
                weight = "bold" if i == j else "normal"
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=6.5, color=color, fontweight=weight)
        ax.set_title(MODEL_LABELS.get(model, model), fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True",      fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused 6th slot
    axes[1, 2].set_visible(False)

    fig.suptitle(
        "Confusion Matrices — Baseline Models\n"
        "(normalised by true class, +TTA; ResNet-18 uses non-TTA matrix)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, "A1_baseline_confusion_matrices")
    plt.close(fig)


# ── Figure A2: Appendix — training curves grid ───────────────────────────────

def fig_appendix_training_curves(seed: int) -> None:
    """4×2 grid of val-BMA training curves for all 7 main models."""
    curve_models = BASELINE_MODELS + [PROPOSED_MODEL, DEKAN_MODEL]
    ncols, nrows = 2, 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 16))

    for idx, model in enumerate(curve_models):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        df = load_train_log(model, seed)
        label = MODEL_LABELS.get(model, model)
        if df is None:
            ax.text(0.5, 0.5, f"{label}\n(no log)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            continue
        if model == PROPOSED_MODEL:    color = C_PROPOSED
        elif model == DEKAN_MODEL:     color = C_DEKAN
        else:                          color = C_BASELINE

        ax.plot(df["epoch"], df["val_bma"],   color=color,  linewidth=1.5, label="Val BMA")
        ax.plot(df["epoch"], df["train_bma"], color=color,  linewidth=1.0,
                alpha=0.45, linestyle="--", label="Train BMA")
        best_ep  = df.loc[df["val_bma"].idxmax(), "epoch"]
        best_bma = df["val_bma"].max()
        ax.axvline(best_ep, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_title(f"{label}  (best val={best_bma:.3f})", fontsize=9, fontweight="bold",
                     color=color)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("BMA",   fontsize=8)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=7, loc="lower right")

    # Hide unused 8th slot
    row, col = divmod(len(curve_models), ncols)
    axes[row, col].set_visible(False)

    fig.suptitle(
        "Training Curves — All Main Models  (BMA vs Epoch, seed 42)\n"
        "Dashed = train BMA,  solid = val BMA,  dotted line = best val checkpoint",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, "A2_all_training_curves")
    plt.close(fig)


# ── Summary table (LaTeX-ready CSV) ───────────────────────────────────────────

def export_summary_table(seed: int) -> None:
    all_models = (BASELINE_MODELS + [PROPOSED_MODEL]
                  + ABLATION_MODELS[:-1]
                  + [DEKAN_MODEL] + DEKAN_ABLATION_MODELS[:-1])
    rows = []
    for m in all_models:
        base = load_metrics(m, seed, tta=False)
        tta  = load_metrics(m, seed, tta=True)
        if base is None:
            continue
        eff = EFFICIENCY.get(m, {})
        # Secondary metrics (F1/AUC/Accuracy) at the TTA setting to match the
        # paper's main table; fall back to no-TTA only if a TTA file is missing.
        sec = tta if tta else base
        rows.append({
            "Model":        MODEL_LABELS.get(m, m),
            "Params (M)":   f"{eff.get('params', 0):.2f}",
            "GMACs":        f"{eff.get('gmac', 0):.3f}",
            "BMA":          f"{base.get('bma', 0):.4f}",
            "BMA+TTA":      f"{tta.get('bma', 0):.4f}" if tta else "—",
            "Macro-F1":     f"{sec.get('macro_f1', 0):.4f}",
            "Macro-AUC":    f"{sec.get('macro_auc', 0):.4f}",
            "Accuracy":     f"{sec.get('accuracy', 0):.4f}",
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    out = FIGURES_DIR / "summary_table.csv"
    FIGURES_DIR.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[saved] {out}")
    print("\n" + df.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--model",    type=str, default="hybrid_full",
                    help="Model to use for single-model figures (curves, CM).")
    ap.add_argument("--baseline", type=str, default="efficientnet_b0",
                    help="Baseline to compare against in per-class recall figure.")
    ap.add_argument("--only",     type=str, default=None,
                    help="Comma-separated list of figure numbers to generate, e.g. '1,3,4'.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_style()

    only = set(args.only.split(",")) if args.only else None

    def should(n: str) -> bool:
        return only is None or n in only

    print(f"\nGenerating figures  seed={args.seed}  model={args.model}")
    print(f"Output directory: {FIGURES_DIR.resolve()}\n")

    if should("1"):  fig_training_curves(args.seed, args.model)
    if should("9"):  fig_training_curves(args.seed, "dekan_full", "09_training_curves_dekan")
    if should("2"):  fig_model_comparison(args.seed)
    if should("3"):  fig_confusion_matrix(args.seed, args.model)
    if should("4"):  fig_per_class_recall(args.seed, args.model, args.baseline)
    if should("5"):  fig_efficiency_scatter(args.seed)
    if should("6"):  fig_ablation(args.seed)
    if should("7"):  fig_dekan_ablation(args.seed)
    if should("8"):  fig_confusion_matrix_dekan(args.seed)
    if should("A1"): fig_appendix_confusion_matrices(args.seed)
    if should("A2"): fig_appendix_training_curves(args.seed)
    export_summary_table(args.seed)

    print("\n[done] all figures written to figures/")


if __name__ == "__main__":
    main()