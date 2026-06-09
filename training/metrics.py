"""Evaluation metrics for ISIC-2019 classification.

Primary metric: Balanced Multi-class Accuracy (BMA) = macro-averaged recall.
  - The ISIC-2019 challenge used this metric (see official leaderboard).
  - Overall accuracy is misleading on the 53× imbalanced dataset (a model that
    always predicts NV scores ~51% overall accuracy).

Secondary metrics: macro-F1, macro-AUC, overall accuracy.
Per-class: recall, precision, F1 for each of the 8 classes.

All metrics are computed via torchmetrics for numerical correctness and GPU
support. The MetricCollection pattern means one .update() call updates all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from utils.labels import CLASSES

NUM_CLASSES = len(CLASSES)


class ISICMetrics:
    """Wrapper that groups all metrics and provides a clean .compute() dict.

    All tensors accumulated in CPU memory (safe with any GPU).

    Usage:
        m = ISICMetrics()
        for batch in loader:
            logits, labels = ...
            m.update(logits, labels)
        results = m.compute()   # dict of {metric_name: value}
        m.reset()
    """

    def __init__(self):
        # Torchmetrics operates on CPU to avoid per-batch GPU syncs in the
        # validation loop.  Predictions are moved to CPU before .update().
        kwargs = dict(num_classes=NUM_CLASSES)

        self.bma          = MulticlassRecall(**kwargs, average="macro")
        self.accuracy     = MulticlassAccuracy(**kwargs, average="micro")
        self.macro_f1     = MulticlassF1Score(**kwargs, average="macro")
        self.macro_auc    = MulticlassAUROC(**kwargs, average="macro")
        self.conf_matrix  = MulticlassConfusionMatrix(**kwargs, normalize="true")

        # Per-class metrics
        self.per_class_recall     = MulticlassRecall(**kwargs, average="none")
        self.per_class_precision  = MulticlassPrecision(**kwargs, average="none")
        self.per_class_f1         = MulticlassF1Score(**kwargs, average="none")

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Args:
            logits:  (B, C) — raw scores or probabilities (AUC needs probs; we
                     apply softmax internally so raw logits are fine too).
            targets: (B,) integer labels in [0, NUM_CLASSES).
        """
        logits  = logits.detach().cpu()
        targets = targets.detach().cpu()
        probs   = torch.softmax(logits, dim=1)
        preds   = logits.argmax(dim=1)

        self.bma.update(preds, targets)
        self.accuracy.update(preds, targets)
        self.macro_f1.update(preds, targets)
        self.macro_auc.update(probs, targets)
        self.conf_matrix.update(preds, targets)
        self.per_class_recall.update(preds, targets)
        self.per_class_precision.update(preds, targets)
        self.per_class_f1.update(preds, targets)

    def compute(self) -> dict:
        """Return a flat dict of all metric values."""
        conf = self.conf_matrix.compute().numpy()           # (C, C) float
        per_recall = self.per_class_recall.compute().numpy()
        per_prec   = self.per_class_precision.compute().numpy()
        per_f1     = self.per_class_f1.compute().numpy()

        return {
            "bma":          float(self.bma.compute()),
            "accuracy":     float(self.accuracy.compute()),
            "macro_f1":     float(self.macro_f1.compute()),
            "macro_auc":    float(self.macro_auc.compute()),
            "conf_matrix":  conf,                           # numpy (C, C)
            "per_class":    pd.DataFrame({
                "class":     CLASSES,
                "recall":    per_recall,
                "precision": per_prec,
                "f1":        per_f1,
            }),
        }

    def reset(self) -> None:
        self.bma.reset()
        self.accuracy.reset()
        self.macro_f1.reset()
        self.macro_auc.reset()
        self.conf_matrix.reset()
        self.per_class_recall.reset()
        self.per_class_precision.reset()
        self.per_class_f1.reset()


def format_results(results: dict) -> str:
    """Pretty-print the results dict to a human-readable string."""
    lines = [
        f"  BMA (balanced accuracy)  : {results['bma']:.4f}",
        f"  Overall accuracy         : {results['accuracy']:.4f}",
        f"  Macro F1                 : {results['macro_f1']:.4f}",
        f"  Macro AUC                : {results['macro_auc']:.4f}",
        "",
        "  Per-class metrics:",
    ]
    pc = results["per_class"]
    for _, row in pc.iterrows():
        lines.append(
            f"    {row['class']:6s}  recall={row['recall']:.3f}  "
            f"prec={row['precision']:.3f}  f1={row['f1']:.3f}"
        )
    return "\n".join(lines)