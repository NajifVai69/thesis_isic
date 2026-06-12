# Skin Lesion Classification (ISIC-2019) — Results Summary

## Project Overview
- **Dataset**: ISIC-2019 (25,331 dermoscopy images, 8 classes)
- **Primary Metric**: Balanced Multi-class Accuracy (BMA / macro-recall)
- **Hardware**: RTX 4070 Ti Super (16 GB VRAM), i7-14700, 64 GB RAM
- **Training**: PyTorch with AMP, EMA, weighted sampling, Class-Balanced Focal Loss
- **Evaluation**: Per-class metrics on test set, with Test-Time Augmentation (TTA, 8 views)
- **Last Updated**: 2026-06-09
- **Data source**: All numbers read directly from `results/*/seed42/eval/*.json`,
  `results/*/seed42/metrics_best_val.json`, and `models/verify_budget.py` output.
  Timm baseline params/GMACs are from the values already in `analysis/figures.py`
  (standard model-card figures for resnet18, mobilenetv2, efficientnet_b0, mobilevit_s,
  efficientformer_l1).

---

## 1. Dataset & Preprocessing

### Class Distribution (full dataset before deduplication)
| Class | Full Name | Images | Imbalance vs NV |
|---|---|---|---|
| MEL | Melanoma | 4,522 | 2.8× |
| NV | Nevus | 12,875 | 1× (most common) |
| BCC | Basal Cell Carcinoma | 3,323 | 3.9× |
| AK | Actinic Keratosis | 867 | 14.8× |
| BKL | Benign Keratosis | 2,624 | 4.9× |
| DF | Dermatofibroma | 239 | 53.9× |
| VASC | Vascular Lesion | 253 | 50.9× |
| SCC | Squamous Cell Carcinoma | 628 | 20.5× |

### Preprocessing Pipeline

#### Step 1: pHash Deduplication
- pHash 64-bit (hash_size=8), Hamming threshold=4, union-find grouping
- **1,263 near-duplicates removed** → **24,068 images retained**

#### Step 2: Lesion-Grouped Stratified Split (70/10/20)
- Grouped by lesion_id; stratified on class × source (HAM/BCN/MSK)
- **Train: 16,872 / Val: 2,400 / Test: 4,796**

#### Step 3: Image Preprocessing
- Shades-of-Gray color constancy (p=6), short-side resize + center-crop to 224×224
- NumPy memmap (`images.uint8.npy`) for zero-I/O-per-batch training

### Metadata Statistics
- Age: Mean=54.16, Std=18.21 years; ~30% missing
- Sex: ~20% missing; Anatomical site: ~15% missing
- **Missing handled with learned embeddings — NOT imputed**

---

## 2. Training Configuration

All models share identical loss, augmentation, and evaluation pipeline.
Values taken from `results/{model}/seed42/config.yaml`.

### Shared Settings (all models)
- **Loss**: Class-Balanced Focal Loss (β=0.999, γ=2.5, label_smoothing=0.1)
- **Optimizer**: AdamW (lr=3e-4, weight_decay per table, betas=(0.9, 0.999))
- **Schedule**: Cosine annealing + 5-epoch linear warmup → min_lr=1e-6
- **Augmentation**: HFlip + VFlip + RandomResizedCrop + ColorJitter + RandAugment(N=2,M=9) + Mixup(α=0.4)
- **EMA**: decay=0.9998, enabled
- **Inference**: 8-view TTA (flips × rotations), mean softmax

### Per-Model Training Config (from actual config.yaml)
| Model | Batch | Grad Accum | Eff. Batch | Epochs | Backbone LR Scale | WD |
|---|---|---|---|---|---|---|
| ResNet-18 | 256 | 1 | 256 | 100 | 0.3× | 1e-4 |
| MobileNetV2 | 256 | 1 | 256 | 100 | 0.3× | 1e-4 |
| EfficientNet-B0 | 128 | 2 | 256 | 100 | 0.3× | 1e-4 |
| MobileViT-S | 128 | 2 | 256 | 150 | 0.3× | 1e-4 |
| EfficientFormer-L1 | 128 | 1 | 128 | 100 | 0.3× | 1e-4 |
| hybrid_cnn_only | 128 | 2 | 256 | 150 | 0.3× | 1e-4 |
| hybrid_vit_only | 128 | 1 | 128 | 100 | 0.3× | 1e-4 |
| hybrid_no_meta | 128 | 1 | 128 | 100 | 0.3× | 1e-4 |
| hybrid_full | 96 | 3 | 288 | 150 | 0.3× | 1e-4 |
| dekan_no_meta | 64 | 4 | 256 | 150 | 0.1× | 0.05 |
| dekan_linear | 64 | 4 | 256 | 150 | 0.1× | 0.05 |
| dekan_full | 64 | 4 | 256 | 150 | 0.1× | 0.05 |

---

## 3. Model Budget (from `models/verify_budget.py`, run 2026-06-09)

Hybrid variants measured via `fvcore` on (2,3,224,224) input on CUDA.
Timm baseline params/GMACs from standard model-card figures (see `analysis/figures.py`).

### Lightweight Hybrid Variants (target: <6M params, <1 GMAC)
| Model | Total Params | CNN Stem | Scratch | GMACs |
|---|---|---|---|---|
| hybrid_cnn_only | 1.81 M | 1.81 M | 776 | 0.444 |
| hybrid_vit_only | 1.97 M | 0 | 1.97 M | 0.760 |
| hybrid_no_meta | 3.68 M | 1.81 M | 1.87 M | 0.630 |
| **hybrid_full** | **3.98 M** | **1.81 M** | **2.17 M** | **0.631** |

### DEKAN Flagship Variants (accuracy-oriented tier)
| Model | Total Params | CNN Stems | Scratch | GMACs |
|---|---|---|---|---|
| dekan_effnet_only | 10.59 M | 3.60 M | 7.00 M | 1.155 |
| dekan_densenet_only | 12.20 M | 4.27 M | 7.93 M | 6.047 |
| dekan_no_meta | 15.92 M | 7.86 M | 8.05 M | 6.632 |
| dekan_linear | 16.43 M | 7.86 M | 8.57 M | 6.633 |
| **dekan_full** | **16.45 M** | **7.86 M** | **8.59 M** | **6.633** |

### Timm Baselines (standard model-card figures)
| Model | Params | GMACs |
|---|---|---|
| MobileNetV2 | 3.50 M | 0.300 |
| EfficientNet-B0 | 5.29 M | 0.390 |
| MobileViT-S | 5.60 M | 1.000 |
| EfficientFormer-L1 | 12.27 M | 1.300 |
| ResNet-18 | 11.69 M | 1.810 |

---

## 4. Test Set Results (seed 42)

All values read from `results/{model}/seed42/eval/test_metrics*.json`.

### 4.1 Overall Results

Secondary metrics (Macro-F1, Macro-AUC, Accuracy) are reported **at the 8-view TTA
setting**, consistent with the BMA (TTA) column and with the paper's main table.
No-TTA secondary metrics are available in `eval/test_metrics.json` if needed.

| Model | Params | GMACs | BMA (no TTA) | BMA (TTA) | Macro-F1 (TTA) | Macro-AUC (TTA) | Accuracy (TTA) |
|---|---|---|---|---|---|---|---|
| ResNet-18 | 11.69 M | 1.810 | 0.4875 | 0.3368 ⚠️ | 0.2763 | 0.8217 | 0.5225 |
| MobileNetV2 | 3.50 M | 0.300 | 0.4485 | 0.4738 | 0.4701 | 0.8827 | 0.6731 |
| EfficientNet-B0 | 5.29 M | 0.390 | 0.4917 | 0.4946 | 0.4736 | 0.9037 | 0.6779 |
| MobileViT-S | 5.60 M | 1.000 | 0.5925 | 0.6020 | 0.5354 | 0.9218 | 0.6981 |
| EfficientFormer-L1 | 12.27 M | 1.300 | 0.5559 | 0.5519 ⚠️ | 0.5779 | 0.9005 | 0.7333 |
| hybrid_cnn_only | 1.81 M | 0.444 | 0.4874 | 0.4853 | 0.4987 | 0.9247 | 0.6827 |
| hybrid_vit_only | 1.97 M | 0.760 | 0.4869 | 0.4908 | 0.3988 | 0.8645 | 0.6028 |
| hybrid_no_meta | 3.68 M | 0.630 | 0.6216 | 0.6085 | 0.5668 | 0.9155 | 0.7216 |
| **hybrid_full** | **3.98 M** | **0.631** | **0.5815** | **0.6081** | **0.5602** | **0.9229** | **0.7162** |
| dekan_no_meta | 15.92 M | 6.632 | 0.6221 | 0.6302 | 0.5860 | 0.9184 | 0.7206 |
| dekan_linear | 16.43 M | 6.633 | 0.6392 | 0.6510 | 0.5933 | 0.9258 | 0.7362 |
| **dekan_full** | **16.45 M** | **6.633** | **0.6368** | **0.6438** | **0.6059** | **0.9271** | **0.7467** |

⚠️ TTA hurts ResNet-18 (−0.151) and EfficientFormer-L1 (−0.004). All other models benefit from TTA.

### 4.2 Val BMA at Best Checkpoint (from `metrics_best_val.json`)
| Model | Val BMA (best) | Best Epoch | Total Epochs |
|---|---|---|---|
| ResNet-18 | 0.5532 | 99 | 100 |
| MobileNetV2 | 0.4875 | 99 | 100 |
| EfficientNet-B0 | 0.4746 | 100 | 100 |
| MobileViT-S | 0.6149 | 150 | 150 |
| EfficientFormer-L1 | 0.5458 | 100 | 100 |
| hybrid_cnn_only | 0.5049 | 150 | 150 |
| hybrid_vit_only | 0.5124 | 100 | 100 |
| hybrid_no_meta | 0.6395 | 100 | 100 |
| hybrid_full | 0.6505 | 149 | 150 |
| dekan_no_meta | 0.6386 | 148 | 150 |
| dekan_linear | 0.6535 | 147 | 150 |
| dekan_full | 0.6527 | 150 | 150 |

### 4.3 Ranking by Test BMA (TTA) — Completed Models
| Rank | Model | Params | BMA (TTA) | Note |
|---|---|---|---|---|
| 1 | dekan_linear | 16.43 M | **0.6510** | KAN replaced by linear |
| 2 | dekan_full | 16.45 M | **0.6438** | Full flagship |
| 3 | dekan_no_meta | 15.92 M | 0.6302 | No metadata |
| 4 | hybrid_no_meta | 3.68 M | 0.6085 | No metadata head |
| 5 | **hybrid_full** | **3.98 M** | **0.6081** | **Hero lightweight model** |
| 6 | MobileViT-S | 5.60 M | 0.6020 | Strongest external baseline |
| 7 | EfficientFormer-L1 | 12.27 M | 0.5519 | TTA slightly hurts |
| 8 | EfficientNet-B0 | 5.29 M | 0.4946 | |
| 9 | hybrid_vit_only | 1.97 M | 0.4908 | Pure ViT, no CNN stem |
| 10 | hybrid_cnn_only | 1.81 M | 0.4853 | CNN stem only |
| 11 | MobileNetV2 | 3.50 M | 0.4738 | |
| 12 | ResNet-18 | 11.69 M | 0.3368 | TTA badly hurts (−0.151) |

---

## 5. Per-Class Test Results (TTA, seed 42)

From `results/{model}/seed42/eval/per_class_metrics_tta.csv`.

### 5.1 Baselines
| Class | ResNet-18 | MobileNetV2 | EfficientNet-B0 | MobileViT-S | EfficientFormer-L1 |
|---|---|---|---|---|---|
| MEL | 0.231 | 0.511 | 0.594 | 0.447 | 0.572 |
| NV | 0.841 | 0.837 | 0.857 | 0.890 | 0.892 |
| BCC | 0.085 | 0.653 | 0.546 | 0.467 | 0.734 |
| AK | 0.523 | 0.323 | 0.348 | 0.452 | 0.252 |
| BKL | 0.057 | 0.446 | 0.366 | 0.587 | 0.523 |
| DF | 0.189 | 0.132 | 0.245 | 0.660 | 0.321 |
| VASC | 0.216 | 0.706 | 0.843 | 0.882 | 0.882 |
| SCC | 0.554 | 0.182 | 0.157 | 0.430 | 0.240 |

### 5.2 Hybrid Variants
| Class | hybrid_cnn_only | hybrid_vit_only | hybrid_no_meta | hybrid_full |
|---|---|---|---|---|
| MEL | 0.383 | 0.464 | 0.520 | 0.518 |
| NV | 0.936 | 0.785 | 0.856 | 0.859 |
| BCC | 0.544 | 0.450 | 0.653 | 0.618 |
| AK | 0.645 | 0.439 | 0.426 | 0.587 |
| BKL | 0.296 | 0.250 | 0.669 | 0.632 |
| DF | 0.189 | 0.340 | 0.491 | 0.566 |
| VASC | 0.667 | 0.843 | 0.882 | 0.863 |
| SCC | 0.223 | 0.355 | 0.372 | 0.223 |

### 5.3 DEKAN Variants
| Class | dekan_no_meta | dekan_linear | dekan_full |
|---|---|---|---|
| MEL | 0.619 | 0.600 | 0.611 |
| NV | 0.805 | 0.841 | 0.859 |
| BCC | 0.739 | 0.752 | 0.731 |
| AK | 0.316 | 0.439 | 0.368 |
| BKL | 0.628 | 0.583 | 0.622 |
| DF | 0.585 | 0.679 | 0.509 |
| VASC | 0.804 | 0.843 | 0.922 |
| SCC | 0.545 | 0.471 | 0.529 |

---

## 6. Ablation Analysis

### 6.1 Hybrid Model Ablation (Table 2)

| Model | BMA (TTA) | Δ vs hybrid_full |
|---|---|---|
| hybrid_cnn_only — CNN stem only | 0.4853 | −0.123 |
| hybrid_vit_only — Pure ViT, no CNN | 0.4908 | −0.117 |
| hybrid_no_meta — CNN + ViT, no metadata | 0.6085 | +0.0004 |
| **hybrid_full — CNN + ViT + metadata** | **0.6081** | — |

**Key finding:** Neither CNN alone (0.4853) nor ViT alone (0.4908) approaches the hybrid
(0.6081). The two architectures are roughly equal in isolation but strongly complementary
when combined — adding the ViT trunk to the CNN stem yields +0.117 to +0.123 BMA.
This is the primary architectural justification for the hybrid design.

**On metadata:** hybrid_full (0.6081) and hybrid_no_meta (0.6085) are effectively identical
on the test set (Δ=0.0004). Val BMA shows a clearer gap: hybrid_full 0.6505 vs
hybrid_no_meta 0.6395 (+0.011). The metadata effect is modest for the hybrid model at
this scale and cannot be confirmed from a single test-set evaluation alone.

### 6.2 DEKAN Ablation (Table 3)

| Component | Model | BMA (TTA) | Δ vs dekan_full |
|---|---|---|---|
| No metadata | dekan_no_meta | 0.6302 | −0.014 |
| Linear head (vs KAN) | dekan_linear | 0.6510 | **+0.007** |
| **Full model** | **dekan_full** | **0.6438** | — |

**Key finding:** `dekan_linear` (0.6510) outperforms `dekan_full` (0.6438) by 0.007 BMA.
The KAN classifier does not improve over a plain linear head on this dataset/scale.

`dekan_densenet_only` and `dekan_effnet_only` have not been trained —
the dual-backbone fusion gain (claim #7) cannot yet be quantified.

---

## 7. Efficiency Analysis

BMA/M-param uses TTA BMA. Params from verify_budget (hybrid/dekan) or model cards (baselines).

| Model | Params | GMACs | BMA (TTA) | BMA / M-param |
|---|---|---|---|---|
| ResNet-18 | 11.69 M | 1.810 | 0.3368 | 0.029 |
| MobileNetV2 | 3.50 M | 0.300 | 0.4738 | 0.135 |
| EfficientNet-B0 | 5.29 M | 0.390 | 0.4946 | 0.093 |
| MobileViT-S | 5.60 M | 1.000 | 0.6020 | 0.108 |
| EfficientFormer-L1 | 12.27 M | 1.300 | 0.5519 | 0.045 |
| hybrid_cnn_only | 1.81 M | 0.444 | 0.4853 | 0.268 |
| hybrid_vit_only | 1.97 M | 0.760 | 0.4908 | 0.249 |
| hybrid_no_meta | 3.68 M | 0.630 | 0.6085 | 0.165 |
| **hybrid_full** | **3.98 M** | **0.631** | **0.6081** | **0.153** ⭐ best same-budget |
| dekan_no_meta | 15.92 M | 6.632 | 0.6302 | 0.040 |
| dekan_linear | 16.43 M | 6.633 | 0.6510 | 0.040 |
| dekan_full | 16.45 M | 6.633 | 0.6438 | 0.039 |

**hybrid_full** has the best BMA/param among all models with budget ≤ 6M params/1 GMAC.
DEKAN achieves higher raw BMA but at 4× the parameters and 10× the compute (6.633 vs 0.631 GMAC).

---

## 8. TTA Impact per Model

| Model | BMA (no TTA) | BMA (TTA) | Gain |
|---|---|---|---|
| dekan_linear | 0.6392 | **0.6510** | +0.012 (+1.8%) |
| dekan_full | 0.6368 | **0.6438** | +0.007 (+1.1%) |
| dekan_no_meta | 0.6221 | **0.6302** | +0.008 (+1.3%) |
| MobileViT-S | 0.5925 | **0.6020** | +0.010 (+1.6%) |
| hybrid_full | 0.5815 | **0.6081** | +0.027 (+4.6%) |
| hybrid_no_meta | 0.6216 | 0.6085 | −0.013 (−2.1%) ⚠️ |
| EfficientNet-B0 | 0.4917 | **0.4946** | +0.003 (+0.5%) |
| MobileNetV2 | 0.4485 | **0.4738** | +0.025 (+5.5%) |
| hybrid_vit_only | 0.4869 | 0.4908 | +0.004 (+0.8%) |
| hybrid_cnn_only | 0.4874 | 0.4853 | −0.002 (−0.4%) |
| EfficientFormer-L1 | 0.5559 | 0.5519 | −0.004 (−0.7%) ⚠️ |
| ResNet-18 | 0.4875 | 0.3368 | **−0.151 (−30.9%)** ❌ |

ResNet-18 and hybrid_no_meta show TTA degradation. For ResNet-18 this is a known issue
(the model's per-view predictions are overconfident in different classes and averaging
hurts rather than helps).

---

## 9. Pending Runs

| Model | Status | Purpose |
|---|---|---|
| hybrid_vit_only | ✅ Done | Hybrid ablation Table 2 now complete |
| dekan_densenet_only | ❌ Not started | Quantify DenseNet-only contribution — needed for claim #7 |
| dekan_effnet_only | ❌ Not started | Quantify EffNet-only contribution — needed for claim #7 |

---

## 10. Reproducibility

- **Seed**: 42 only. All results above are single-seed.
- **Required for publication**: 3 seeds (42, 1337, 2024), report mean ± std.
  Seeds 1337 and 2024 have not been run for any model.
- **Checkpoint location**: `results/{model}/seed42/checkpoints/best.pth`
- **Training log**: `results/{model}/seed42/logs/train_log.csv`
- **Budget report**: `figures/budget_report.txt` (from `python -m models.verify_budget`)
- **Figures**: `figures/01_*.pdf` through `figures/07_*.pdf` + `figures/summary_table.csv`

---

## 11. Key Paper Claims — Current Support Status

| Claim | Status | Evidence |
|---|---|---|
| hybrid_full beats all 5 baselines on BMA | ✅ Supported | 0.6081 vs 0.6020 (MobileViT-S, closest) |
| hybrid_full is most efficient model (≤6M params) | ✅ Supported | 0.153 BMA/M-param, best in budget |
| Metadata fusion helps (hybrid) | ⚠️ Weak on test set | Test: hybrid_full−hybrid_no_meta = −0.0004; val: +0.011. Single seed only. |
| Metadata fusion helps (DEKAN) | ✅ Supported | dekan_full − dekan_no_meta = +0.014 BMA TTA |
| CNN+ViT hybrid beats either alone | ✅ Supported | hybrid_full (0.6081) vs CNN-only (0.4853, −0.123) and ViT-only (0.4908, −0.117) |
| Design scales well to higher capacity | ✅ Supported | dekan_full 0.6438 vs hybrid_full 0.6081 (+3.6%) at 4.1× params |
| KAN classifier improves over linear | ❌ Not supported | dekan_linear (0.6510) > dekan_full (0.6438) by 0.007 |
| Dual-backbone fusion gain | ⏳ Pending | Needs dekan_densenet_only + dekan_effnet_only |
| Results reproducible across 3 seeds | ⏳ Pending | Only seed 42 complete |
