# Formulas Used in the Thesis — Explained

This covers every formula that appears (explicitly or as a named quantity) in
`paper/chapters/` and `paper/appendix/appendix_1.tex`, plus the preprocessing
code it references. Grouped by where it's used in the pipeline.

---

## 1. Shades-of-Gray Color Constancy (Preprocessing, Chapter 5 §1)

**Where:** `preprocessing/color_constancy.py::shades_of_gray`

**Formula (Minkowski p-norm, per channel c ∈ {R,G,B}):**

```
norm_c = ( mean_over_pixels( I_c ^ p ) ) ^ (1/p)

gray   = mean(norm_R, norm_G, norm_B)

scale_c = gray / norm_c

I_c_corrected = clip( I_c * scale_c, 0, 255 )
```

With **p = 6** (the "canonical Shades-of-Gray" value, per Finlayson & Trezzi 2004).

**How it works / intuition:**
- Each color channel's overall "brightness level" is summarized by a Minkowski
  p-norm of its pixel values — at p=1 this is just the mean (grey-world
  assumption), and as p→∞ it approaches the max pixel value (max-RGB
  assumption). p=6 sits between these, weighting bright pixels more than the
  mean but less than the max.
- `gray` is the average of the three channel norms — what the "neutral" norm
  *should* be if there were no color cast.
- `scale_c` rescales each channel so its norm matches `gray` — e.g., if the
  blue channel's norm is unusually high (a bluish color cast from a
  dermatoscope's light source), `scale_blue < 1` darkens blue back toward
  neutral.
- **Why it matters for this thesis:** ISIC-2019 is a union of HAM10000,
  BCN_20000, and MSK, each captured with different dermatoscope hardware and
  lighting. Without this correction, the model could learn to associate
  *color cast* (a proxy for source dataset) with class label — a confound,
  since source correlates with class distribution. Doing this once in
  preprocessing means the model never has to learn this invariance itself.

---

## 2. Class-Balanced Focal Loss (Chapter 5 §"Class-Balanced Focal Loss")

This is the **one formally-typeset equation** in the thesis (Eq. in chapter_5.tex)
and the most important formula to know cold for defense.

### 2a. Effective number of samples

```
E(n) = (1 - β^n) / (1 - β)
```

- `n` = number of training images in a class (e.g., n_NV = 12,875, n_DF = 239)
- `β` ∈ [0,1) is a hyperparameter — **β = 0.999** in this thesis
- **Intuition:** as you add more samples of a class, each *additional* sample
  overlaps more with existing samples (in feature space), so it contributes
  less "new information." E(n) models this diminishing-returns effect — it
  grows much more slowly than n itself for large n.

### 2b. Per-class weight

```
α_c = (1 - β) / (1 - β^(n_c))     [then normalized so Σ α_c = number of classes]
```

- This is **1 / E(n_c)** (up to the normalization), so classes with fewer
  effective samples get a *larger* weight.
- For NV (n=12,875) this weight is small; for DF (n=239) it's much larger —
  this is the direct mechanism that counteracts the 53.9× imbalance.

### 2c. The full per-sample loss

```
L = - α_y · (1 - p_y)^γ · CE_smooth(logits, y)
```

- `y` = true class, `p_y` = model's predicted probability for the true class
  (after softmax)
- `α_y` = the class-balanced weight for the true class (from 2b)
- `(1 - p_y)^γ` = the **focal modulation term** (Lin et al. 2017), with
  **γ = 2.5**
- `CE_smooth` = cross-entropy with **label smoothing = 0.1** applied to the
  one-hot target before computing CE

**How the three pieces work together:**
| Term | What it does | Effect |
|---|---|---|
| `α_y` | Reweights by class rarity | Rare classes (DF, VASC, AK, SCC) get amplified gradient |
| `(1-p_y)^γ` | Down-weights "easy" examples (where p_y is already high/confident) | Training focuses on hard/ambiguous examples regardless of class |
| label smoothing | Softens the target from [0,...,1,...,0] to e.g. [0.0125,...,0.9125,...,0.0125] | Prevents overconfidence, regularizes |

**Why β=0.999 and not more aggressive (e.g. 0.9999):**
A larger β makes `1 - β^n` ≈ 1 for *all* classes regardless of n (since β^n →
0 very fast), which collapses E(n) ≈ 1/(1-β) for every class — but for very
small n, β^n is still close to 1, producing an enormous α for rare classes
relative to common ones. In preliminary runs this extreme ratio caused rare
classes (DF, VASC) to get such large loss gradients early in training that
optimization became unstable / rare-class accuracy collapsed. β=0.999 is a
sweet spot: meaningful reweighting without destabilizing optimization.

---

## 3. Evaluation Metrics (Chapter 5 §"Evaluation Protocol", Chapter 6)

These aren't written as LaTeX equations in the thesis, but they're the
formulas behind every number on your slides 12/14/15 — know these cold.

### 3a. Per-class recall (sensitivity)

```
recall_c = TP_c / (TP_c + FN_c)
```
= "Of all images that are truly class c, what fraction did the model
correctly identify as class c?"

### 3b. Balanced Multi-class Accuracy (BMA) — **PRIMARY METRIC**

```
BMA = (1/C) · Σ_{c=1}^{C} recall_c          (C = 8 classes)
```

- The **unweighted mean of the 8 per-class recalls**. Also called
  "macro-recall" or "balanced accuracy."
- **Why it's primary (the core argument for slide 11/12):** plain accuracy is
  `(correct predictions) / (total predictions)` — it implicitly weights each
  *image*, so a model that always predicts NV (12,875 images, the majority
  class) scores ~50% accuracy while having 0% recall on DF/VASC/SCC/AK. BMA
  weights each *class* equally regardless of how many images it has — a model
  must do reasonably well on DF (239 images) AND NV (12,875 images) to score
  well on BMA.

### 3c. Macro-F1

```
F1_c = 2 · (precision_c · recall_c) / (precision_c + recall_c)

macro-F1 = (1/C) · Σ_c F1_c
```
where `precision_c = TP_c / (TP_c + FP_c)`. Like BMA but also penalizes a
class for "false alarms" stolen from other classes, not just missed
detections.

### 3d. Macro-AUC

```
macro-AUC = (1/C) · Σ_c AUC(class c vs. rest)
```
Each class's AUC is computed as the standard one-vs-rest ROC-AUC using the
model's softmax probability for class c as the score; macro-AUC averages
these across the 8 classes. Unlike recall/F1, AUC doesn't depend on a
decision threshold — it measures how well-separated the predicted
probabilities are.

### 3e. Overall accuracy

```
accuracy = (total correct) / (total images)
```
Reported as a secondary metric specifically *because* it can look deceptively
good under imbalance (see 3b) — the BMA-vs-accuracy gap on your slides is
itself evidence for why BMA was chosen.

---

## 4. Test-Time Augmentation (TTA) Averaging (Chapter 5 §"Evaluation Protocol")

```
softmax_TTA(x) = (1/V) · Σ_{v=1}^{V} softmax( model(T_v(x)) )
```

- `V = 8` views: the original image plus a fixed set of flips
  (horizontal/vertical) and rotations
- `T_v` = the v-th transform (identity, hflip, vflip, 90°/180°/270° rotations,
  and combinations)
- The model's softmax output is computed for each transformed view, and the
  8 softmax vectors are **averaged** (not the logits — softmax outputs) to
  get the final prediction probabilities.

**Why this can help or hurt (relevant to your TTA-impact table):**
- If the model's predictions are *consistent* across views (i.e., correctly
  invariant to flips/rotations — lesions have no canonical orientation),
  averaging reduces variance and sharpens the correct class → BMA improves
  (most models, e.g. hybrid_full: +0.027).
- If the model is *not* rotation/flip-invariant and produces confidently
  *different wrong* predictions per view, averaging blends these wrong
  predictions together and can produce a worse final answer than the
  single-view prediction → BMA drops (ResNet-18: −0.151, the most extreme
  case).

---

## 5. Efficiency Metric — BMA per Million Parameters (Chapter 6 §"Efficiency")

```
BMA/M-param = BMA(TTA) / (Params in millions)
```

- A simple normalization used to compare models of very different sizes on a
  "balanced accuracy per unit of model size" basis.
- hybrid_full: 0.6081 / 3.98 ≈ **0.153** (best among models with meaningful
  absolute BMA)
- dekan_full: 0.6438 / 16.45 ≈ **0.039**

**Caveat (good Q&A material):** this metric rewards small models regardless
of *absolute* usefulness — hybrid_cnn_only (0.4853/1.81 ≈ 0.268) scores
*higher* than hybrid_full on this metric despite being far less accurate. It
should always be read alongside the absolute BMA column, not in isolation.

---

## 6. Optimization Schedule (Chapter 5 §"Training Protocol")

Not written as an equation in the thesis, but the two pieces are standard
formulas worth knowing:

### 6a. Linear warmup (epochs 1–5)
```
lr(epoch) = base_lr * (epoch / warmup_epochs)        for epoch ≤ 5
```

### 6b. Cosine annealing (epochs 6–end)
```
lr(epoch) = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * (epoch - warmup) / (total_epochs - warmup)))
```
- `base_lr = 3×10⁻⁴`, `min_lr = 10⁻⁶`
- **Intuition:** warmup ramps the LR up gradually so the randomly-initialized
  transformer/classifier doesn't immediately produce large, destabilizing
  gradients into the pretrained CNN stem. Cosine annealing then smoothly
  decays the LR to near-zero, giving a long "fine-tuning tail" at low LR for
  stable convergence — visible in your training curves (Figs for slide 16) as
  smooth, non-jagged loss/BMA curves with no late-training instability.

---

## Quick reference — which formula matters for which slide

| Slide | Formula(s) |
|---|---|
| 11 (Training/Eval Protocol) | §2 (CB-Focal Loss), §6 (LR schedule), §4 (TTA averaging) |
| 12 (Main Results table) | §3b (BMA), §3c (macro-F1), §3d (macro-AUC), §3e (accuracy) |
| 14 (Per-class recall) | §3a (recall) |
| 15 (Efficiency trade-off) | §5 (BMA/M-param) |

**The one formula to be able to write on a whiteboard from memory:**
```
L = -α_y (1 - p_y)^γ · CE_smooth(logits, y)
```
with α from effective-number reweighting (β=0.999) and γ=2.5.
