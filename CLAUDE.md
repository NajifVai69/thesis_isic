# Skin Lesion Classification — ISIC-2019

## Project goal
Research paper on skin disease classification targeting **MICCAI / ISBI / EMBC** submission.
Primary contribution: a **lightweight hybrid CNN–ViT with clinical metadata cross-attention fusion**.
We beat same-budget lightweight baselines; the headline metric is **Balanced Multi-class Accuracy (BMA)**, not overall accuracy.

## Dataset
**ISIC-2019** — Kaggle mirror: https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification
- 25,331 labeled dermoscopy images, 8 classes: `MEL, NV, BCC, AK, BKL, DF, VASC, SCC`
- Class-index order is locked in `utils/labels.py::CLASSES` — never reorder.
- Union of three sources: **HAM10000**, **BCN_20000**, **MSK** (inferred by image-ID prefix in `utils/labels.py::infer_source`)
- Severe imbalance: NV ~12,875 vs DF/VASC ~239–253 (~53× ratio)
- Metadata per image: `age_approx`, `sex`, `anatom_site_general`, `lesion_id` — has meaningful missingness (~30% in some fields); handled with a learned `-1` missing token, NOT imputation
- Per-class folders on disk: `AK/`, `BCC/`, `BKL/`, `DF/`, `MEL/`, `NV/`, `SCC/`, `VASC/`

## Hardware (Windows machine — all training happens here)
- **GPU:** RTX 4070 Ti Super 16 GB VRAM
- **CPU:** i7-14700 (20C/28T)
- **RAM:** 64 GB DDR5
- **SSD:** 1 TB
- **OS:** Windows — `num_workers=0` in ALL DataLoaders (non-negotiable; deadlocks with >0)

## Critical constraints (never break these)
- `num_workers=0` always — Windows DataLoader deadlock
- Always use `torch.amp.autocast` + `GradScaler` (AMP) — required for 16 GB budget
- Always `model.to(memory_format=torch.channels_last)` + `tensor.to(memory_format=torch.channels_last)` on 2D images
- Always `torch.compile(model, mode="reduce-overhead")` when using PyTorch 2.x
- Batch sizes: ResNet18/MobileNetV2/EfficientNet-B0 → 256; MobileViT-S/EfficientFormer → 128; proposed hybrid → 96 (with gradient accumulation to effective 256)

## Architecture decisions (agreed)
### Proposed model
1. **CNN stem**: MobileNetV2-style inverted residuals (first 2–3 stages only, ~1–2M params) → local dermoscopic texture
2. **ViT trunk**: small (6–8 layers, dim 192–256, 4 heads) on the CNN feature map → global structure (asymmetry, border)
3. **Metadata cross-attention head**: embed `age`, `sex`, `anatom_site` → cross-attend to ViT CLS token → classifier. Missing fields = learned `-1` token embedding
4. Budget target: **< 6M params, < 1 GFLOP**

### Baselines (all trained with identical loss/aug/schedule for fair comparison)
1. ResNet18
2. MobileNetV2
3. EfficientNet-B0
4. MobileViT-S (fair lightweight-hybrid baseline — critical, reviewers will ask)
5. EfficientFormer-L1 (second lightweight-hybrid baseline)

## Training decisions (agreed)
- **Loss:** Class-Balanced Focal Loss (Cui et al. CVPR 2019, effective-number weighting β=0.9999)
- **Optimizer:** AdamW, lr=3e-4 (1e-3 for scratch-trained parts), weight decay=0.05
- **Schedule:** Cosine with 5-epoch warmup, 60 epochs total
- **Augmentation:** flips, 90° rotations, random resized crop + RandAugment (N=2, M=9). No CutMix/MixUp (hurts fine-grained dermoscopy)
- **EMA:** weight EMA decay=0.9998 — cheap, +0.5–1% BMA
- **TTA:** 8 views (hflip + vflip + 4-rot, mean softmax) at eval — report in paper
- **Seeds:** 3 seeds, report mean ± std. Non-negotiable for publication.

## Evaluation (agreed)
- **Primary metric:** BMA (Balanced Multi-class Accuracy = macro-recall)
- **Secondary:** macro-F1, macro-AUC, per-class recall, overall accuracy
- Confusion matrix (normalized by true class) per model
- Train + val loss/BMA curves per model
- **Efficiency table:** params, GMACs (via `fvcore`), latency batch=1 on GPU + CPU
- **Grad-CAM** on CNN stem + **attention rollout** on ViT trunk (qualitative figure)
- **External eval:** ISIC-2019 test → map 8-class softmax to mel-vs-rest on ISIC-2020 (zero-shot, binary)
- **Ablation table:** (1) CNN only, (2) pure ViT, (3) hybrid no metadata, (4) full model

## Preprocessing pipeline (Phase 1 — COMPLETE)
All scripts run via: `python -m preprocessing.run_all --config configs/default.yaml`

### Step 1 — pHash dedup (`preprocessing/dedup_phash.py`)
- pHash (64-bit, hash_size=8) every image
- Chunked pairwise Hamming distance (chunk=1024, threshold=4)
- Union-find grouping; keep lexicographically smallest image-id per group
- Outputs: `{work_dir}/phash.csv`, `{work_dir}/dedup_keep.csv`

### Step 2 — Stratified split (`preprocessing/split.py`)
- 70/10/20 train/val/test
- Stratified on **class × source** (HAM/BCN/MSK)
- Grouped by **lesion_id** — all images of the same lesion go to the same split
- Outputs: `{work_dir}/split.csv`

### Step 3 — Memmap builder (`preprocessing/build_memmap.py`)
- Decode → Shades-of-Gray (p=6) → short-side-resize + center-crop → 224×224 uint8
- Writes one (N, 224, 224, 3) uint8 `.npy` memmap (~3.6 GB, fits fully in 64 GB RAM)
- Also writes `labels.int64.npy`, `meta.npz` (age float32, sex_idx/site_idx/source_idx int16 with -1 for missing), `memmap_index.csv`
- Loaded at training time with `np.load(path, mmap_mode='r')` — whole tensor stays in page cache after first epoch, per-batch cost collapses to a memcpy

## Key files
| File | Purpose |
|------|---------|
| `configs/default.yaml` | All paths and hyperparameter knobs |
| `utils/config.py` | Config loader, Paths dataclass, `resolve_image_path()` |
| `utils/labels.py` | `load_groundtruth()`, `load_metadata()`, `infer_source()`, `CLASSES` |
| `preprocessing/color_constancy.py` | `shades_of_gray(img, p)` |
| `preprocessing/dedup_phash.py` | pHash + near-dup grouping |
| `preprocessing/split.py` | Lesion-grouped stratified split |
| `preprocessing/build_memmap.py` | Full decode + CC + resize → memmap |
| `preprocessing/run_all.py` | Orchestrator (subprocesses for Windows safety) |
| `notebooks/01_eda.ipynb` | Class dist, source dist, missingness, image grids, size stats |

## What is NOT done yet (Phase 2+)
- [ ] PyTorch Dataset class reading the memmap (`datasets/`)
- [ ] GPU-side augmentation pipeline
- [ ] Class-Balanced Focal Loss (`losses/`)
- [ ] ResNet18 baseline training script
- [ ] MobileNetV2 + EfficientNet-B0 baselines
- [ ] MobileViT-S + EfficientFormer-L1 baselines
- [ ] Proposed hybrid model (`models/`)
- [ ] Training + eval scripts with AMP, EMA, TTA, curves, confusion matrix
- [ ] Grad-CAM / attention rollout visualization
- [ ] External eval on ISIC-2020
- [ ] `configs/training.yaml`

## Novelty framing (for the paper)
- NOT "we combined 5 imbalance techniques". Single principled loss (CB-Focal).
- Differentiator vs Pacheco & Krohling (2021): they did metadata fusion on heavy CNNs; we extend it to a **lightweight hybrid backbone** with a specific efficiency claim (< 6M params, < 1 GFLOP).
- Related work to cite explicitly: Pacheco & Krohling 2021, MobileViT, EfficientFormer, Cui et al. 2019 (CB loss), Finlayson & Trezzi 2004 (Shades-of-Gray).
- External test on ISIC-2020 is the generalization claim; reviewers will ask for it.

## Code style
- All scripts importable as `python -m preprocessing.<name>` from repo root
- `pathlib.Path` everywhere — no `os.path`
- `sys.path.insert(0, repo_root)` at the top of each runnable script
- No hardcoded Windows paths in code — all paths come from `configs/default.yaml`
- `tqdm` progress bars on all long loops
- Explicit error messages when images are missing (warn + continue, never silent)
