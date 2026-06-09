# Skin Lesion Classification on ISIC-2019

Research codebase for a lightweight hybrid CNN–ViT with metadata cross-attention
fusion. This first commit covers Phase 1: **data preparation and EDA**.

## Layout

```
configs/        YAML configs (start with default.yaml)
preprocessing/  dedup, color constancy, split, memmap builder
datasets/       PyTorch Datasets (Phase 2)
models/         baselines + proposed model (Phase 3)
losses/         class-balanced focal (Phase 3)
training/       train/eval scripts (Phase 3)
utils/          shared helpers (config, label/source inference)
notebooks/      01_eda.ipynb
results/        confusion matrices, curves, csv logs (Phase 3)
```

## One-time setup (Windows + CUDA 12)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
# Install PyTorch CUDA wheel first (matches your 4070 Ti Super):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Then everything else:
pip install -r requirements.txt
```

## Edit the config

Open `configs/default.yaml` and set:

- `data.root`: the folder that contains the per-class folders (`AK/`, `BCC/`, …)
  and `ISIC_2019_Training_GroundTruth.csv`, `ISIC_2019_Training_Metadata.csv`.
- `data.work_dir`: where the preprocessing artifacts will be written.
  Put this on the SSD; ~5 GB total.

Forward slashes work on Windows (`pathlib` handles it).

## Run preprocessing

```powershell
python -m preprocessing.run_all --config configs/default.yaml
```

This runs three steps in order:

1. **`dedup_phash`** — pHash every image, group near-duplicates by Hamming
   distance, keep one image per group. Writes `phash.csv`, `dedup_keep.csv`.
2. **`split`** — stratified 70/10/20 split on (class × source), grouped by
   `lesion_id` so the same lesion never crosses splits. Writes `split.csv`.
3. **`build_memmap`** — decode every kept image once, apply Shades-of-Gray
   color constancy, short-side-resize + center-crop to 224×224, store the
   whole dataset as a single uint8 `.npy` we mmap at training time. Writes
   `images.uint8.npy` (~3.6 GB), `labels.int64.npy`, `meta.npz`,
   `memmap_index.csv`.

Each step can be re-run individually with `python -m preprocessing.<step>` or
skipped via `--skip <step>`.

## EDA

```powershell
jupyter notebook notebooks/01_eda.ipynb
```

The notebook covers: class distribution, source distribution, class×source
crosstab, sample image grids per class, metadata missingness, and image-size
statistics. The last cell loads the memmap if preprocessing has been run.

## Why a memmap?

Training on Windows must use `num_workers=0` (the PyTorch DataLoader has
well-known deadlocks with `num_workers>0` on Windows + CUDA + certain image
libraries). To compensate we eliminate per-batch JPEG decode entirely: the
whole training set is decoded once into a contiguous uint8 array, after which
batches are a memcpy from RAM (the 64 GB host RAM keeps the whole tensor
resident). This is the single most important Windows-specific optimization
and roughly doubles throughput vs. on-the-fly decoding with `num_workers=0`.
