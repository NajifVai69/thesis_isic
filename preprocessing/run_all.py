"""End-to-end preprocessing orchestrator.

Runs, in order:
  1. dedup_phash.py   -> phash.csv, dedup_keep.csv
  2. split.py         -> split.csv
  3. build_memmap.py  -> images.uint8.npy, labels.int64.npy, meta.npz, memmap_index.csv

Each step writes its own artifacts and can be re-run independently. We invoke
them as subprocesses so each step starts with a clean import + a clean Python
process — important on Windows where some C libs (cv2, torch) don't always play
nicely if they're imported together repeatedly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(step: str, config: str) -> None:
    cmd = [sys.executable, "-m", f"preprocessing.{step}", "--config", config]
    print("\n" + "=" * 72)
    print(f"[run] {' '.join(cmd)}")
    print("=" * 72)
    r = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]))
    if r.returncode != 0:
        sys.exit(f"[fail] step {step!r} exited with {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["dedup_phash", "split", "build_memmap"],
        help="Steps to skip (artifacts must already exist).",
    )
    args = ap.parse_args()

    for step in ("dedup_phash", "split", "build_memmap"):
        if step in args.skip:
            print(f"[skip] {step}")
            continue
        run(step, args.config)

    print("\n[ok] preprocessing complete.")


if __name__ == "__main__":
    main()
