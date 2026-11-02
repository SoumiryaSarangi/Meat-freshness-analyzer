"""
annotate_fat_dataset.py — Automated annotation generator for Module 3 dataset.

Generates ground-truth segmentation masks (0=background, 1=lean_meat, 2=visible_fat)
for meat images to produce the annotated dataset required for training the U-Net.

Produces:
  data/dataset_fat/
  ├── images/     <- 350+ curated meat images (.jpg)
  └── masks/      <- matching 8-bit grayscale masks (.png with values 0, 1, 2)

Usage:
  python scripts/annotate_fat_dataset.py --target-count 360
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def generate_segmentation_mask(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Segment image into background (0), lean meat (1), and visible fat (2).

    Returns:
        (mask_uint8, fat_coverage_pct)
    """
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # 1. Background vs Meat
    # - Dark/black conveyor or shadow: V < 42 or L < 42
    # - Neutral grey tray / conveyor: A between 122 and 133 and S < 38 and L < 165
    # - Glare / high white specular background: L > 240 and S < 25
    is_dark_bg = (V < 42) | (L < 42)
    is_grey_bg = (A >= 122) & (A <= 133) & (S < 38) & (L < 165)
    is_glare_bg = (L > 240) & (S < 25)
    is_bg = is_dark_bg | is_grey_bg | is_glare_bg

    is_meat = ~is_bg
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    is_meat = cv2.morphologyEx(is_meat.astype(np.uint8), cv2.MORPH_OPEN, k5)
    is_meat = cv2.morphologyEx(is_meat, cv2.MORPH_CLOSE, k5) > 0

    # Retain significant connected components (remove floating speckles)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(is_meat.astype(np.uint8))
    if num_labels > 1:
        min_blob = (h * w) * 0.015
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_blob:
                is_meat[labels == lbl] = False

    if is_meat.sum() < 500:
        is_meat = np.ones((h, w), dtype=bool)

    # 2. Within meat: Lean Meat vs Visible Fat
    # Meat pixels analyzed with Otsu on L* channel
    meat_L = L[is_meat]
    if len(meat_L) > 100:
        thresh, _ = cv2.threshold(meat_L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = np.clip(thresh, 140, 175)
    else:
        thresh = 150

    # Multi-condition fat detection:
    cond1 = (L >= thresh) & (A < 147) & (S < 85)
    cond2 = (V > 165) & (S < 55) & (B >= 124)
    cond3 = (L > 180) & (S < 70)
    is_fat = is_meat & (cond1 | cond2 | cond3)

    # Clean fat mask
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    is_fat = cv2.morphologyEx(is_fat.astype(np.uint8), cv2.MORPH_OPEN, k3)
    is_fat = cv2.morphologyEx(is_fat, cv2.MORPH_CLOSE, k3) > 0

    is_lean = is_meat & (~is_fat)

    # Build 3-class mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[is_lean] = 1   # Lean Meat
    mask[is_fat] = 2    # Visible Fat
    # 0 remains Background

    total_meat = int(is_meat.sum())
    fat_px = int(is_fat.sum())
    fat_pct = (fat_px / total_meat * 100.0) if total_meat > 0 else 0.0

    return mask, fat_pct


def main():
    parser = argparse.ArgumentParser(description="Generate labelled dataset for Module 3 fat analysis.")
    parser.add_argument("--sources", nargs="+", default=[
        "data/dataset_freshness/train/good",
        "data/dataset_freshness/val/good"
    ], help="Source directories containing fresh meat images")
    parser.add_argument("--out", default="data/dataset_fat", help="Output directory")
    parser.add_argument("--target-count", type=int, default=360, help="Total pairs to annotate (min 300)")
    args = parser.parse_args()

    out_root = Path(args.out)
    images_out = out_root / "images"
    masks_out = out_root / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    # Collect source images
    source_files = []
    for src in args.sources:
        p = Path(src)
        if p.is_dir():
            source_files.extend(list(p.glob("*.jpg")) + list(p.glob("*.png")))

    source_files = sorted(set(source_files))
    print(f"Collected {len(source_files)} available candidate images.")

    # Categorize into candidate buckets
    print("Screening images into fat coverage buckets (Low <15%, Moderate 15-40%, High >40%)...")
    candidates = {"low": [], "moderate": [], "high": []}

    for idx, f in enumerate(source_files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        mask, fat_pct = generate_segmentation_mask(img)

        # Ensure mask contains all 3 classes (or at least substantial meat)
        unique_classes = set(np.unique(mask).tolist())
        if not ({1, 2}.issubset(unique_classes)):
            continue

        if fat_pct < 15.0:
            candidates["low"].append((f, img, mask, fat_pct))
        elif fat_pct > 40.0:
            candidates["high"].append((f, img, mask, fat_pct))
        else:
            candidates["moderate"].append((f, img, mask, fat_pct))

        # Early exit if we have enough candidates in all buckets
        if len(candidates["low"]) >= 120 and len(candidates["high"]) >= 120 and len(candidates["moderate"]) >= 180:
            print(f"Sufficient candidate pool reached after screening {idx + 1} images.")
            break

    print(f"Found candidates: Low: {len(candidates['low'])}, Moderate: {len(candidates['moderate'])}, High: {len(candidates['high'])}")

    # Target allocations
    # We want at least ~80 low, ~180-200 moderate, ~80-100 high
    n_low = min(len(candidates["low"]), int(args.target_count * 0.28))
    n_high = min(len(candidates["high"]), int(args.target_count * 0.28))
    n_mod = args.target_count - (n_low + n_high)
    n_mod = min(len(candidates["moderate"]), n_mod)

    selected = (
        candidates["low"][:n_low] +
        candidates["moderate"][:n_mod] +
        candidates["high"][:n_high]
    )

    print(f"Selected {len(selected)} images for annotation (Low: {n_low}, Moderate: {n_mod}, High: {n_high}).")

    manifest = []
    print("Writing images and masks to data/dataset_fat/...")

    for f_path, img, mask, fat_pct in selected:
        out_img_name = f_path.name
        out_mask_name = f_path.stem + ".png"

        dest_img_path = images_out / out_img_name
        dest_mask_path = masks_out / out_mask_name

        # Copy original image
        shutil.copy2(f_path, dest_img_path)

        # Save 8-bit single-channel mask strictly with values 0, 1, 2
        cv2.imwrite(str(dest_mask_path), mask.astype(np.uint8))

        total_meat = int(np.sum(mask == 1) + np.sum(mask == 2))
        fat_px = int(np.sum(mask == 2))
        lean_px = int(np.sum(mask == 1))
        bg_px = int(np.sum(mask == 0))

        cat = "low" if fat_pct < 15.0 else ("high" if fat_pct > 40.0 else "moderate")

        manifest.append({
            "image": str(dest_img_path.relative_to(out_root.parent)),
            "mask": str(dest_mask_path.relative_to(out_root.parent)),
            "filename": out_img_name,
            "fat_coverage_pct": round(fat_pct, 2),
            "category": cat,
            "fat_pixels": fat_px,
            "lean_pixels": lean_px,
            "background_pixels": bg_px,
            "total_meat_pixels": total_meat,
        })

    manifest_path = out_root / "annotation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAnnotation generation complete!")
    print(f"  Total images saved: {len(manifest)} in {images_out}")
    print(f"  Total masks saved:  {len(manifest)} in {masks_out}")
    print(f"  Manifest:           {manifest_path}")

    # Summary table
    print("\nDataset Composition Summary:")
    print(f"  {'Category':<12} | {'Count':<6} | {'Coverage Range':<15}")
    print("  " + "-" * 40)
    for c in ["low", "moderate", "high"]:
        matching = [m for m in manifest if m["category"] == c]
        if matching:
            min_cov = min(m["fat_coverage_pct"] for m in matching)
            max_cov = max(m["fat_coverage_pct"] for m in matching)
            print(f"  {c:<12} | {len(matching):<6} | {min_cov:.1f}% - {max_cov:.1f}%")

    print("\nNext step: prepare and split the dataset:")
    print(f"  python scripts/prepare_fat_dataset.py --data {out_root}")


if __name__ == "__main__":
    main()

