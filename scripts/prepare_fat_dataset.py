"""
prepare_fat_dataset.py — Dataset preparation for Module 3 fat segmenter.

Validates image/mask pairs, computes per-image fat coverage statistics,
performs a stratified 80/20 train/val split, and copies files into the
expected directory structure for train_fat_segmenter.py.

Usage
-----
    python scripts/prepare_fat_dataset.py --data data/dataset_fat

    # Preview without copying files
    python scripts/prepare_fat_dataset.py --data data/dataset_fat --dry-run

Output structure produced
-------------------------
    data/dataset_fat/
    ├── train/
    │   ├── images/   <- 80% of annotated pairs
    │   └── masks/
    └── val/
        ├── images/   <- 20% of annotated pairs
        └── masks/

Mask encoding expected
----------------------
    - 8-bit grayscale PNG
    - Pixel values strictly in {0, 1, 2}
        0 = background
        1 = lean_meat
        2 = visible_fat
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


LOW_THRESHOLD  = 15.0   # below this  = low
HIGH_THRESHOLD = 40.0   # above this  = high


def compute_fat_coverage(mask: np.ndarray) -> float:
    fat_px  = int(np.sum(mask == 2))
    lean_px = int(np.sum(mask == 1))
    total   = fat_px + lean_px
    return fat_px / total * 100.0 if total > 0 else 0.0


def fat_bucket(pct: float) -> str:
    if pct >= HIGH_THRESHOLD:
        return "high"
    if pct >= LOW_THRESHOLD:
        return "moderate"
    return "low"


def validate_mask(mask_path: Path) -> list:
    errors = []
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        errors.append("Could not read mask file")
        return errors
    unique = set(np.unique(mask).tolist())
    invalid = unique - {0, 1, 2}
    if invalid:
        errors.append(f"Invalid pixel values {sorted(invalid)} (allowed: 0, 1, 2)")
    return errors


def copy_pair(img_path, mask_path, dest_img_dir, dest_mask_dir, dry_run):
    if dry_run:
        print(f"  [DRY-RUN] {img_path.name} -> {dest_img_dir.name}/")
        return
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_mask_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path,  dest_img_dir  / img_path.name)
    shutil.copy2(mask_path, dest_mask_dir / mask_path.name)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the fat segmentation dataset for Module 3 training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",      required=True,  help="Dataset root (must contain images/ and masks/)")
    parser.add_argument("--val-split", type=float, default=0.20, help="Fraction of data for validation")
    parser.add_argument("--seed",      type=int,   default=42,   help="Random seed for reproducible split")
    parser.add_argument("--dry-run",   action="store_true",      help="Print actions without copying any files")
    args = parser.parse_args()

    data_root  = Path(args.data)
    images_dir = data_root / "images"
    masks_dir  = data_root / "masks"

    if not images_dir.is_dir():
        print(f"ERROR: images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)
    if not masks_dir.is_dir():
        print(f"ERROR: masks directory not found: {masks_dir}", file=sys.stderr)
        sys.exit(1)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not image_files:
        print(f"ERROR: No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(image_files)} images in {images_dir}")

    # Validate pairs and masks
    valid_pairs   = []
    missing_masks = []
    mask_errors   = []

    for img_path in image_files:
        mask_path = masks_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            missing_masks.append(img_path.name)
            continue
        errors = validate_mask(mask_path)
        if errors:
            for err in errors:
                mask_errors.append(f"  {mask_path.name}: {err}")
            continue
        valid_pairs.append((img_path, mask_path))

    if missing_masks:
        print(f"\nWARNING: {len(missing_masks)} images are missing a mask:")
        for name in missing_masks[:10]:
            print(f"  {name}")
        if len(missing_masks) > 10:
            print(f"  ... and {len(missing_masks) - 10} more")

    if mask_errors:
        print(f"\nWARNING: {len(mask_errors)} mask validation errors:")
        for err in mask_errors[:10]:
            print(err)
        if len(mask_errors) > 10:
            print(f"  ... and {len(mask_errors) - 10} more")

    if not valid_pairs:
        print("\nERROR: No valid image/mask pairs found.", file=sys.stderr)
        sys.exit(1)

    print(f"{len(valid_pairs)} valid image/mask pairs will be split.")

    # Compute fat coverage and assign buckets
    print("\nComputing fat coverage per image...")
    buckets = {"low": [], "moderate": [], "high": []}
    coverage_values = []

    for img_path, mask_path in valid_pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        pct  = compute_fat_coverage(mask)
        coverage_values.append(pct)
        buckets[fat_bucket(pct)].append((img_path, mask_path))

    print("\nDataset fat coverage distribution:")
    for bname, pairs in buckets.items():
        pct_label = {"low": "<15%", "moderate": "15-40%", "high": ">40%"}[bname]
        print(f"  {bname:10s} ({pct_label}): {len(pairs):4d} images")
    print(f"\n  Mean fat coverage:   {np.mean(coverage_values):.1f}%")
    print(f"  Median fat coverage: {np.median(coverage_values):.1f}%")

    # Stratified train/val split
    rng = random.Random(args.seed)
    train_pairs = []
    val_pairs   = []

    for bucket_name, pairs in buckets.items():
        rng.shuffle(pairs)
        n_val   = max(1, round(len(pairs) * args.val_split)) if len(pairs) > 1 else 0
        n_train = len(pairs) - n_val
        train_pairs.extend(pairs[:n_train])
        val_pairs.extend(pairs[n_train:])

    print(f"\nSplit -> train: {len(train_pairs)} | val: {len(val_pairs)}")

    # Per-class pixel balance on train set
    print("\nComputing per-class pixel balance (train set)...")
    class_counts = {0: 0, 1: 0, 2: 0}
    for _, mask_path in train_pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        for cls in (0, 1, 2):
            class_counts[cls] += int(np.sum(mask == cls))

    total_px   = sum(class_counts.values()) or 1
    class_names = {0: "background", 1: "lean_meat", 2: "visible_fat"}
    print("  Class pixel balance (train set):")
    for cls_id, name in class_names.items():
        pct_px = class_counts[cls_id] / total_px * 100
        print(f"    [{cls_id}] {name:15s}: {class_counts[cls_id]:>10,} px  ({pct_px:.1f}%)")

    # Copy files
    train_img_dir  = data_root / "train" / "images"
    train_mask_dir = data_root / "train" / "masks"
    val_img_dir    = data_root / "val"   / "images"
    val_mask_dir   = data_root / "val"   / "masks"

    action = "Would copy" if args.dry_run else "Copying"
    print(f"\n{action} train pairs...")
    for img_path, mask_path in train_pairs:
        copy_pair(img_path, mask_path, train_img_dir, train_mask_dir, args.dry_run)

    print(f"{action} val pairs...")
    for img_path, mask_path in val_pairs:
        copy_pair(img_path, mask_path, val_img_dir, val_mask_dir, args.dry_run)

    print("\n" + "-" * 60)
    if args.dry_run:
        print("DRY-RUN complete -- no files were copied.")
    else:
        print("Dataset preparation complete.")
        print(f"  Train: {train_img_dir}")
        print(f"  Val:   {val_img_dir}")

    print(f"\nNext step:")
    print(f"  python scripts/train_fat_segmenter.py --data {data_root}")


if __name__ == "__main__":
    main()
