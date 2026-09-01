"""
Merge an additional labeled dataset into the existing train/val split.

This script solves the "domain shift" problem: if your model performs well
on Kaggle data but poorly on your own camera footage, add your own images
using this script and retrain.

Your additional data must be organised into two folders:
  <fresh_dir>/    -- images of fresh / good meat
  <spoiled_dir>/  -- images of spoiled meat

The script randomly splits each folder 80% train / 20% val and copies
the images into data/dataset_freshness/{train,val}/{good,spoiled}/.

Usage:
    # Provide your own fresh and spoiled folders:
    python scripts/merge_datasets.py --fresh "C:/MyData/Fresh" --spoiled "C:/MyData/Spoiled"

    # Change the 80/20 split ratio:
    python scripts/merge_datasets.py --fresh "C:/MyData/Fresh" --spoiled "C:/MyData/Spoiled" --val-split 0.15

    # Change the output dataset directory:
    python scripts/merge_datasets.py --fresh "C:/MyData/Fresh" --spoiled "C:/MyData/Spoiled" --out data/dataset_freshness
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def get_images(folder: Path) -> list:
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if Path(file).suffix.lower() in valid_exts:
                images.append(os.path.join(root, file))
    return images


def split_and_copy(imgs: list, target_dir: Path, class_name: str,
                   prefix: str, val_split: float) -> None:
    random.shuffle(imgs)
    split_idx = int(len(imgs) * (1 - val_split))
    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:]

    for split, subset in [("train", train_imgs), ("val", val_imgs)]:
        for i, img_path in enumerate(subset):
            dest = target_dir / split / class_name / f"{prefix}_ext_{i}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest)

    print(f"  {class_name:8s} → {len(train_imgs)} train  |  {len(val_imgs)} val")


def merge_data(fresh_dir: Path, spoiled_dir: Path,
               target_dir: Path, val_split: float) -> None:
    if not fresh_dir.exists():
        raise FileNotFoundError(f"Fresh folder not found: {fresh_dir}")
    if not spoiled_dir.exists():
        raise FileNotFoundError(f"Spoiled folder not found: {spoiled_dir}")

    fresh_imgs = get_images(fresh_dir)
    spoiled_imgs = get_images(spoiled_dir)

    if not fresh_imgs:
        raise ValueError(f"No images found in fresh folder: {fresh_dir}")
    if not spoiled_imgs:
        raise ValueError(f"No images found in spoiled folder: {spoiled_dir}")

    print(f"Found {len(fresh_imgs)} fresh  images in: {fresh_dir}")
    print(f"Found {len(spoiled_imgs)} spoiled images in: {spoiled_dir}")
    print(f"Val split: {val_split*100:.0f}%\n")

    random.seed(42)
    split_and_copy(fresh_imgs,   target_dir, "good",    "FRESH",   val_split)
    split_and_copy(spoiled_imgs, target_dir, "spoiled", "SPOILED", val_split)

    print(f"\nDataset merge complete! Output: {target_dir.resolve()}")
    print("\nNext step — retrain the model:")
    print("  python scripts/train_classifier.py --data data/dataset_freshness --epochs 30 --batch 64")


def main():
    parser = argparse.ArgumentParser(
        description="Merge an additional labeled dataset into train/val split."
    )
    parser.add_argument(
        "--fresh", required=True,
        help="Path to folder containing fresh/good meat images"
    )
    parser.add_argument(
        "--spoiled", required=True,
        help="Path to folder containing spoiled meat images"
    )
    parser.add_argument(
        "--out", default="data/dataset_freshness",
        help="Output dataset directory (default: data/dataset_freshness)"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Fraction of images to put in the val set (default: 0.2 = 20%%)"
    )
    args = parser.parse_args()
    merge_data(Path(args.fresh), Path(args.spoiled), Path(args.out), args.val_split)


if __name__ == "__main__":
    main()

