"""
Prepare the freshness dataset from the Kaggle 'Meat Freshness Image Dataset' zip.

Expected zip structure (archive (2).zip):
    Meat Freshness.v1-new-dataset.multiclass/
        train/
            FRESH-*.jpg
            HALF-FRESH-*.jpg
            SPOILED-*.jpg
        valid/
            FRESH-*.jpg
            HALF-FRESH-*.jpg
            SPOILED-*.jpg

Output structure (for train_classifier.py):
    data/dataset_freshness/
        train/
            good/       <-- FRESH images
            spoiled/    <-- HALF-FRESH + SPOILED images (conservative: discard borderline)
        val/
            good/
            spoiled/

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --zip "C:/Users/ss/Downloads/archive (2).zip"
    python scripts/prepare_dataset.py --zip "C:/Users/ss/Downloads/archive (2).zip" --half-fresh good
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path


# Default zip path (Kaggle archive (2).zip = Meat Freshness v1)
DEFAULT_ZIP = r"C:\Users\ss\Downloads\archive (2).zip"
OUT_DIR = Path("data/dataset_freshness")

# Filename prefix -> class mapping
# HALF-FRESH goes into "spoiled" by default (conservative food-safety choice).
# Pass --half-fresh good to change this.
PREFIX_TO_CLASS_CONSERVATIVE = {
    "FRESH":      "good",
    "HALF-FRESH": "spoiled",
    "SPOILED":    "spoiled",
}
PREFIX_TO_CLASS_OPTIMISTIC = {
    "FRESH":      "good",
    "HALF-FRESH": "good",
    "SPOILED":    "spoiled",
}

SPLIT_MAP = {
    "train": "train",
    "valid": "val",   # Kaggle uses "valid", we standardise to "val"
}


def classify_filename(name: str, prefix_map: dict) -> str | None:
    """Return 'good' or 'spoiled' based on the filename prefix, or None if unknown."""
    upper = name.upper()
    for prefix, cls in prefix_map.items():
        if upper.startswith(prefix):
            return cls
    return None


def prepare(zip_path: str, out_dir: Path, half_fresh: str) -> None:
    prefix_map = (
        PREFIX_TO_CLASS_OPTIMISTIC
        if half_fresh == "good"
        else PREFIX_TO_CLASS_CONSERVATIVE
    )

    print(f"HALF-FRESH will be treated as: {prefix_map['HALF-FRESH'].upper()}")
    print(f"Reading zip: {zip_path}")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Zip not found at: {zip_path}\n"
            "Pass the correct path with --zip. E.g.:\n"
            '  python scripts/prepare_dataset.py --zip "C:/Users/YourName/Downloads/archive (2).zip"'
        )

    counts = {split: {"good": 0, "spoiled": 0} for split in ("train", "val")}

    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = zf.namelist()
        image_entries = [e for e in entries if e.lower().endswith((".jpg", ".jpeg", ".png"))]
        total = len(image_entries)
        print(f"Found {total} images in zip.")

        for i, entry in enumerate(image_entries, 1):
            parts = entry.split("/")
            # Expected: [root_folder, split_folder, filename]
            if len(parts) < 3:
                continue

            split_raw = parts[-2]   # "train" or "valid"
            filename   = parts[-1]

            split = SPLIT_MAP.get(split_raw)
            if split is None:
                continue  # skip unexpected folders

            cls = classify_filename(filename, prefix_map)
            if cls is None:
                print(f"  [WARN] Could not classify: {filename} — skipping.")
                continue

            dest_dir = out_dir / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / filename

            if not dest_file.exists():
                with zf.open(entry) as src, open(dest_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            counts[split][cls] += 1

            if i % 200 == 0 or i == total:
                print(f"  [{i}/{total}] processed...", end="\r")

    print("\n\nDataset ready!")
    print("=" * 45)
    for split in ("train", "val"):
        g = counts[split]["good"]
        s = counts[split]["spoiled"]
        print(f"  {split:5s} | good={g:4d}  spoiled={s:4d}  total={g+s:4d}")
    print("=" * 45)
    print(f"Output directory: {out_dir.resolve()}")
    print("\nNext step:")
    print("  python scripts/train_classifier.py --data data/dataset_freshness --epochs 15")


def main():
    parser = argparse.ArgumentParser(description="Prepare Meat Freshness dataset.")
    parser.add_argument(
        "--zip", default=DEFAULT_ZIP,
        help="Path to the Kaggle zip file (archive (2).zip = Meat Freshness v1)"
    )
    parser.add_argument(
        "--out", default=str(OUT_DIR),
        help="Output directory for the prepared dataset"
    )
    parser.add_argument(
        "--half-fresh", choices=["good", "spoiled"], default="spoiled",
        help="Treat HALF-FRESH images as 'good' or 'spoiled' (default: spoiled)"
    )
    args = parser.parse_args()
    prepare(args.zip, Path(args.out), args.half_fresh)


if __name__ == "__main__":
    main()
