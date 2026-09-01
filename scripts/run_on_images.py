"""
Test-mode pipeline runner: processes a folder of images without needing a
live camera or a trained YOLO detector.

For each image it:
  1. Treats the full image as the "meat crop" (no detector required).
  2. Runs the freshness classifier → good / spoiled + confidence.
  3. If good, runs size classification based on image dimensions.
  4. Draws a coloured bounding box + labels and saves the annotated image.
  5. Logs every decision to a CSV.

Usage:
    # Quick test on the validation set after training:
    python scripts/run_on_images.py --images data/dataset_freshness/val

    # Custom folder, custom config:
    python scripts/run_on_images.py --images my_photos/ --config config.yaml

    # Show each image on screen as it's processed (press any key to continue):
    python scripts/run_on_images.py --images data/dataset_freshness/val --show
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.freshness_classifier import FreshnessClassifier
from src.size_classifier import SizeClassifier, SizeResult

BOX_COLOR_GOOD    = (60, 180, 75)     # BGR green
BOX_COLOR_SPOILED = (40,  40, 220)    # BGR red
FONT              = cv2.FONT_HERSHEY_SIMPLEX
IMG_EXTS          = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(source: str) -> list[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(p.rglob(f"*{ext}"))
        imgs.extend(p.rglob(f"*{ext.upper()}"))
    return sorted(set(imgs))


def draw_box(frame, label_lines: list[str], color: tuple) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (2, 2), (w - 2, h - 2), color, 3)
    y = 30
    for line in label_lines:
        cv2.putText(frame, line, (10, y), FONT, 0.7, color, 2)
        y += 30


def run(cfg: dict, images_path: str, show: bool, out_dir: Path, csv_path: Path) -> None:
    # --- Load models ---
    weights_fc = cfg["freshness_classifier"]["weights"]
    if not os.path.exists(weights_fc):
        print(
            f"\n[ERROR] Freshness classifier weights not found: {weights_fc}\n"
            "Train first with:\n"
            "  python scripts/train_classifier.py --data data/dataset_freshness --epochs 15\n"
        )
        sys.exit(1)

    freshness_clf = FreshnessClassifier(
        weights_path=weights_fc,
        input_size=cfg["freshness_classifier"]["input_size"],
        device=cfg["freshness_classifier"]["device"],
        class_names=cfg["freshness_classifier"]["class_names"],
        good_confidence_threshold=cfg["freshness_classifier"]["good_confidence_threshold"],
    )
    size_clf = SizeClassifier(
        pixels_per_mm=cfg["size_classifier"]["pixels_per_mm"],
        size_threshold_mm=cfg["size_classifier"]["size_threshold_mm"],
    )

    images = collect_images(images_path)
    if not images:
        print(f"[ERROR] No images found in: {images_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV setup ---
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "freshness_label", "good_conf", "spoiled_conf",
                     "size_category", "longest_dim_mm", "routing"])

    print(f"\nProcessing {len(images)} images...")
    print(f"Output → {out_dir}")
    print(f"Log    → {csv_path}\n")

    counts = {"good": 0, "spoiled": 0, "grinding": 0, "packing": 0}

    for i, img_path in enumerate(images, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [WARN] Could not read {img_path.name} — skipping.")
            continue

        # Use whole frame as the meat crop (no detector needed)
        freshness = freshness_clf.classify(frame)

        if freshness.label == "spoiled":
            routing = "discard"
            size_cat = "-"
            longest_mm = 0.0
            label_lines = [
                f"SPOILED  {freshness.spoiled_confidence * 100:.1f}%",
                "→ DISCARD",
            ]
            color = BOX_COLOR_SPOILED
            counts["spoiled"] += 1
        else:
            # Fake a Detection-like object for the size classifier
            h, w = frame.shape[:2]

            class _FakeDetection:
                width = w
                height = h

            size: SizeResult = size_clf.classify(_FakeDetection())
            routing = "grinding" if size.category == "small" else "packing"
            size_cat = size.category
            longest_mm = size.longest_dimension_mm
            label_lines = [
                f"GOOD  {freshness.good_confidence * 100:.1f}%",
                f"{size.category.upper()} ({longest_mm:.0f} mm) → {routing.upper()}",
            ]
            color = BOX_COLOR_GOOD
            counts["good"] += 1
            counts[routing] += 1

        draw_box(frame, label_lines, color)

        out_file = out_dir / f"annotated_{img_path.stem}.jpg"
        cv2.imwrite(str(out_file), frame)

        writer.writerow([
            img_path.name,
            freshness.label,
            f"{freshness.good_confidence:.4f}",
            f"{freshness.spoiled_confidence:.4f}",
            size_cat,
            f"{longest_mm:.1f}",
            routing,
        ])

        if show:
            cv2.imshow("Meat QC — Test Mode (any key = next, q = quit)", frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

        if i % 50 == 0 or i == len(images):
            print(f"  [{i:4d}/{len(images)}]  good={counts['good']}  "
                  f"spoiled={counts['spoiled']}  "
                  f"grinding={counts['grinding']}  packing={counts['packing']}", end="\r")

    csv_file.close()
    if show:
        cv2.destroyAllWindows()

    total = counts["good"] + counts["spoiled"]
    print(f"\n\n{'='*50}")
    print(f"  Done! Processed {total} images.")
    print(f"  Good:     {counts['good']}  ({counts['good']/max(total,1)*100:.1f}%)")
    print(f"  Spoiled:  {counts['spoiled']}  ({counts['spoiled']/max(total,1)*100:.1f}%)")
    print(f"  → Grinding (small):  {counts['grinding']}")
    print(f"  → Packing  (big):    {counts['packing']}")
    print(f"{'='*50}")
    print(f"  Annotated images: {out_dir}")
    print(f"  Decision log CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run meat QC pipeline on a folder of images.")
    parser.add_argument("--images", required=True,
                        help="Folder of images (or single image path)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--show", action="store_true",
                        help="Display each annotated image on screen")
    parser.add_argument("--out-dir", default="output/test_annotated",
                        help="Output directory for annotated images")
    parser.add_argument("--log-csv", default="output/test_decisions_log.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg, args.images, args.show, Path(args.out_dir), Path(args.log_csv))


if __name__ == "__main__":
    main()
