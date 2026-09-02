"""
Validation runner: processes a folder of images and evaluates freshness
classification accuracy.

For each image it:
  1. Runs the freshness classifier → good / spoiled + confidence.
  2. Draws a coloured bounding box + label and saves the annotated image.
  3. Logs every decision to a CSV.

Usage:
    # Quick test on the validation set after training:
    python scripts/run_on_images.py --images data/dataset_freshness/val

    # Custom folder, custom config:
    python scripts/run_on_images.py --images my_photos/ --config backend/config.yaml

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

# Import FreshnessClassifier from the backend package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from freshness_classifier import FreshnessClassifier  # noqa: E402

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
        y += 28


def run(cfg: dict, image_source: str, show: bool,
        out_dir: Path, csv_path: Path, config_dir: Path) -> None:
    images = collect_images(image_source)
    if not images:
        raise FileNotFoundError(f"No images found in: {image_source}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fc = cfg["freshness_classifier"]
    weights = fc["weights"]
    # Resolve relative weight paths against the config file's directory
    if not os.path.isabs(weights):
        weights = str((config_dir / weights).resolve())

    clf = FreshnessClassifier(
        weights_path=weights,
        input_size=fc.get("input_size", 224),
        device=fc.get("device", "cuda"),
        class_names=fc.get("class_names", ("good", "spoiled")),
        good_confidence_threshold=fc.get("good_confidence_threshold", 0.60),
    )

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "freshness_label", "good_conf", "spoiled_conf"])

    counts = {"good": 0, "spoiled": 0}

    print(f"\nProcessing {len(images)} images from: {image_source}")
    print(f"Output dir: {out_dir}")
    print(f"CSV log:    {csv_path}\n")

    t0 = time.time()

    for i, img_path in enumerate(images, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [WARN] Could not read: {img_path}")
            continue

        freshness = clf.classify(frame)

        if freshness.label == "spoiled":
            label_lines = [
                f"SPOILED  {freshness.spoiled_confidence * 100:.1f}%",
            ]
            color = BOX_COLOR_SPOILED
            counts["spoiled"] += 1
        else:
            label_lines = [
                f"GOOD  {freshness.good_confidence * 100:.1f}%",
            ]
            color = BOX_COLOR_GOOD
            counts["good"] += 1

        draw_box(frame, label_lines, color)

        out_file = out_dir / f"annotated_{img_path.stem}.jpg"
        cv2.imwrite(str(out_file), frame)

        writer.writerow([
            img_path.name,
            freshness.label,
            f"{freshness.good_confidence:.4f}",
            f"{freshness.spoiled_confidence:.4f}",
        ])

        if show:
            cv2.imshow("Meat QC — Test Mode (any key = next, q = quit)", frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

        if i % 50 == 0 or i == len(images):
            print(f"  [{i:4d}/{len(images)}]  good={counts['good']}  "
                  f"spoiled={counts['spoiled']}", end="\r")

    csv_file.close()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    total = counts["good"] + counts["spoiled"]
    print(f"\n\n{'='*50}")
    print(f"  Done! Processed {total} images in {elapsed:.1f}s.")
    print(f"  Good:     {counts['good']}  ({counts['good']/max(total,1)*100:.1f}%)")
    print(f"  Spoiled:  {counts['spoiled']}  ({counts['spoiled']/max(total,1)*100:.1f}%)")
    print(f"{'='*50}")
    print(f"  Annotated images: {out_dir}")
    print(f"  Decision log CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run freshness classifier on a folder of images.")
    parser.add_argument("--images", required=True,
                        help="Folder of images (or single image path)")
    parser.add_argument("--config", default="backend/config.yaml",
                        help="Path to config.yaml (default: backend/config.yaml)")
    parser.add_argument("--show", action="store_true",
                        help="Display each annotated image on screen")
    parser.add_argument("--out-dir", default="output/test_annotated",
                        help="Output directory for annotated images")
    parser.add_argument("--log-csv", default="output/test_decisions_log.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    config_dir = Path(args.config).resolve().parent
    run(cfg, args.images, args.show, Path(args.out_dir), Path(args.log_csv), config_dir)


if __name__ == "__main__":
    main()
