"""
evaluate_fat_segmenter.py — Evaluation script for Module 3 fat segmenter.

Computes on the val set:
  - Per-class IoU (background, lean_meat, visible_fat)
  - Mean IoU (mIoU)
  - Per-class Dice coefficient
  - Pixel accuracy
  - Fat coverage MAE: |predicted_fat_pct - ground_truth_fat_pct|

Saves the 5 worst predictions as side-by-side comparison images.

Usage
-----
    python scripts/evaluate_fat_segmenter.py \\
        --data  data/dataset_fat \\
        --weights models/fat_segmenter.pt \\
        --out-dir output/fat_eval
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow imports from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

try:
    from fat_analyser import FatAnalyser, _IMAGENET_MEAN, _IMAGENET_STD
except ImportError as e:
    print(f"ERROR: Could not import fat_analyser: {e}", file=sys.stderr)
    sys.exit(1)

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou_per_class(preds, labels, num_classes=3):
    ious = np.full(num_classes, np.nan, dtype=np.float64)
    for cls in range(num_classes):
        inter = np.logical_and(preds == cls, labels == cls).sum()
        union = np.logical_or( preds == cls, labels == cls).sum()
        if union > 0:
            ious[cls] = float(inter / union)
    return ious


def compute_dice_per_class(preds, labels, num_classes=3):
    dices = np.full(num_classes, np.nan, dtype=np.float64)
    for cls in range(num_classes):
        inter = np.logical_and(preds == cls, labels == cls).sum()
        denom = (preds == cls).sum() + (labels == cls).sum()
        if denom > 0:
            dices[cls] = float(2.0 * inter / denom)
    return dices


def gt_fat_coverage(mask):
    fat_px  = int(np.sum(mask == 2))
    lean_px = int(np.sum(mask == 1))
    total   = fat_px + lean_px
    return fat_px / total * 100.0 if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison(img_bgr, gt_mask, pred_result, save_path):
    """Save original | GT coloured mask | predicted overlay side-by-side."""
    h, w = img_bgr.shape[:2]

    gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gt_vis[gt_mask == 1] = [220, 60,  60]
    gt_vis[gt_mask == 2] = [255, 200, 50]
    gt_vis[gt_mask == 0] = [40,  40,  40]
    gt_vis_bgr = cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR)

    if pred_result.overlay_image_base64:
        b64_data    = pred_result.overlay_image_base64.split(",", 1)[-1]
        from PIL import Image as PILImage
        overlay_pil = PILImage.open(io.BytesIO(base64.b64decode(b64_data)))
        overlay_bgr = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.resize(overlay_bgr, (w, h))
    else:
        overlay_bgr = np.zeros_like(img_bgr)

    div   = np.full((h, 3, 3), 60, dtype=np.uint8)
    combo = np.hstack([img_bgr, div, gt_vis_bgr, div, overlay_bgr])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combo, "Original",   (5,       h - 8), font, 0.5, (200, 200, 200), 1)
    cv2.putText(combo, "GT Mask",    (w + 8,   h - 8), font, 0.5, (200, 200, 200), 1)
    cv2.putText(combo, "AI Overlay", (2*w + 12, h - 8), font, 0.5, (200, 200, 200), 1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), combo)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper (pixel-level predictions for metrics)
# ─────────────────────────────────────────────────────────────────────────────

def predict_mask(model, img_bgr, input_size, device):
    """Run model and return predicted class mask at original image resolution."""
    h, w    = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb     = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor  = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_small = np.argmax(probs, axis=0).astype(np.uint8)
    return cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Module 3 fat segmenter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",       required=True)
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--out-dir",    default="output/fat_eval")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--input-size", type=int, default=256)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    val_images = Path(args.data) / "val" / "images"
    val_masks  = Path(args.data) / "val" / "masks"
    out_dir    = Path(args.out_dir)

    for d in (val_images, val_masks):
        if not d.is_dir():
            print(f"ERROR: {d} not found", file=sys.stderr)
            sys.exit(1)

    print(f"Loading model from {args.weights} on {device} ...")
    analyser = FatAnalyser(
        weights_path=args.weights,
        input_size=args.input_size,
        device=str(device),
    )

    IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in val_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not image_paths:
        print(f"ERROR: No images in {val_images}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(image_paths)} images ...\n")

    all_preds_flat  = []
    all_labels_flat = []
    per_image       = []

    for img_path in image_paths:
        mask_path = val_masks / (img_path.stem + ".png")
        if not mask_path.exists():
            continue

        img_bgr  = cv2.imread(str(img_path))
        gt_mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt_mask is None:
            continue

        h, w        = img_bgr.shape[:2]
        gt_resized  = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        pred_mask   = predict_mask(analyser.model, img_bgr, args.input_size, device)
        meat_mask   = np.where(gt_resized > 0, 255, 0).astype(np.uint8)
        fat_result  = analyser.analyse(img_bgr, meat_mask)

        gt_pct  = gt_fat_coverage(gt_resized)
        fat_mae = abs(fat_result.fat_coverage_pct - gt_pct)

        all_preds_flat.append( pred_mask.ravel())
        all_labels_flat.append(gt_resized.ravel())
        per_image.append({
            "filename":    img_path.name,
            "gt_fat_pct":   round(gt_pct, 2),
            "pred_fat_pct": fat_result.fat_coverage_pct,
            "fat_mae":      round(fat_mae, 2),
            "img_bgr":      img_bgr,
            "gt_mask":      gt_resized,
            "result":       fat_result,
        })

    if not per_image:
        print("ERROR: No images were successfully evaluated.", file=sys.stderr)
        sys.exit(1)

    # Aggregate
    all_preds  = np.concatenate(all_preds_flat)
    all_labels = np.concatenate(all_labels_flat)
    iou        = compute_iou_per_class(all_preds, all_labels)
    dice       = compute_dice_per_class(all_preds, all_labels)
    pix_acc    = float(np.mean(all_preds == all_labels))
    valid_iou  = iou[~np.isnan(iou)]
    miou       = float(np.mean(valid_iou)) if len(valid_iou) > 0 else 0.0
    mean_mae   = float(np.mean([r["fat_mae"] for r in per_image]))

    class_names = ["background", "lean_meat", "visible_fat"]

    print("=" * 55)
    print("Module 3 — Fat Segmenter Evaluation")
    print("=" * 55)
    print(f"\n{'Class':<16}  {'IoU':>7}  {'Dice':>7}")
    print("-" * 34)
    for i, name in enumerate(class_names):
        iou_s  = f"{iou[i]:.4f}"  if not np.isnan(iou[i])  else "   N/A"
        dice_s = f"{dice[i]:.4f}" if not np.isnan(dice[i]) else "   N/A"
        print(f"{name:<16}  {iou_s:>7}  {dice_s:>7}")

    print()
    print(f"  Mean IoU (mIoU):       {miou:.4f}")
    print(f"  Pixel accuracy:        {pix_acc:.4f}")
    print(f"  Fat coverage MAE:      {mean_mae:.2f} percentage points")
    print(f"  Images evaluated:      {len(per_image)}")

    targets = [
        ("mIoU >= 0.70",           miou >= 0.70),
        ("Fat IoU >= 0.65",        (not np.isnan(iou[2])) and float(iou[2]) >= 0.65),
        ("Lean IoU >= 0.75",       (not np.isnan(iou[1])) and float(iou[1]) >= 0.75),
        ("Dice (fat) >= 0.72",     (not np.isnan(dice[2])) and float(dice[2]) >= 0.72),
        ("Fat coverage MAE <= 5pp", mean_mae <= 5.0),
    ]
    print("\nProduction readiness:")
    for label, passed in targets:
        print(f"  {'[PASS]' if passed else '[FAIL]'}  {label}")

    # Save metrics JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "images_evaluated": len(per_image),
        "pixel_accuracy":   round(pix_acc, 4),
        "mean_iou":         round(miou, 4),
        "fat_coverage_mae_percentage_points": round(mean_mae, 2),
        "per_class": {
            name: {
                "iou":  round(float(iou[i]),  4) if not np.isnan(iou[i])  else None,
                "dice": round(float(dice[i]), 4) if not np.isnan(dice[i]) else None,
            }
            for i, name in enumerate(class_names)
        },
    }
    metrics_path = out_dir / "fat_segmentation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # Save worst 5 predictions
    worst     = sorted(per_image, key=lambda x: x["fat_mae"], reverse=True)[:5]
    worst_dir = out_dir / "worst_predictions"
    print(f"\nSaving {len(worst)} worst predictions to {worst_dir}/")
    for rank, row in enumerate(worst, start=1):
        save_path = worst_dir / f"worst_{rank}_{row['filename']}.jpg"
        save_comparison(row["img_bgr"], row["gt_mask"], row["result"], save_path)
        print(f"  #{rank}: {row['filename']:35s}  "
              f"GT={row['gt_fat_pct']:.1f}%  "
              f"Pred={row['pred_fat_pct']:.1f}%  "
              f"MAE={row['fat_mae']:.1f}pp")

    print("\nDone.")


if __name__ == "__main__":
    main()
