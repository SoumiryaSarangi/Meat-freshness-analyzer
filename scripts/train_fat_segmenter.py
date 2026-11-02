"""
train_fat_segmenter.py — Training script for Module 3 fat coverage U-Net.

Trains a U-Net with MobileNetV2 encoder on the annotated fat segmentation
dataset (3 classes: background, lean_meat, visible_fat).

Usage
-----
    # GPU training (recommended)
    python scripts/train_fat_segmenter.py --data data/dataset_fat --epochs 50 --batch 8

    # CPU-only
    python scripts/train_fat_segmenter.py --data data/dataset_fat --epochs 50 --batch 4 --device cpu

Requirements
------------
    pip install segmentation-models-pytorch albumentations

Output
------
    models/fat_segmenter.pt          <- best checkpoint (by val mIoU)
    models/fat_segmenter_final.pt    <- final epoch checkpoint

Dataset structure expected
--------------------------
    data/dataset_fat/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── val/
        ├── images/
        └── masks/
"""

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from fat_analyser import build_unet_model, _IMAGENET_MEAN, _IMAGENET_STD

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    A = None
    ALBUMENTATIONS_AVAILABLE = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FatSegDataset(Dataset):
    """PyTorch Dataset for 3-class fat segmentation.

    Args:
        root:       Path to split directory (e.g. data/dataset_fat/train).
        input_size: Image will be resized to (input_size, input_size).
        augment:    If True, apply training augmentations.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, root, input_size=256, augment=False):
        self.images_dir = Path(root) / "images"
        self.masks_dir  = Path(root) / "masks"
        self.input_size = input_size
        self.augment    = augment

        self.image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in self.IMAGE_EXTS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        missing = [
            p.name for p in self.image_paths
            if not (self.masks_dir / (p.stem + ".png")).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing masks for {len(missing)} images. First missing: {missing[0]}"
            )

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if ALBUMENTATIONS_AVAILABLE:
            resize = A.Resize(input_size, input_size)
            if augment:
                self._transform = A.Compose([
                    resize,
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            else:
                self._transform = A.Compose([
                    resize,
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
        else:
            self._transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path  = self.image_paths[idx]
        mask_path = self.masks_dir / (img_path.stem + ".png")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise IOError(f"Failed to load image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to load mask: {mask_path}")

        if self._transform is not None:
            transformed = self._transform(image=img_rgb, mask=mask)
            image = transformed["image"]
            mask  = torch.from_numpy(transformed["mask"]).long()
        else:
            # Native transforms using OpenCV and numpy
            img_resized  = cv2.resize(img_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

            if self.augment:
                # Random horizontal flip
                if random.random() > 0.5:
                    img_resized  = cv2.flip(img_resized, 1)
                    mask_resized = cv2.flip(mask_resized, 1)
                # Random vertical flip
                if random.random() > 0.5:
                    img_resized  = cv2.flip(img_resized, 0)
                    mask_resized = cv2.flip(mask_resized, 0)
                # Random brightness & contrast
                if random.random() > 0.5:
                    alpha = 1.0 + random.uniform(-0.15, 0.15)
                    beta  = random.uniform(-20, 20)
                    img_resized = np.clip(alpha * img_resized + beta, 0, 255).astype(np.uint8)

            norm_img = (img_resized.astype(np.float32) / 255.0 - self._mean) / self._std
            image = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()
            mask  = torch.from_numpy(mask_resized).long()

        return {"image": image, "mask": mask}


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss averaged over non-background classes (1 and 2)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        dice_scores = []
        for cls in range(1, num_classes):
            prob   = probs[:, cls]
            target = (targets == cls).float()
            inter  = (prob * target).sum()
            denom  = prob.sum() + target.sum()
            dice_scores.append((2.0 * inter + self.smooth) / (denom + self.smooth))
        return 1.0 - torch.stack(dice_scores).mean()


class CombinedLoss(nn.Module):
    """0.5 * CrossEntropyLoss + 0.5 * DiceLoss."""

    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(preds, labels, num_classes=3):
    """Per-class IoU; returns NaN for absent classes."""
    ious = np.full(num_classes, np.nan, dtype=np.float64)
    for cls in range(num_classes):
        inter = np.logical_and(preds == cls, labels == cls).sum()
        union = np.logical_or( preds == cls, labels == cls).sum()
        if union > 0:
            ious[cls] = inter / union
    return ious


def compute_dice(preds, labels, cls):
    pred_c  = preds  == cls
    label_c = labels == cls
    inter   = np.logical_and(pred_c, label_c).sum()
    denom   = pred_c.sum() + label_c.sum()
    return (2.0 * inter / denom) if denom > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Full validation pass — returns (loss, pixel_acc, iou_per_class, fat_dice)."""
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch in loader:
        images  = batch["image"].to(device)
        masks   = batch["mask"].to(device)
        logits  = model(images)
        loss    = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
        all_preds.append( logits.argmax(1).cpu().numpy().ravel())
        all_labels.append(masks.cpu().numpy().ravel())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    mean_loss  = total_loss / (len(loader.dataset) or 1)
    pix_acc    = float(np.mean(all_preds == all_labels))
    iou        = compute_iou(all_preds, all_labels)
    fat_dice   = compute_dice(all_preds, all_labels, cls=2)
    return mean_loss, pix_acc, iou, fat_dice


def train_one_epoch(model, loader, criterion, device, optimizer):
    """One training epoch — returns mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        images  = batch["image"].to(device)
        masks   = batch["mask"].to(device)
        logits  = model(images)
        loss    = criterion(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / (len(loader.dataset) or 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train the Module 3 fat segmentation U-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",       required=True)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch",      type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--input-size", type=int,   default=256)
    parser.add_argument("--out",        default="models/fat_segmenter.pt")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--workers",    type=int,   default=2)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[train] Device: {device}")

    data_root    = Path(args.data)
    train_ds     = FatSegDataset(data_root / "train", args.input_size, augment=True)
    val_ds       = FatSegDataset(data_root / "val",   args.input_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))

    print(f"[train] Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = build_unet_model(
        num_classes=3,
        pretrained=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CombinedLoss()

    out_path   = Path(args.out)
    final_path = out_path.with_stem(out_path.stem + "_final")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_miou  = -1.0
    best_epoch = 0

    print(f"\n{'Epoch':>6}  {'TrLoss':>8}  {'VaLoss':>8}  {'PixAcc':>7}  "
          f"{'mIoU':>6}  {'FatIoU':>7}  {'FatDice':>8}")
    print("-" * 68)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss  = train_one_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, pix_acc, iou, fat_dice = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        valid_iou = iou[~np.isnan(iou)]
        miou      = float(np.mean(valid_iou)) if len(valid_iou) > 0 else 0.0
        fat_iou   = float(iou[2]) if not np.isnan(iou[2]) else 0.0

        print(f"{epoch:>6d}  {tr_loss:>8.4f}  {va_loss:>8.4f}  {pix_acc:>7.4f}  "
              f"{miou:>6.4f}  {fat_iou:>7.4f}  {fat_dice:>8.4f}  ({time.time()-t0:.0f}s)")

        if miou > best_miou:
            best_miou  = miou
            best_epoch = epoch
            torch.save(model.state_dict(), out_path)
            print(f"         -> Best checkpoint saved (mIoU={best_miou:.4f})")

    torch.save(model.state_dict(), final_path)

    # Final per-class report
    model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
    _, pix_acc, iou, _ = evaluate(model, val_loader, criterion, device)

    # Gather all preds+labels for final dice
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for batch in val_loader:
            p = model(batch["image"].to(device)).argmax(1).cpu().numpy().ravel()
            l = batch["mask"].numpy().ravel()
            all_p.append(p); all_l.append(l)
    all_p = np.concatenate(all_p); all_l = np.concatenate(all_l)

    class_names = ["background", "lean_meat", "visible_fat"]
    print(f"\nBest epoch: {best_epoch}  |  Val mIoU: {best_miou:.4f}")
    print(f"\n{'Class':<16}  {'IoU':>7}  {'Dice':>7}")
    print("-" * 34)
    for cls_id, name in enumerate(class_names):
        iou_s  = f"{iou[cls_id]:.4f}" if not np.isnan(iou[cls_id]) else "   N/A"
        dice_s = f"{compute_dice(all_p, all_l, cls_id):.4f}"
        print(f"{name:<16}  {iou_s:>7}  {dice_s:>7}")
    print(f"\nPixel accuracy: {pix_acc:.4f}")
    print(f"\nSaved:  {out_path}  (best)  |  {final_path}  (final)")

    print("\nEnable fat analysis in backend/config.yaml:")
    print("  fat_analyser:")
    print("    enabled: true")


if __name__ == "__main__":
    main()
