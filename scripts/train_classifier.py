"""
Train the freshness (good vs spoiled) classifier.

Expects a folder-per-class dataset (torchvision.datasets.ImageFolder format):

    data/dataset_freshness/
      train/
        good/     *.jpg
        spoiled/  *.jpg
      val/
        good/     *.jpg
        spoiled/  *.jpg

If you download the Kaggle "Meat Freshness Image Dataset" (Fresh/Half-Fresh/
Spoiled), fold "Half-Fresh" into whichever class matches your quality bar
(commonly "spoiled", since it's already past ideal use), or keep it separate
and extend class_names/config.yaml to a 3-class setup.

Usage:
    python scripts/train_classifier.py --data data/dataset_freshness --epochs 15
"""

import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from freshness_classifier import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset_freshness")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--out", default="models/freshness_classifier.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(f"{args.data}/train", transform=train_transform)
    val_ds = datasets.ImageFolder(f"{args.data}/val", transform=val_transform)
    print(f"Classes (index order matters -- must match config.yaml class_names): {train_ds.classes}")

    # num_workers=0 is required on Windows (avoids multiprocessing spawn errors).
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_model(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / len(val_ds)

        print(f"Epoch {epoch + 1}/{args.epochs} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  Saved new best model ({val_acc:.4f}) to {args.out}")

    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    # The if-guard is required on Windows so DataLoader workers don't
    # recursively re-import and re-execute the training loop.
    main()
