"""
Train the YOLO meat detector.

Expects a YOLO-format dataset (images + .txt label files, plus a dataset.yaml
describing the class list and train/val splits). Roboflow Universe/Annotate
can export directly to this format -- see README for dataset options.

Example data/dataset_yolo/dataset.yaml:

    path: data/dataset_yolo
    train: images/train
    val: images/val
    names:
      0: meat

Usage:
    python scripts/train_yolo.py --data data/dataset_yolo/dataset.yaml --epochs 100
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset_yolo/dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt",
                         help="base checkpoint to fine-tune from (nano is fastest for a conveyor camera)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default="models/yolo_meat_detector.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/detect",
        name="meat_detector",
    )

    best_weights = results.save_dir / "weights" / "best.pt"
    print(f"Training complete. Best weights at: {best_weights}")
    print(f"Copy or symlink this to {args.out} for use in config.yaml.")


if __name__ == "__main__":
    main()
