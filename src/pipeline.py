"""
Real-time meat QC pipeline.

Flow per frame:
  1. Read frame from camera/conveyor feed.
  2. YOLO detector -> bounding box + confidence per meat piece.
  3. Centroid tracker -> stable ID per piece, so each physical piece is only
     classified/routed once as it crosses the frame.
  4. On first sighting of a piece: crop the bounding box, run the freshness
     CNN -> good/spoiled + confidence.
       - spoiled -> route to "discard", stop here for this piece.
       - good    -> run the size classifier on the same bounding box ->
                    small/big -> route to "grinding" or "packing".
  5. Draw the bounding box + confidence score on the frame for monitoring.

Run with: python -m src.pipeline --config config.yaml
"""

import argparse
import os
import time

import cv2
import yaml

from .detector import MeatDetector
from .freshness_classifier import FreshnessClassifier
from .size_classifier import SizeClassifier
from .tracker import CentroidTracker
from .router import build_router

BOX_COLOR_GOOD = (60, 180, 75)     # BGR: green
BOX_COLOR_SPOILED = (40, 40, 220)  # BGR: red


def draw_annotation(frame, detection, label_lines, color):
    cv2.rectangle(frame, (detection.x1, detection.y1),
                  (detection.x2, detection.y2), color, 2)
    y = max(detection.y1 - 10, 20)
    for line in label_lines:
        cv2.putText(frame, line, (detection.x1, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y -= 22


def _check_weights(cfg: dict) -> None:
    """Fail fast with a helpful message if trained weights are missing."""
    issues = []
    fc_path  = cfg["freshness_classifier"]["weights"]
    det_path = cfg["detector"]["weights"]
    if not os.path.exists(fc_path):
        issues.append(
            f"  Freshness classifier: {fc_path}\n"
            "    Train with: python scripts/train_classifier.py "
            "--data data/dataset_freshness --epochs 15"
        )
    if not os.path.exists(det_path):
        issues.append(
            f"  YOLO detector: {det_path}\n"
            "    Train with: python scripts/train_yolo.py --data data/dataset_yolo/dataset.yaml\n"
            "    (Or test without a detector using: python scripts/run_on_images.py --images <folder>)"
        )
    if issues:
        print("\n[ERROR] Missing model weights — train these first:\n")
        for msg in issues:
            print(msg)
        raise SystemExit(1)


def run(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    _check_weights(cfg)

    detector = MeatDetector(
        weights_path=cfg["detector"]["weights"],
        confidence_threshold=cfg["detector"]["confidence_threshold"],
        iou_threshold=cfg["detector"]["iou_threshold"],
        device=cfg["detector"]["device"],
        target_class=cfg["detector"]["target_class"],
    )
    freshness_clf = FreshnessClassifier(
        weights_path=cfg["freshness_classifier"]["weights"],
        input_size=cfg["freshness_classifier"]["input_size"],
        device=cfg["freshness_classifier"]["device"],
        class_names=cfg["freshness_classifier"]["class_names"],
        good_confidence_threshold=cfg["freshness_classifier"]["good_confidence_threshold"],
    )
    size_clf = SizeClassifier(
        pixels_per_mm=cfg["size_classifier"]["pixels_per_mm"],
        size_threshold_mm=cfg["size_classifier"]["size_threshold_mm"],
    )
    tracker = CentroidTracker(
        max_disappeared_frames=cfg["tracking"]["max_disappeared_frames"],
        max_match_distance_px=cfg["tracking"]["max_match_distance_px"],
    )
    router = build_router(cfg)

    cap = cv2.VideoCapture(cfg["camera"]["source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["frame_height"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {cfg['camera']['source']}")

    min_frame_interval = 1.0 / cfg["camera"]["fps_limit"]
    out_dir = cfg["output"]["output_dir"]
    if cfg["output"]["save_annotated_frames"]:
        os.makedirs(out_dir, exist_ok=True)

    print("Pipeline running. Press 'q' to quit.")
    last_tick = 0.0

    try:
        while True:
            now = time.time()
            if now - last_tick < min_frame_interval:
                continue
            last_tick = now

            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; stopping.")
                break

            detections = detector.detect(frame)
            centroids = [d.centroid for d in detections]
            assigned = tracker.update(centroids)

            # Match assigned IDs back to detections by nearest centroid.
            id_by_centroid = {v: k for k, v in assigned.items()}

            for detection in detections:
                piece_id = id_by_centroid.get(detection.centroid)
                if piece_id is None or tracker.is_processed(piece_id):
                    continue
                tracker.mark_processed(piece_id)

                crop = frame[detection.y1:detection.y2, detection.x1:detection.x2]
                if crop.size == 0:
                    continue

                freshness = freshness_clf.classify(crop)

                if freshness.label == "spoiled":
                    router.route("discard", piece_id, {
                        "good_confidence": round(freshness.good_confidence, 3),
                        "spoiled_confidence": round(freshness.spoiled_confidence, 3),
                    })
                    draw_annotation(
                        frame, detection,
                        [f"SPOILED {freshness.spoiled_confidence * 100:.1f}%"],
                        BOX_COLOR_SPOILED,
                    )
                    continue

                size = size_clf.classify(detection)
                destination = "grinding" if size.category == "small" else "packing"
                router.route(destination, piece_id, {
                    "good_confidence": round(freshness.good_confidence, 3),
                    "spoiled_confidence": round(freshness.spoiled_confidence, 3),
                    "size_category": size.category,
                    "longest_dimension_mm": round(size.longest_dimension_mm, 1),
                })
                draw_annotation(
                    frame, detection,
                    [f"GOOD {freshness.good_confidence * 100:.1f}%",
                     f"{size.category.upper()} ({size.longest_dimension_mm:.0f}mm) -> {destination}"],
                    BOX_COLOR_GOOD,
                )

            cv2.imshow("Meat QC Pipeline", frame)
            if cfg["output"]["save_annotated_frames"]:
                cv2.imwrite(os.path.join(out_dir, f"frame_{int(now * 1000)}.jpg"), frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
