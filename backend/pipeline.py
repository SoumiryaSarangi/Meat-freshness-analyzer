"""
Per-image pipeline orchestrator.

Processes one uploaded image through the full chain:
    1. Segment  — GrabCut to isolate the meat region
    2. Classify — MobileNetV2 freshness classifier on the meat crop
    3. Size     — Reference-card or area-based size estimation
                  (SKIPPED if multiple pieces detected)
    4. Decide   — Map (label, size_category) -> routing decision
                  Routes multi-piece trays straight to grinding.

Returns a single result dict per image, ready to be serialised to JSON.
"""

import cv2
import numpy as np

from segmentation import segment_meat
from size_estimator import estimate_size, SizeResult
from freshness_classifier import FreshnessClassifier


def process_image(
    image_bgr: np.ndarray,
    filename: str,
    classifier: FreshnessClassifier,
    cfg: dict,
) -> dict:
    """Run the full pipeline on one image.

    Args:
        image_bgr:  OpenCV-decoded image (BGR, uint8).
        filename:   Original upload filename (for the response payload).
        classifier: Loaded FreshnessClassifier instance (shared across requests).
        cfg:        Parsed config.yaml dict.

    Returns:
        Dict with keys: filename, label, good_confidence, spoiled_confidence,
        decision, size_category, size_method, longest_dimension_mm,
        piece_count_estimate, box, segmentation_fallback.
    """
    size_cfg = cfg.get("size_classifier", {})
    dist_threshold = float(size_cfg.get("multi_piece_dist_threshold", 0.4))

    # --- Step 1: Segment meat region ---
    seg = segment_meat(image_bgr, dist_threshold=dist_threshold)

    # --- Step 2: Freshness classification on the meat crop ---
    freshness = classifier.classify(seg.meat_crop)

    # --- Step 3 & 4: Size estimation + routing decision ---
    if freshness.label == "spoiled":
        # Spoiled: always discard regardless of size or piece count
        decision = "discard"
        size_category = None
        size_method = None
        longest_dimension_mm = None

    elif seg.likely_multiple_pieces:
        # Multiple pieces detected: skip per-piece size measurement and send
        # straight to grinding.  Freshness is still reported normally.
        decision = "grinding"
        size_category = None
        size_method = "multiple_pieces_detected"
        longest_dimension_mm = None

    else:
        # Single piece, good: run size estimation and route accordingly
        size_result: SizeResult = estimate_size(
            meat_box=seg.box,
            image_bgr_or_shape=image_bgr,
            card_aspect_tolerance=float(size_cfg.get("card_aspect_tolerance", 0.18)),
            size_threshold_mm=float(size_cfg.get("size_threshold_mm", 120.0)),
            fallback_area_threshold=float(size_cfg.get("fallback_area_threshold", 0.35)),
        )
        size_category = size_result.category
        size_method = size_result.size_method
        longest_dimension_mm = size_result.longest_dimension_mm
        decision = "grinding" if size_result.category == "small" else "packing"

    return {
        "filename": filename,
        "label": freshness.label,
        "good_confidence": round(freshness.good_confidence, 4),
        "spoiled_confidence": round(freshness.spoiled_confidence, 4),
        "decision": decision,
        "size_category": size_category,
        "size_method": size_method,
        "longest_dimension_mm": longest_dimension_mm,
        "piece_count_estimate": seg.piece_count_estimate,
        "box": list(seg.box),           # [x, y, w, h] for frontend overlay use
        "segmentation_fallback": seg.fallback,
    }
