"""
Per-image pipeline orchestrator.

Processes one uploaded image through the full chain:
    1. Segment  — GrabCut to isolate the meat region
    2. Classify — MobileNetV2 freshness classifier on the meat crop
    3. Size     — Reference-card or area-based size estimation
    4. Decide   — Map (label, size_category) → routing decision

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
        decision, size_category, size_method, longest_dimension_mm, box.
    """
    size_cfg = cfg.get("size_classifier", {})

    # --- Step 1: Segment meat region ---
    seg = segment_meat(image_bgr)

    # --- Step 2: Freshness classification on the meat crop ---
    freshness = classifier.classify(seg.meat_crop)

    # --- Step 3: Size estimation ---
    size_result: SizeResult = estimate_size(
        meat_box=seg.box,
        image_bgr_or_shape=image_bgr,
        card_aspect_tolerance=float(size_cfg.get("card_aspect_tolerance", 0.18)),
        size_threshold_mm=float(size_cfg.get("size_threshold_mm", 120.0)),
        fallback_area_threshold=float(size_cfg.get("fallback_area_threshold", 0.35)),
    )

    # --- Step 4: Routing decision ---
    if freshness.label == "spoiled":
        decision = "discard"
    elif size_result.category == "small":
        decision = "grinding"
    else:
        decision = "packing"

    return {
        "filename": filename,
        "label": freshness.label,
        "good_confidence": round(freshness.good_confidence, 4),
        "spoiled_confidence": round(freshness.spoiled_confidence, 4),
        "decision": decision,
        "size_category": size_result.category if freshness.label == "good" else None,
        "size_method": size_result.size_method if freshness.label == "good" else None,
        "longest_dimension_mm": size_result.longest_dimension_mm if freshness.label == "good" else None,
        "box": list(seg.box),           # [x, y, w, h] for frontend overlay use
        "segmentation_fallback": seg.fallback,
    }
