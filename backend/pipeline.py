"""
Per-image pipeline orchestrator.

Processes one uploaded image through the full chain:
    1. Segment  — GrabCut to isolate the meat region
    2. Classify — MobileNetV2 freshness classifier on the meat crop
    3. Size     — Reference-card or area-based size estimation
                  (SKIPPED if multiple pieces detected)
    4. Decide   — Map (label, size_category) -> routing decision
                  Routes multi-piece trays straight to grinding.
    5. Fat      — U-Net visible fat coverage analysis (Module 3).
                  Only runs on pieces that pass freshness inspection
                  (label == "good"). Requires fat_analyser to be passed in
                  and fat_analyser.enabled == True in config.yaml.

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
    fat_analyser=None,
) -> dict:
    """Run the full pipeline on one image.

    Args:
        image_bgr:    OpenCV-decoded image (BGR, uint8).
        filename:     Original upload filename (for the response payload).
        classifier:   Loaded FreshnessClassifier instance (shared across requests).
        cfg:          Parsed config.yaml dict.
        fat_analyser: Optional loaded FatAnalyser instance (Module 3).
                      If None or if the piece is spoiled, fat analysis is skipped.

    Returns:
        Dict with keys: filename, label, good_confidence, spoiled_confidence,
        decision, size_category, size_method, longest_dimension_mm,
        piece_count_estimate, box, segmentation_fallback,
        fat_coverage_pct, lean_coverage_pct, fat_category,
        segmentation_quality, fat_overlay_image_base64, fat_analysis_available,
        is_surface_fat_only.
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

    result = {
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
        "boxes": [list(b) for b in getattr(seg, "boxes", [seg.box])],  # all detected piece boxes
        "segmentation_fallback": seg.fallback,
        # ── Module 3: Visible Fat Coverage ────────────────────────────────────
        "fat_coverage_pct": None,
        "lean_coverage_pct": None,
        "fat_category": None,
        "segmentation_quality": None,
        "fat_overlay_image_base64": None,
        "fat_analysis_available": False,
        "is_surface_fat_only": True,    # scientific disclaimer — always set
    }

    # --- Step 5: Visible fat coverage analysis (Module 3) ---
    # Only run on good pieces; spoiled pieces are always discarded — no point
    # computing fat coverage for meat that is being rejected.
    if fat_analyser is not None and freshness.label == "good":
        result = _run_fat_analysis(result, seg, fat_analyser, image_bgr)

    return result


def _run_fat_analysis(result: dict, seg, fat_analyser, image_bgr: np.ndarray) -> dict:
    """Attempt visible fat coverage analysis on a good, segmented meat piece.

    Mutates *result* in-place and returns it.  Any exception is silently caught
    so that a fat analysis failure never crashes the main pipeline.

    Args:
        result:       Partially-built result dict from process_image().
        seg:          SegmentationResult from segment_meat().
        fat_analyser: Loaded FatAnalyser instance.
        image_bgr:    Full input image.

    Returns:
        The same result dict with fat_* fields populated.
    """
    try:
        fat = fat_analyser.analyse(
            meat_crop_bgr=seg.meat_crop,
            meat_mask_uint8=seg.mask,
            boxes=getattr(seg, "boxes", [seg.box]),
            full_image_bgr=image_bgr,
        )
        result["fat_coverage_pct"] = round(fat.fat_coverage_pct, 2)
        result["lean_coverage_pct"] = round(fat.lean_coverage_pct, 2)
        result["fat_category"] = fat.fat_category
        result["segmentation_quality"] = round(fat.segmentation_quality, 4)
        result["fat_overlay_image_base64"] = fat.overlay_image_base64
        result["fat_analysis_available"] = True
        result["is_surface_fat_only"] = True   # always
    except Exception as exc:
        # Fat analysis is non-critical — log and continue
        import sys
        print(f"[pipeline] fat_analyser.analyse() failed for '{result['filename']}': {exc}",
              file=sys.stderr)
    return result
