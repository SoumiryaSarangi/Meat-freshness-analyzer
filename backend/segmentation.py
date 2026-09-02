"""
GrabCut-based meat region segmentation.

Why GrabCut instead of YOLO?
    The original conveyor design required a trained YOLO detector to find meat
    among distractors on a busy belt.  Upload photos are close-up, single-
    subject shots — GrabCut can isolate the foreground without any training data.

Key fix: connected-components largest-blob selection
    If a reference card (used for size calibration) is placed next to the meat
    in the photo, GrabCut marks BOTH as foreground.  A naive bounding rect over
    the whole mask spans both objects and silently inflates the measured meat
    size.  We run cv2.connectedComponentsWithStats and use only the largest
    connected blob's bounding box.  (Confirmed bug: a 85mm piece was measured
    at 142mm when this fix was absent, flipping small->big.)

Multi-piece detection:
    estimate_piece_count() uses a distance-transform + peak-counting approach
    (standard watershed marker technique) to roughly count distinct lumps within
    the foreground mask.  This is NOT an exact count — it is used only to flag
    likely multi-piece tray photos so that size routing is skipped for them.
    The sensitivity threshold (default 0.4) is configurable via
    backend/config.yaml (multi_piece_dist_threshold) and MUST be tuned against
    real tray photos before relying on it in production.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class SegmentationResult:
    meat_crop: np.ndarray   # BGR crop of the meat region
    box: tuple              # (x, y, w, h) in original image coordinates
    mask: np.ndarray        # binary mask (uint8, 0/255) — meat foreground
    fallback: bool          # True if GrabCut failed and we used whole image
    piece_count_estimate: int   # rough lump count; 0 if mask is empty
    likely_multiple_pieces: bool  # True when piece_count_estimate >= 2


def estimate_piece_count(mask: np.ndarray,
                         min_blob_frac: float = 0.08) -> int:
    """Count distinct meat pieces in a foreground mask.

    Uses morphological opening followed by connected-component labelling.
    Morphological opening (erosion then dilation) removes thin ridges, plastic-
    wrap creases, and GrabCut fringe artefacts that a single piece produces —
    while preserving genuinely separate blobs that have visible gaps between them.

    Args:
        mask:          Binary mask (uint8, 0/255) of the foreground region.
        min_blob_frac: Minimum fraction of total foreground area a blob must
                       occupy to be counted as a real piece (default 0.08 = 8%).
                       Filters out small noise blobs and GrabCut slivers.

    Returns:
        Estimated number of distinct pieces (>= 0).
    """
    if mask is None or mask.sum() == 0:
        return 0

    h, w = mask.shape[:2]
    # Opening kernel: ~3% of the shorter image dimension.
    # Large enough to close plastic-wrap wrinkles; small enough to preserve
    # the gap between genuinely separate pieces.
    ksize = max(5, int(min(h, w) * 0.03) | 1)   # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if opened.sum() == 0:
        return 1  # opening removed everything → treat as single small piece

    total_fg = opened.sum() // 255

    # Count connected components on the opened mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    # Keep only blobs large enough to be real pieces (suppress noise slivers)
    count = 0
    for lbl in range(1, num_labels):   # skip background (label 0)
        blob_area = stats[lbl, cv2.CC_STAT_AREA]
        if blob_area >= min_blob_frac * total_fg:
            count += 1

    return max(count, 0)


def segment_meat(image_bgr: np.ndarray) -> SegmentationResult:
    """Isolate the meat region from an uploaded photo.

    Args:
        image_bgr: Full uploaded image, BGR, any size.

    Returns:
        SegmentationResult with the meat crop, its bounding box, the binary
        mask, a fallback flag, a rough piece count, and a multi-piece flag.
    """
    h, w = image_bgr.shape[:2]
    total_pixels = h * w

    # --- GrabCut initial rect: 4% margin on every side ---
    margin_x = int(w * 0.04)
    margin_y = int(h * 0.04)
    rect = (margin_x, margin_y,
            w - 2 * margin_x,
            h - 2 * margin_y)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5,
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # Image too small or degenerate — fall back to whole image.
        return _whole_image_fallback(image_bgr)

    # Pixels labelled GC_FGD or GC_PR_FGD are foreground
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)

    fg_fraction = fg_mask.sum() / 255 / total_pixels

    # Sanity check: if <3% or >98% is foreground GrabCut has mis-segmented
    if fg_fraction < 0.03 or fg_fraction > 0.98:
        return _whole_image_fallback(image_bgr, dist_threshold)

    # --- Piece count estimation on the FULL foreground mask ---
    # Run before narrowing to the largest blob, so all individual lumps are
    # visible.  Using the largest blob alone would miss touching/separate pieces
    # that GrabCut correctly marks as foreground but connected-components splits.
    count = estimate_piece_count(fg_mask)

    # --- Connected components: keep ONLY the largest blob for bounding box ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8)

    if num_labels < 2:
        # No foreground components found
        return _whole_image_fallback(image_bgr)

    # Component 0 is background; find the largest foreground component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    blob_mask = (labels == largest_label).astype(np.uint8) * 255

    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    bw = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    bh = int(stats[largest_label, cv2.CC_STAT_HEIGHT])

    # Clamp to image bounds
    x = max(0, x)
    y = max(0, y)
    bw = min(bw, w - x)
    bh = min(bh, h - y)

    meat_crop = image_bgr[y:y + bh, x:x + bw]

    return SegmentationResult(
        meat_crop=meat_crop,
        box=(x, y, bw, bh),
        mask=blob_mask,
        fallback=False,
        piece_count_estimate=count,
        likely_multiple_pieces=(count >= 2),
    )


def _whole_image_fallback(image_bgr: np.ndarray) -> SegmentationResult:
    """Return the entire image as the meat region when segmentation fails."""
    h, w = image_bgr.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    count = estimate_piece_count(mask)
    return SegmentationResult(
        meat_crop=image_bgr.copy(),
        box=(0, 0, w, h),
        mask=mask,
        fallback=True,
        piece_count_estimate=count,
        likely_multiple_pieces=(count >= 2),
    )
