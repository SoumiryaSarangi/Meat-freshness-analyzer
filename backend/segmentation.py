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
                         dist_threshold: float = 0.4) -> int:
    """Rough count of distinct 'lumps' within a foreground mask.

    Uses a distance-transform + peak-region technique (standard watershed-
    marker approach).  Not exact — used only to flag likely multi-piece
    photos, not to actually separate or measure individual pieces.

    Args:
        mask:           Binary mask (uint8, 0/255) of the foreground region.
        dist_threshold: Fraction of dist.max() above which a pixel is
                        considered a distinct peak (default 0.4).
                        ⚠️  This threshold controls sensitivity and WILL need
                        tuning against real tray photos — the 0.4 default was
                        chosen on synthetic test images only.  Lower values
                        (e.g. 0.3) detect more subtle piece separations;
                        higher values (e.g. 0.5) reduce false positives on
                        single pieces with uneven surfaces.
                        Adjust via config.yaml: multi_piece_dist_threshold.

    Returns:
        Estimated number of distinct pieces (>= 0).
        Returns 0 if the mask is empty or degenerate.
    """
    if mask is None or mask.sum() == 0:
        return 0

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return 0

    _, peaks = cv2.threshold(dist, dist_threshold * dist.max(), 255, 0)
    peaks = peaks.astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(peaks)
    return max(num_labels - 1, 0)  # subtract background label


def segment_meat(image_bgr: np.ndarray,
                 dist_threshold: float = 0.4) -> SegmentationResult:
    """Isolate the meat region from an uploaded photo.

    Args:
        image_bgr:      Full uploaded image, BGR, any size.
        dist_threshold: Sensitivity for piece-count estimation (see
                        estimate_piece_count for full docs).

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
        return _whole_image_fallback(image_bgr, dist_threshold)

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
    count = estimate_piece_count(fg_mask, dist_threshold)

    # --- Connected components: keep ONLY the largest blob for bounding box ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8)

    if num_labels < 2:
        # No foreground components found
        return _whole_image_fallback(image_bgr, dist_threshold)

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


def _whole_image_fallback(image_bgr: np.ndarray,
                          dist_threshold: float = 0.4) -> SegmentationResult:
    """Return the entire image as the meat region when segmentation fails."""
    h, w = image_bgr.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    count = estimate_piece_count(mask, dist_threshold)
    return SegmentationResult(
        meat_crop=image_bgr.copy(),
        box=(0, 0, w, h),
        mask=mask,
        fallback=True,
        piece_count_estimate=count,
        likely_multiple_pieces=(count >= 2),
    )
