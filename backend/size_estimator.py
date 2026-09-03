"""
Size estimation for uploaded meat photos.

Two modes:
    "measured"  — A standard ID-1 card (credit/debit/ID, 85.60 × 53.98 mm,
                  aspect ratio ≈ 1.586) was detected in the photo.  pixels_per_mm
                  is derived from the card's measured pixel width, so the meat
                  size is physically calibrated.
    "estimated" — No reference card found.  Falls back to the ratio of the meat
                  bounding-box area to the total image area and uses a fixed
                  threshold (default 0.35) to call big vs small.  This is a rough
                  heuristic — treat results with caution.

Why reference-card instead of fixed pixels_per_mm?
    The conveyor design used a camera mounted at a fixed, known height so one
    calibration constant applied to every frame.  Phone uploads can be taken
    from any distance, making a fixed constant meaningless.  A reference card
    visible in the same photo gives per-image calibration at zero extra cost.
"""

from dataclasses import dataclass

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CARD_ASPECT_RATIO = CARD_WIDTH_MM / CARD_HEIGHT_MM  # ≈ 1.586


@dataclass
class SizeResult:
    category: str            # "big" or "small"
    longest_dimension_mm: float
    size_method: str         # "measured" or "estimated"
    pixels_per_mm: float | None  # None when estimated


def estimate_size(
    meat_box: tuple,
    image_bgr_or_shape,
    card_aspect_tolerance: float = 0.18,
    size_threshold_mm: float = 120.0,
    fallback_area_threshold: float = 0.35,
) -> SizeResult:
    """Estimate whether the meat piece is big or small.

    Args:
        meat_box:              (x, y, w, h) bounding box of the meat in the
                               original image (pixels).
        image_bgr_or_shape:    Full original image as a numpy BGR array.
                               Used to scan the whole frame for a reference card.
        card_aspect_tolerance: ±tolerance around CARD_ASPECT_RATIO (1.586) to
                               accept a contour as a candidate reference card.
        size_threshold_mm:     Meat longest dimension above this → "big".
        fallback_area_threshold: Fraction of total image area → "big" when no
                               reference card is detected.

    Returns:
        SizeResult with category, longest_dimension_mm, size_method, pixels_per_mm.
    """
    # Accept either an image array or a shape tuple for backward-compat
    if isinstance(image_bgr_or_shape, np.ndarray):
        image = image_bgr_or_shape
        img_h, img_w = image.shape[:2]
    else:
        image = None
        img_h, img_w = image_bgr_or_shape[:2]

    total_area = img_h * img_w
    _, _, mw, mh = meat_box
    meat_longest_px = max(mw, mh)

    pixels_per_mm = _detect_reference_card(image, card_aspect_tolerance) if image is not None else None

    if pixels_per_mm is not None:
        longest_mm = meat_longest_px / pixels_per_mm
        category = "big" if longest_mm >= size_threshold_mm else "small"
        return SizeResult(
            category=category,
            longest_dimension_mm=round(longest_mm, 1),
            size_method="measured",
            pixels_per_mm=round(pixels_per_mm, 4),
        )

    # --- Fallback: area-based heuristic ---
    meat_area = mw * mh
    area_fraction = meat_area / total_area if total_area > 0 else 0.0
    category = "big" if area_fraction >= fallback_area_threshold else "small"
    # Report a rough mm estimate using the area fraction (not physically meaningful)
    approx_mm = meat_longest_px / 4.0  # placeholder scale; unreliable
    return SizeResult(
        category=category,
        longest_dimension_mm=round(approx_mm, 1),
        size_method="estimated",
        pixels_per_mm=None,
    )


def _detect_reference_card(
    image_bgr_or_shape,
    aspect_tolerance: float,
) -> float | None:
    """Try to detect an ID-1 card in the image and return pixels_per_mm.

    Strategy:
        1. Convert to grayscale, apply Canny edge detection.
        2. Find external contours; fit a minAreaRect to each.
        3. Accept the rect if its aspect ratio is within ±aspect_tolerance of
           CARD_ASPECT_RATIO (1.586).
        4. Among all candidates pick the one with the largest area.
        5. Derive pixels_per_mm from the card's long side in pixels.

    Returns:
        pixels_per_mm (float) if a card was found, else None.
    """
    # Accept either an image array or a pre-computed shape tuple
    if isinstance(image_bgr_or_shape, np.ndarray):
        image = image_bgr_or_shape
    else:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_area = 0
    best_ppm = None
    total_image_area = image.shape[0] * image.shape[1]
    min_area = total_image_area * 0.015

    for cnt in contours:
        if len(cnt) < 5:
            continue
            
        contour_area = cv2.contourArea(cnt)
        
        rect = cv2.minAreaRect(cnt)
        (_, (rw, rh), _) = rect
        if rw == 0 or rh == 0:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        aspect = long_side / short_side
        rect_area = long_side * short_side

        if abs(aspect - CARD_ASPECT_RATIO) <= aspect_tolerance:
            # Additional sanity checks to reject false positives:
            # 1. Area must be reasonably large (at least 1.5% of image)
            if rect_area < min_area:
                logger.debug(f"Card candidate rejected: area {rect_area:.1f} too small (min {min_area:.1f})")
                continue
                
            # 2. Solidity: the contour should fill most of its bounding rectangle (ID cards are solid rectangles)
            # A perfect rectangle has solidity 1.0. We'll require > 0.80 to be safe with rounded corners/perspective.
            solidity = contour_area / rect_area if rect_area > 0 else 0
            if solidity < 0.80:
                logger.debug(f"Card candidate rejected: solidity {solidity:.3f} < 0.80 (area={rect_area:.1f})")
                continue

            if rect_area > best_area:
                best_area = rect_area
                best_ppm = long_side / CARD_WIDTH_MM

    return best_ppm
