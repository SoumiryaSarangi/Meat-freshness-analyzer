"""
Image annotation module — Meat QC pipeline.

Draws classification results (rounded bounding box, verdict pill) directly
onto a resized copy of the original image using Pillow.

Called by app.py after process_image() returns a result dict.
Returns raw JPEG bytes, or None if annotation cannot proceed
(no box in result, error case, or an unexpected exception).

Module 3 extension: if the result contains fat_coverage_pct (from the fat
analyser), a secondary "FAT · X%" badge is drawn below the freshness pill.
The fat badge is colour-coded by category: low=green, moderate=amber, high=orange.

This is a presentation layer only — it does NOT modify or call any
classification logic.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_FONT_PATH  = ASSETS_DIR / "SpaceGrotesk-Bold.ttf"

# ---------------------------------------------------------------------------
# Color palette  (R, G, B)
# ---------------------------------------------------------------------------
_GOOD_COLOR    = (55, 227, 154)   # #37E39A
_SPOILED_COLOR = (255, 92, 92)    # #FF5C5C
_REVIEW_COLOR  = (255, 201, 60)   # #FFC93C

# Module 3 — fat badge colours keyed by fat_category value
_FAT_BADGE_COLOR = {
    "low":      (55, 227, 154),   # green  — same as good
    "moderate": (255, 201, 60),   # amber
    "high":     (255, 120, 30),   # orange — visual alert
}

# Box outline color by decision value
_BOX_COLOR_BY_DECISION = {
    "discard":             _SPOILED_COLOR,
    "grinding":            _GOOD_COLOR,    # good piece, small
    "packing":             _GOOD_COLOR,    # good piece, large
    "needs_manual_review": _REVIEW_COLOR,
}

# Pill background color by freshness label
_PILL_COLOR_BY_LABEL = {
    "good":    _GOOD_COLOR,
    "spoiled": _SPOILED_COLOR,
}

_MAX_LONGEST_SIDE = 1280
_JPEG_QUALITY     = 85


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Load bundled Space Grotesk Bold; fall back gracefully to PIL default."""
    try:
        return ImageFont.truetype(str(_FONT_PATH), size)
    except Exception:
        # load_default(size=…) available since Pillow 10.1
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """WCAG 2.1 relative luminance (0 = black, 1 = white)."""
    def _lin(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _text_color_for_bg(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return dark or near-white text depending on background luminance."""
    return (15, 15, 15) if _relative_luminance(bg) > 0.35 else (242, 243, 240)


def _rounded_rect(
    draw: "ImageDraw.ImageDraw",
    xy: tuple[int, int, int, int],
    radius: int,
    *,
    fill: "tuple | None" = None,
    outline: "tuple | None" = None,
    width: int = 1,
) -> None:
    """Thin wrapper around Pillow's rounded_rectangle."""
    draw.rounded_rectangle(xy, radius=max(radius, 1), fill=fill,
                           outline=outline, width=width)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_image(image_bgr: np.ndarray, result: dict) -> bytes | None:
    """Draw classification results onto a copy of *image_bgr*.

    Args:
        image_bgr:  OpenCV-format image (BGR uint8 ndarray).
        result:     Result dict from process_image().  Must contain "box"
                    key ([x, y, w, h]) — returns None otherwise.

    Returns:
        JPEG bytes (longest side ≤ 1280 px, quality 85), or None if
        annotation cannot proceed.
    """
    if not result or "box" not in result or result.get("error"):
        return None

    try:
        # ── 1.  Convert BGR → RGB, build Pillow image ──────────────────
        img_rgb = image_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(img_rgb)
        orig_w, orig_h = pil_img.size

        # ── 2.  Resize so longest side ≤ 1280 px ──────────────────────
        scale = 1.0
        if max(orig_w, orig_h) > _MAX_LONGEST_SIDE:
            scale = _MAX_LONGEST_SIDE / max(orig_w, orig_h)
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        img_w, img_h = pil_img.size

        draw = ImageDraw.Draw(pil_img)

        # ── 3.  Scale bounding box(es) ──────────────────────────────────
        all_boxes = result.get("boxes") or [result["box"]]
        scaled_boxes = []
        for raw_b in all_boxes:
            sbx, sby, sbw, sbh = (int(v * scale) for v in raw_b)
            sbx = max(0, min(sbx, img_w - 1))
            sby = max(0, min(sby, img_h - 1))
            sbw = max(1, min(sbw, img_w - sbx))
            sbh = max(1, min(sbh, img_h - sby))
            scaled_boxes.append((sbx, sby, sbw, sbh))

        bx, by, bw, bh = scaled_boxes[0] if scaled_boxes else (0, 0, img_w, img_h)

        # ── 4.  Pick colors based on decision / label ──────────────────
        decision = result.get("decision", "")
        label    = result.get("label", "")

        # Spoiled always gets red box regardless of decision
        if label == "spoiled":
            box_color  = _SPOILED_COLOR
            pill_color = _SPOILED_COLOR
        elif decision == "needs_manual_review":
            box_color  = _REVIEW_COLOR
            pill_color = _REVIEW_COLOR
        else:
            box_color  = _BOX_COLOR_BY_DECISION.get(decision, _GOOD_COLOR)
            pill_color = _PILL_COLOR_BY_LABEL.get(label, _GOOD_COLOR)

        txt_color = _text_color_for_bg(pill_color)

        # ── 5.  Draw rounded bounding boxes ───────────────────────────
        box_w = max(2, int(min(img_w, img_h) * 0.004))
        for (sbx, sby, sbw, sbh) in scaled_boxes:
            _rounded_rect(
                draw,
                (sbx, sby, sbx + sbw, sby + sbh),
                radius=max(6, int(min(sbw, sbh) * 0.06)),
                outline=box_color,
                width=box_w,
            )

        # ── 6.  Build label text ───────────────────────────────────────
        if decision == "needs_manual_review":
            label_text = "MULTIPLE PIECES \u00b7 REVIEW"
        elif label == "spoiled":
            pct = int(result.get("spoiled_confidence", 0) * 100)
            label_text = f"SPOILED \u00b7 {pct}%"
        else:
            pct = int(result.get("good_confidence", 0) * 100)
            label_text = f"GOOD \u00b7 {pct}%"

        # ── 7.  Measure + draw verdict pill ───────────────────────────
        font_size = max(14, int(min(img_w, img_h) * 0.038))
        font      = _load_font(font_size)

        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_off_x = -text_bbox[0]   # normalise for fonts with negative bearing
        text_off_y = -text_bbox[1]

        pad_x = int(font_size * 0.55)
        pad_y = int(font_size * 0.32)
        pill_w = text_w + pad_x * 2
        pill_h = text_h + pad_y * 2

        # Place pill just inside the top-left corner of the box
        px = bx + box_w + 4
        py = by + box_w + 4
        # Keep pill within image canvas
        px = max(2, min(px, img_w - pill_w - 2))
        py = max(2, min(py, img_h - pill_h - 2))

        _rounded_rect(
            draw,
            (px, py, px + pill_w, py + pill_h),
            radius=int(pill_h / 2.2),
            fill=pill_color,
        )
        draw.text(
            (px + pad_x + text_off_x, py + pad_y + text_off_y),
            label_text,
            font=font,
            fill=txt_color,
        )

        # ── 8.  Module 3 — optional fat coverage badge ────────────────────
        fat_pct = result.get("fat_coverage_pct")
        fat_category = result.get("fat_category")
        if fat_pct is not None and fat_category is not None and label != "spoiled":
            fat_text = f"FAT \u00b7 {fat_pct:.1f}%"
            fat_color = _FAT_BADGE_COLOR.get(fat_category, _REVIEW_COLOR)
            fat_txt_color = _text_color_for_bg(fat_color)

            fat_bbox = draw.textbbox((0, 0), fat_text, font=font)
            fat_tw = fat_bbox[2] - fat_bbox[0]
            fat_th = fat_bbox[3] - fat_bbox[1]
            fat_off_x = -fat_bbox[0]
            fat_off_y = -fat_bbox[1]

            fat_pw = fat_tw + pad_x * 2
            fat_ph = fat_th + pad_y * 2

            # Position: directly below the freshness pill, same left edge
            fpx = px
            fpy = py + pill_h + 6
            fpx = max(2, min(fpx, img_w - fat_pw - 2))
            fpy = max(2, min(fpy, img_h - fat_ph - 2))

            _rounded_rect(
                draw,
                (fpx, fpy, fpx + fat_pw, fpy + fat_ph),
                radius=int(fat_ph / 2.2),
                fill=fat_color,
            )
            draw.text(
                (fpx + pad_x + fat_off_x, fpy + pad_y + fat_off_y),
                fat_text,
                font=font,
                fill=fat_txt_color,
            )

        # ── 9.  Encode to JPEG ─────────────────────────────────────────────
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
        return buf.getvalue()

    except Exception:
        # Never crash the API — annotation is purely cosmetic
        return None
