"""
Image annotation module — Meat QC pipeline.

Draws classification results (rounded bounding box, verdict pill) directly
onto a resized copy of the original image using Pillow.

Called by app.py after process_image() returns a result dict.
Returns raw JPEG bytes, or None if annotation cannot proceed
(no box in result, error case, or an unexpected exception).

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

        # ── 3.  Scale bounding box ──────────────────────────────────────
        bx, by, bw, bh = (int(v * scale) for v in result["box"])
        # Clamp to image bounds
        bx = max(0, min(bx, img_w - 1))
        by = max(0, min(by, img_h - 1))
        bw = max(1, min(bw, img_w - bx))
        bh = max(1, min(bh, img_h - by))

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

        # ── 5.  Draw rounded bounding box ─────────────────────────────
        box_w = max(2, int(min(img_w, img_h) * 0.004))
        _rounded_rect(
            draw,
            (bx, by, bx + bw, by + bh),
            radius=max(8, int(min(bw, bh) * 0.05)),
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

        # ── 8.  Encode to JPEG ─────────────────────────────────────────
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=_JPEG_QUALITY,
                     optimize=True, progressive=True)
        return buf.getvalue()

    except Exception:
        # Never crash the API — annotation is purely cosmetic
        return None
