# Fat Segmentation Dataset — Annotation Guide

> **Module 3: AI-Based Visible Fat Coverage Analysis**  
> Follow this guide before starting annotation. Consistent labelling is the
> single biggest driver of model quality.

---

## Overview

The U-Net model needs pixel-level labels for every image. Each pixel in every
mask must be assigned one of three integer values:

| Class ID | Name | Meaning | Typical colour in image |
|---|---|---|---|
| **0** | `background` | Conveyor belt, tray, anything outside the meat | Grey, black, metallic |
| **1** | `lean_meat` | Red or pink muscle tissue | Deep red, pink, maroon |
| **2** | `visible_fat` | White or cream fat deposits on the surface | White, cream, ivory, light yellow |

**All three classes must appear in every mask**, even if a region is very small.
Do not leave any pixel unlabelled.

---

## Recommended Annotation Tool

### LabelMe (free, local, simple)

```bash
pip install labelme
labelme
```

- Use **polygon** tool for fat regions and lean regions
- Export as PNG semantic masks (`File → Save As → PNG`)
- Set class colours: background=black, lean=dark red, fat=white

### CVAT (team use, web-based)

1. Upload images to a CVAT task
2. Create three labels: `background` (id=0), `lean_meat` (id=1), `visible_fat` (id=2)
3. Use **semantic segmentation** annotation type
4. Export as **Segmentation Mask** format → PNG files

---

## Step-by-Step Annotation Workflow

### 1. Load image

Open the raw meat image in your annotation tool.

### 2. Annotate background first

Draw a polygon covering the **entire non-meat area** (conveyor belt, tray border,
any visible props). Label as class `0` (background).

> **Tip:** If the background is simple and uniform, it is faster to label lean
> meat + fat first and then flood-fill the remainder as background.

### 3. Annotate lean meat regions

Draw polygons over all **red and pink muscle tissue** regions. Label as class `1`
(lean_meat).

**Include:**
- Deep red muscle fibres
- Pink connective tissue between fat seams
- Any reddish areas under thin fat layers where the meat colour shows through

**Exclude (label as fat instead):**
- Any white or cream surface visible even when it overlies muscle

### 4. Annotate visible fat regions

Draw polygons over all **white, cream, or ivory surface deposits**. Label as
class `2` (visible_fat).

**Include:**
- Surface fat caps (the thick white exterior layer)
- Intramuscular fat seams visible on the cut surface (marbling lines)
- Cream-coloured connective tissue attached to fat

**Exclude:**
- Bone (label as background)
- Reflective highlights caused by lighting (keep as lean/fat based on the
  underlying tissue, not the highlight)
- Freezer burn (label as lean_meat unless it is clearly a fat region)

### 5. Check mask coverage

Zoom in and verify:
- No unlabelled pixels exist (especially at polygon edges)
- Fat seams as thin as 2–3 px are labelled (they matter for marbled cuts)
- Mask boundary follows the actual tissue boundary, not the image edge

### 6. Save

Save the mask as a PNG file with the **same base name** as the image:
```
images/IMG_001.jpg  →  masks/IMG_001.png
```

Mask PNG encoding: **8-bit grayscale**, values 0, 1, or 2 only.
Do not use colour PNG or JPEG for masks (JPEG compression destroys pixel values).

---

## Dataset Balance Requirements

Collect images across all three coverage categories:

| Category | Fat coverage | Target minimum |
|---|---|---|
| Low | < 15% | 100 images |
| Moderate | 15–40% | 100 images |
| High | > 40% | 100 images |

Also ensure variety in:
- Meat cuts: ribeye, brisket, shoulder, loin, chuck, sirloin
- Piece shapes: irregular, rectangular, rounded
- Fat distribution: uniform cap, marbled seams, mixed
- Piece sizes: small (< 120 mm), large (≥ 120 mm)
- Conveyor background: use the actual setup to match production lighting

---

## Quality Checklist Before Training

Run `scripts/prepare_fat_dataset.py --data data/dataset_fat --dry-run` to
validate the dataset before splitting:

- [ ] Every image has a corresponding mask PNG
- [ ] All mask pixel values are in {0, 1, 2}
- [ ] At least 300 annotated pairs in total
- [ ] At least 80 images in each fat-coverage bucket
- [ ] Masks saved as 8-bit grayscale PNG (not colour, not JPEG)

---

## Common Mistakes to Avoid

| Mistake | Why it matters |
|---|---|
| Labelling surface gloss/highlight as fat | Increases false fat detections under studio lighting |
| Skipping thin marbling seams | Underestimates fat coverage in marbled cuts |
| Labelling bone as lean_meat | Creates spurious lean predictions |
| Using JPEG for mask files | JPEG introduces non-integer pixel values → dataset validation fails |
| Inconsistent edge treatment | Hard boundaries → model learns to predict class at wrong pixel |

---

## Scientific Disclaimer

> ⚠️ These labels represent **visible surface fat** only.  
> The model will learn to detect surface-level fat deposits visible in the 2D
> camera image. This is NOT:
> - Intramuscular fat percentage (measured by lab extraction)
> - Fat content by weight (nutritional fat %)
> - Fat throughout the full 3D volume of the piece
>
> Always communicate this limitation when reporting results.

---

*Reference: Module 3 implementation plan — Marbl Meat QC Pipeline*

