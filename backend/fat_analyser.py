"""
fat_analyser.py — Visible fat coverage estimation module.

Module 3 of the Marbl Meat QC Pipeline.

Loads a trained U-Net segmentation model and produces per-image:
  - Per-pixel class mask  (background=0, lean_meat=1, visible_fat=2)
  - Visible fat coverage percentage  (fat_px / meat_px × 100)
  - Lean meat coverage percentage
  - Colour-overlaid segmentation image (returned as a Base64-encoded JPEG)
  - Segmentation quality score (mean max-softmax confidence over meat pixels)

Scientific disclaimer
---------------------
The output represents VISIBLE SURFACE FAT COVERAGE only.
It is NOT:
  - Intramuscular fat percentage
  - Fat content by weight
  - Nutritional fat percentage
  - Fat throughout the 3D volume of the piece

Always set is_surface_fat_only = True in all returned results and
propagate this flag to the API layer.

Dependencies
------------
  Runtime:  segmentation-models-pytorch >= 0.3.3, torch, torchvision, opencv-python, numpy
  Training: albumentations (see scripts/train_fat_segmenter.py)

The module guards against missing segmentation_models_pytorch with a top-level
SMP_AVAILABLE flag so the server starts cleanly even if the library is absent.
"""

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── Optional dependency guard ────────────────────────────────────────────────
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    smp = None          # type: ignore[assignment]
    SMP_AVAILABLE = False

from torchvision.models import mobilenet_v2


# ── Native PyTorch U-Net with MobileNetV2 Backbone ───────────────────────────
class _ConvBlock(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DecoderBlock(torch.nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Handle possible 1-pixel rounding mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetV2UNet(torch.nn.Module):
    """Native PyTorch U-Net architecture with MobileNetV2 backbone.

    Matches the exact feature extraction stages of MobileNetV2.
    Works natively with standard torch + torchvision with no extra dependencies.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        backbone = mobilenet_v2(weights="DEFAULT" if pretrained else None)
        f = backbone.features
        self.enc0 = f[0:2]    # 1/2, 16 ch
        self.enc1 = f[2:4]    # 1/4, 24 ch
        self.enc2 = f[4:7]    # 1/8, 32 ch
        self.enc3 = f[7:14]   # 1/16, 96 ch
        self.enc4 = f[14:18]  # 1/32, 320 ch

        self.dec4 = _DecoderBlock(320, 96, 128)
        self.dec3 = _DecoderBlock(128, 32, 64)
        self.dec2 = _DecoderBlock(64, 24, 32)
        self.dec1 = _DecoderBlock(32, 16, 24)
        self.dec0 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2),
            _ConvBlock(16, 16)
        )
        self.head = torch.nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        d0 = self.dec0(d1)
        return self.head(d0)


def build_unet_model(num_classes: int = 3, pretrained: bool = False) -> torch.nn.Module:
    """Build U-Net segmentation model.

    Initializes encoder backbone weights from existing local freshness_classifier.pt
    weights if available, avoiding any external network dependency.
    """
    model = MobileNetV2UNet(num_classes=num_classes, pretrained=False)
    if pretrained:
        # Load backbone features from local weights if available
        weights_candidates = [
            Path(__file__).resolve().parent.parent / "models" / "freshness_classifier.pt",
            Path("models/freshness_classifier.pt"),
            Path("../models/freshness_classifier.pt"),
        ]
        loaded = False
        for p in weights_candidates:
            if p.exists():
                try:
                    sd = torch.load(str(p), map_location="cpu", weights_only=True)
                    enc_sd = {}
                    for k, v in sd.items():
                        if k.startswith("features."):
                            idx = int(k.split(".")[1])
                            sub = ".".join(k.split(".")[2:])
                            if idx < 2:
                                enc_sd[f"enc0.{idx}.{sub}"] = v
                            elif idx < 4:
                                enc_sd[f"enc1.{idx-2}.{sub}"] = v
                            elif idx < 7:
                                enc_sd[f"enc2.{idx-4}.{sub}"] = v
                            elif idx < 14:
                                enc_sd[f"enc3.{idx-7}.{sub}"] = v
                            else:
                                enc_sd[f"enc4.{idx-14}.{sub}"] = v
                    model.load_state_dict(enc_sd, strict=False)
                    print(f"[build_unet_model] Initialized MobileNetV2 backbone from '{p.name}' ({len(enc_sd)} tensors)")
                    loaded = True
                    break
                except Exception as e:
                    print(f"[build_unet_model] Could not load weights from '{p}': {e}")
        if not loaded:
            print("[build_unet_model] Using random initialization (no pretrained weights loaded)")
    return model


# ── ImageNet normalisation constants ─────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Segmentation class IDs ────────────────────────────────────────────────────
CLASS_BACKGROUND  = 0
CLASS_LEAN_MEAT   = 1
CLASS_VISIBLE_FAT = 2

# ── Overlay colours (RGB) ─────────────────────────────────────────────────────
_LEAN_OVERLAY_RGB = np.array([220, 60,  60],  dtype=np.float32)   # red tint
_FAT_OVERLAY_RGB  = np.array([255, 200, 50],  dtype=np.float32)   # yellow tint
_LEAN_ALPHA = 0.35
_FAT_ALPHA  = 0.40
_BG_DIM     = 0.35   # background pixel brightness multiplier


@dataclass
class FatAnalysisResult:
    """All outputs from one fat coverage analysis run.

    Fields
    ------
    fat_coverage_pct : float
        Visible fat as a percentage of total segmented meat surface
        (fat_pixels / (fat_pixels + lean_pixels) × 100).
        Excludes background pixels.
    lean_coverage_pct : float
        Lean meat as a percentage of total segmented meat surface.
    fat_category : str
        Qualitative category: "low", "moderate", or "high" based on
        configured thresholds (default: low < 15% <= moderate < 40% <= high).
    segmentation_quality : float
        Mean of the maximum softmax probability across class dimension,
        measured only over meat pixels (0.0-1.0).
        Higher values indicate more confident predictions per-pixel.
    overlay_image_base64 : str | None
        Colour-coded segmentation overlay image encoded as a Base64 JPEG
        string in the format "data:image/jpeg;base64,...".
        Red tint = lean meat, Yellow tint = visible fat, Dimmed = background.
        None if overlay generation fails.
    fat_pixel_count : int
        Raw count of pixels predicted as visible fat within the meat mask.
    lean_pixel_count : int
        Raw count of pixels predicted as lean meat within the meat mask.
    total_meat_pixels : int
        fat_pixel_count + lean_pixel_count (background excluded).
    is_surface_fat_only : bool
        Always True. Scientific disclaimer flag.
        Signals that this result represents surface-visible fat only.
    """
    fat_coverage_pct:       float
    lean_coverage_pct:      float
    fat_category:           str
    segmentation_quality:   float
    overlay_image_base64:   "str | None"
    fat_pixel_count:        int
    lean_pixel_count:       int
    total_meat_pixels:      int
    is_surface_fat_only:    bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        # Enforce the scientific disclaimer — this must never be False
        object.__setattr__(self, "is_surface_fat_only", True)


def _empty_result() -> FatAnalysisResult:
    """Return a zero-valued FatAnalysisResult for error/fallback cases."""
    return FatAnalysisResult(
        fat_coverage_pct=0.0,
        lean_coverage_pct=0.0,
        fat_category="low",
        segmentation_quality=0.0,
        overlay_image_base64=None,
        fat_pixel_count=0,
        lean_pixel_count=0,
        total_meat_pixels=0,
    )


def _classify_fat(pct: float, low: float, high: float) -> str:
    """Map a fat coverage percentage to a qualitative category string.

    Args:
        pct:  Fat coverage percentage (0-100).
        low:  Upper bound for the "low" category (exclusive).
        high: Lower bound for the "high" category (inclusive).

    Returns:
        "low", "moderate", or "high".
    """
    if pct >= high:
        return "high"
    if pct >= low:
        return "moderate"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# HSV-threshold stub — used when no trained model is available
# ─────────────────────────────────────────────────────────────────────────────

def build_fat_stub_result(
    meat_crop_bgr: np.ndarray,
    meat_mask_uint8: np.ndarray,
) -> FatAnalysisResult:
    """Estimate visible fat coverage using a simple HSV white-pixel heuristic.

    This stub is a deterministic fallback used during development before
    a trained U-Net model is available.  It detects white/cream pixels
    (low saturation, high brightness in HSV) as fat and everything else
    in the meat region as lean.

    The stub intentionally over-detects fat because it cannot distinguish
    white fat from brightly-lit lean muscle or bone.  Replace with the
    trained model (FatAnalyser) for production use.

    Args:
        meat_crop_bgr:   BGR crop of the meat region (OpenCV format).
        meat_mask_uint8: Binary mask (0=background, 255=meat) at the
                         original image resolution.  Will be resized to
                         match meat_crop_bgr if shapes differ.

    Returns:
        FatAnalysisResult with is_surface_fat_only=True and
        overlay_image_base64=None (stub does not generate colour overlays).
    """
    try:
        h, w = meat_crop_bgr.shape[:2]

        # Resize mask to match the crop dimensions
        mask_resized = cv2.resize(
            meat_mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST
        )
        meat_region = mask_resized > 0

        # Convert to HSV for colour analysis
        hsv = cv2.cvtColor(meat_crop_bgr, cv2.COLOR_BGR2HSV)
        s_chan = hsv[:, :, 1].astype(np.float32)
        v_chan = hsv[:, :, 2].astype(np.float32)

        # Fat heuristic: low saturation AND high brightness = white/cream pixel
        fat_mask  = (s_chan < 50) & (v_chan > 160)
        lean_mask = ~fat_mask

        fat_px  = int(np.sum(fat_mask  & meat_region))
        lean_px = int(np.sum(lean_mask & meat_region))
        total   = fat_px + lean_px

        fat_pct  = fat_px  / total * 100.0 if total > 0 else 0.0
        lean_pct = lean_px / total * 100.0 if total > 0 else 0.0

        return FatAnalysisResult(
            fat_coverage_pct=round(fat_pct, 2),
            lean_coverage_pct=round(lean_pct, 2),
            fat_category=_classify_fat(fat_pct, low=15.0, high=40.0),
            segmentation_quality=0.0,     # stub has no confidence score
            overlay_image_base64=None,    # stub does not generate overlays
            fat_pixel_count=fat_px,
            lean_pixel_count=lean_px,
            total_meat_pixels=total,
        )

    except Exception:
        return _empty_result()


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class FatAnalyser:
    """U-Net based visible fat coverage analyser.

    Loads a trained segmentation model and exposes a single ``analyse()``
    method that accepts a meat crop image and its foreground mask, returning
    a :class:`FatAnalysisResult` with pixel counts, percentages, a quality
    score, and a coloured overlay image.

    Args:
        weights_path:   Absolute path to the trained model weights (.pt file).
        input_size:     Square resolution to resize crops to before inference
                        (default 256 — the training resolution).
        device:         PyTorch device string — "cpu" or "cuda".
        fat_thresholds: Optional dict with keys "low" and "high" (floats,
                        percentage values) to override the default category
                        thresholds.  Default: low=15.0, high=40.0.

    Raises:
        ImportError:  If segmentation_models_pytorch is not installed.
        FileNotFoundError: If weights_path does not exist.
        RuntimeError: If the weights cannot be loaded into the model.
    """

    def __init__(
        self,
        weights_path: str,
        input_size: int = 256,
        device: str = "cpu",
        fat_thresholds: "dict | None" = None,
    ) -> None:
        self.input_size = input_size
        # Use requested device; fall back to CPU if CUDA not available
        _dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(_dev)

        # Fat category thresholds (percentage of meat surface)
        thresholds = fat_thresholds or {}
        self._low_threshold  = float(thresholds.get("low",  15.0))
        self._high_threshold = float(thresholds.get("high", 40.0))

        # Build U-Net with MobileNetV2 backbone (native or SMP)
        self.model = build_unet_model(num_classes=3, pretrained=False)

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            # If loaded model is SMP vs native, try loading alternate architecture
            alt_model = MobileNetV2UNet(num_classes=3, pretrained=False)
            alt_model.load_state_dict(state_dict)
            self.model = alt_model

        self.model.to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"[FatAnalyser] Loaded weights from '{weights_path}' "
            f"on {self.device}  |  parameters: {n_params:,}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        meat_crop_bgr: np.ndarray,
        meat_mask_uint8: np.ndarray,
        boxes: "list[tuple] | None" = None,
        full_image_bgr: "np.ndarray | None" = None,
    ) -> FatAnalysisResult:
        """Run visible fat coverage analysis.

        Supports both single crops and multi-piece full images:
        - When `full_image_bgr` and `boxes` are provided, analyzes every
          piece at native resolution, aggregates metrics across all pieces,
          and generates a full-image composite overlay.
        - Otherwise analyzes `meat_crop_bgr` directly with `meat_mask_uint8`.

        Args:
            meat_crop_bgr:   BGR crop of the primary meat region or full image.
            meat_mask_uint8: Binary foreground mask (uint8, 0=background, 255=meat).
            boxes:           Optional list of (x, y, w, h) bounding boxes.
            full_image_bgr:  Optional full uploaded image for multi-piece compositing.

        Returns:
            FatAnalysisResult with all fields populated.
        """
        try:
            if full_image_bgr is not None and boxes and len(boxes) > 0:
                return self._analyse_multi_piece(full_image_bgr, meat_mask_uint8, boxes)
            return self._analyse_inner(meat_crop_bgr, meat_mask_uint8)
        except Exception as exc:
            import sys
            print(f"[FatAnalyser] analyse() failed: {exc}", file=sys.stderr)
            return _empty_result()

    # ── Internal implementation ───────────────────────────────────────────────

    def _analyse_multi_piece(
        self,
        full_image_bgr: np.ndarray,
        full_mask_uint8: np.ndarray,
        boxes: list[tuple],
    ) -> FatAnalysisResult:
        """Piece-by-piece inference across all detected meat pieces."""
        h, w = full_image_bgr.shape[:2]
        full_pred = np.zeros((h, w), dtype=np.uint8)
        total_fat_px = 0
        total_lean_px = 0
        qualities = []

        meat_region_full = full_mask_uint8 > 0

        # Background color rejection on full image
        hsv_full = cv2.cvtColor(full_image_bgr, cv2.COLOR_BGR2HSV)
        lab_full = cv2.cvtColor(full_image_bgr, cv2.COLOR_BGR2LAB)
        S_chan = hsv_full[:, :, 1]
        V_chan = hsv_full[:, :, 2]
        L_chan = lab_full[:, :, 0]
        A_chan = lab_full[:, :, 1]
        is_obvious_bg = (
            ((A_chan >= 120) & (A_chan <= 136) & (S_chan < 35)) |
            ((V_chan < 30) | (L_chan < 30)) |
            ((L_chan > 245) & (S_chan < 25))
        )
        meat_region_full = meat_region_full & (~is_obvious_bg)

        for (bx, by, bw, bh) in boxes:
            if bw < 4 or bh < 4:
                continue
            crop = full_image_bgr[by:by+bh, bx:bx+bw]
            c_mask = meat_region_full[by:by+bh, bx:bx+bw]
            if not c_mask.any():
                continue

            # U-Net inference on piece crop
            resized = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred_small = np.argmax(probs, axis=0).astype(np.uint8)
            pred_crop = cv2.resize(pred_small, (bw, bh), interpolation=cv2.INTER_NEAREST)

            # Paste predictions onto full_pred for pixels within the meat region
            full_pred[by:by+bh, bx:bx+bw] = np.where(c_mask, pred_crop, full_pred[by:by+bh, bx:bx+bw])

            fat_px = int(np.sum((pred_crop == CLASS_VISIBLE_FAT) & c_mask))
            lean_px = int(np.sum((pred_crop == CLASS_LEAN_MEAT) & c_mask))
            total_fat_px += fat_px
            total_lean_px += lean_px

            probs_large = cv2.resize(np.max(probs, axis=0), (bw, bh), interpolation=cv2.INTER_LINEAR)
            qualities.append(float(np.mean(probs_large[c_mask])))

        total_meat = total_fat_px + total_lean_px
        fat_pct = (total_fat_px / total_meat * 100.0) if total_meat > 0 else 0.0
        lean_pct = (total_lean_px / total_meat * 100.0) if total_meat > 0 else 0.0
        quality = float(np.mean(qualities)) if qualities else 0.0

        overlay_b64 = self._build_overlay(full_image_bgr, full_pred, meat_region_full)

        return FatAnalysisResult(
            fat_coverage_pct=round(fat_pct, 2),
            lean_coverage_pct=round(lean_pct, 2),
            fat_category=_classify_fat(fat_pct, self._low_threshold, self._high_threshold),
            segmentation_quality=round(quality, 4),
            overlay_image_base64=overlay_b64,
            fat_pixel_count=total_fat_px,
            lean_pixel_count=total_lean_px,
            total_meat_pixels=total_meat,
        )

    def _analyse_inner(
        self,
        meat_crop_bgr: np.ndarray,
        meat_mask_uint8: np.ndarray,
    ) -> FatAnalysisResult:
        """Core inference logic on a single image or crop."""
        orig_h, orig_w = meat_crop_bgr.shape[:2]

        # ── 1. Preprocess image ───────────────────────────────────────────
        resized = cv2.resize(
            meat_crop_bgr, (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        # BGR -> RGB -> float32 -> ImageNet normalise -> CHW tensor
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        # ── 2. Inference ──────────────────────────────────────────────────
        with torch.no_grad():
            logits   = self.model(tensor)                     # (1, 3, H_in, W_in)
            probs    = F.softmax(logits, dim=1).squeeze(0)   # (3, H_in, W_in)
            probs_np = probs.cpu().numpy()                    # (3, H_in, W_in)

        # ── 3. Upsample back to original crop size ────────────────────────
        pred_small = np.argmax(probs_np, axis=0).astype(np.uint8)  # (H_in, W_in)
        pred_mask  = cv2.resize(pred_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Upsample probability array for quality score computation
        probs_large = np.stack([
            cv2.resize(probs_np[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            for c in range(3)
        ], axis=0)  # (3, H_orig, W_orig)

        # ── 4. Resize meat mask and build boolean region ──────────────────
        if meat_mask_uint8.shape[:2] == (orig_h, orig_w):
            mask_resized = meat_mask_uint8
        else:
            mask_resized = cv2.resize(
                meat_mask_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )
        meat_region = mask_resized > 0  # boolean (H_orig, W_orig)

        # Background color rejection: ensure neutral background is not classified as fat
        hsv = cv2.cvtColor(meat_crop_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(meat_crop_bgr, cv2.COLOR_BGR2LAB)
        S_chan = hsv[:, :, 1]
        V_chan = hsv[:, :, 2]
        L_chan = lab[:, :, 0]
        A_chan = lab[:, :, 1]
        is_obvious_bg = (
            ((A_chan >= 120) & (A_chan <= 136) & (S_chan < 35)) |
            ((V_chan < 30) | (L_chan < 30)) |
            ((L_chan > 245) & (S_chan < 25))
        )
        pred_mask[is_obvious_bg] = CLASS_BACKGROUND
        meat_region = meat_region & (~is_obvious_bg)

        # ── 5. Pixel counting (background excluded per specification) ─────
        fat_px  = int(np.sum((pred_mask == CLASS_VISIBLE_FAT) & meat_region))
        lean_px = int(np.sum((pred_mask == CLASS_LEAN_MEAT)   & meat_region))
        total   = fat_px + lean_px

        #   Visible Fat Coverage (%) = (Visible Fat Pixels / Total Segmented
        #                               Meat Pixels) x 100
        #   where Total Segmented Meat Pixels = Visible Fat + Lean Meat pixels
        fat_pct  = fat_px  / total * 100.0 if total > 0 else 0.0
        lean_pct = lean_px / total * 100.0 if total > 0 else 0.0

        # ── 6. Segmentation quality score ─────────────────────────────────
        max_prob = np.max(probs_large, axis=0)   # (H, W)
        quality  = float(np.mean(max_prob[meat_region])) if meat_region.any() else 0.0

        # ── 7. Build colour overlay image ─────────────────────────────────
        overlay_b64 = self._build_overlay(meat_crop_bgr, pred_mask, meat_region)

        return FatAnalysisResult(
            fat_coverage_pct=round(fat_pct, 2),
            lean_coverage_pct=round(lean_pct, 2),
            fat_category=_classify_fat(fat_pct, self._low_threshold, self._high_threshold),
            segmentation_quality=round(quality, 4),
            overlay_image_base64=overlay_b64,
            fat_pixel_count=fat_px,
            lean_pixel_count=lean_px,
            total_meat_pixels=total,
        )

    def _build_overlay(
        self,
        meat_crop_bgr: np.ndarray,
        pred_mask: np.ndarray,
        meat_region: np.ndarray,
    ) -> "str | None":
        """Composite a colour-coded segmentation overlay onto the crop.

        Colour scheme:
          - Lean meat pixels:    red tint   (semi-transparent alpha blend)
          - Visible fat pixels:  yellow tint (semi-transparent alpha blend)
          - Background pixels:   dimmed to 35% brightness

        Returns the result as a "data:image/jpeg;base64,..." string, or None
        on any error so that overlay failures never crash the pipeline.
        """
        try:
            # Work in float32 RGB
            rgb = cv2.cvtColor(meat_crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

            # Dim background
            bg_region = ~meat_region
            rgb[bg_region] *= _BG_DIM

            # Lean meat overlay (red tint)
            lean_region = (pred_mask == CLASS_LEAN_MEAT) & meat_region
            if lean_region.any():
                rgb[lean_region] = (
                    (1.0 - _LEAN_ALPHA) * rgb[lean_region]
                    + _LEAN_ALPHA * _LEAN_OVERLAY_RGB
                )

            # Visible fat overlay (yellow tint)
            fat_region = (pred_mask == CLASS_VISIBLE_FAT) & meat_region
            if fat_region.any():
                rgb[fat_region] = (
                    (1.0 - _FAT_ALPHA) * rgb[fat_region]
                    + _FAT_ALPHA * _FAT_OVERLAY_RGB
                )

            # Clip to valid range and convert
            overlay_rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(overlay_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=82, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"

        except Exception:
            return None
