"""
Size classification: turns a detector bounding box into a real-world size
estimate and a big/small decision, using a fixed camera pixels-per-mm
calibration factor.

Calibration: place a reference object of known width (e.g. a 100mm card) flat
on the belt at the same camera height/distance used in production, measure its
bounding box width in pixels, and set:
    pixels_per_mm = reference_width_px / reference_width_mm
"""

from dataclasses import dataclass

from .detector import Detection


@dataclass
class SizeResult:
    width_mm: float
    height_mm: float
    longest_dimension_mm: float
    category: str  # "small" or "big"


class SizeClassifier:
    def __init__(self, pixels_per_mm: float, size_threshold_mm: float):
        if pixels_per_mm <= 0:
            raise ValueError("pixels_per_mm must be calibrated to a positive value")
        self.pixels_per_mm = pixels_per_mm
        self.size_threshold_mm = size_threshold_mm

    def classify(self, detection: Detection) -> SizeResult:
        width_mm = detection.width / self.pixels_per_mm
        height_mm = detection.height / self.pixels_per_mm
        longest = max(width_mm, height_mm)
        category = "big" if longest >= self.size_threshold_mm else "small"
        return SizeResult(width_mm=width_mm, height_mm=height_mm,
                           longest_dimension_mm=longest, category=category)
