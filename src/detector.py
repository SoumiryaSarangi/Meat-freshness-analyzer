"""
YOLO-based meat detector.

Wraps an Ultralytics YOLO model to return, for each detected piece of meat
in a frame: a bounding box, the model's objectness/class confidence.

Train your own weights with scripts/train_yolo.py before this is usable
in production -- a stock COCO-pretrained YOLO does not know what "meat" is.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def centroid(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class MeatDetector:
    def __init__(self, weights_path: str, confidence_threshold: float = 0.45,
                 iou_threshold: float = 0.45, device: str = "cuda",
                 target_class: str = "meat"):
        self.model = YOLO(weights_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_class = target_class

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame, return one Detection per meat piece."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names  # class id -> class name

        for box in result.boxes:
            cls_id = int(box.cls.item())
            class_name = names.get(cls_id, str(cls_id))
            if self.target_class and class_name != self.target_class:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf.item())

            detections.append(Detection(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                confidence=conf, class_name=class_name,
            ))

        return detections
