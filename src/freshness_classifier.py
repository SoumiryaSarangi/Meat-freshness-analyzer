"""
Freshness (good vs spoiled) classifier.

Takes a cropped meat region (from the detector's bounding box) and returns
a confidence score for "good" vs "spoiled". Backed by a small transfer-learning
CNN (MobileNetV2 head) -- fast enough to run per-detection in real time.

Train with scripts/train_classifier.py on a labeled good/spoiled dataset.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


@dataclass
class FreshnessResult:
    label: str            # "good" or "spoiled"
    good_confidence: float   # 0.0-1.0, probability the piece is good
    spoiled_confidence: float  # 0.0-1.0, probability the piece is spoiled


def build_model(num_classes: int = 2) -> nn.Module:
    """MobileNetV2 backbone with a fresh classification head. Small and fast --
    a good fit for per-detection inference on a conveyor belt."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


class FreshnessClassifier:
    def __init__(self, weights_path: str, input_size: int = 224,
                 device: str = "cuda", class_names=("good", "spoiled"),
                 good_confidence_threshold: float = 0.60):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.class_names = list(class_names)
        self.good_confidence_threshold = good_confidence_threshold

        self.model = build_model(num_classes=len(self.class_names))
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    def classify(self, crop_bgr: np.ndarray) -> FreshnessResult:
        """crop_bgr: the cropped meat region (BGR, as read by OpenCV)."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        good_idx = self.class_names.index("good")
        spoiled_idx = self.class_names.index("spoiled")
        good_conf = float(probs[good_idx])
        spoiled_conf = float(probs[spoiled_idx])

        label = "good" if good_conf >= self.good_confidence_threshold else "spoiled"
        return FreshnessResult(label=label, good_confidence=good_conf,
                                spoiled_confidence=spoiled_conf)
