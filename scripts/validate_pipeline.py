import os
import sys
import glob
import time
import yaml
import cv2
import numpy as np

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipeline import process_image
from freshness_classifier import FreshnessClassifier

def validate_full_pipeline(val_dir, config_path):
    print("==================================================")
    print("  Validating FULL PIPELINE on validation dataset")
    print("==================================================")
    
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    fc_cfg = cfg["freshness_classifier"]
    weights_path = os.path.join(os.path.dirname(__file__), "..", "models", "freshness_classifier.pt")
    
    print("Loading model...")
    clf = FreshnessClassifier(
        weights_path=weights_path,
        input_size=int(fc_cfg.get("input_size", 224)),
        device=fc_cfg.get("device", "cuda"),
        class_names=fc_cfg.get("class_names", ("good", "spoiled")),
        good_confidence_threshold=float(fc_cfg.get("good_confidence_threshold", 0.60)),
    )
    
    good_dir = os.path.join(val_dir, "good")
    spoiled_dir = os.path.join(val_dir, "spoiled")
    
    good_files = glob.glob(os.path.join(good_dir, "*.jpg")) + glob.glob(os.path.join(good_dir, "*.png"))
    spoiled_files = glob.glob(os.path.join(spoiled_dir, "*.jpg")) + glob.glob(os.path.join(spoiled_dir, "*.png"))
    
    print(f"Found {len(good_files)} good images, {len(spoiled_files)} spoiled images.\n")
    
    def process_folder(files, expected_label):
        correct = 0
        total = len(files)
        fallbacks = 0
        
        t0 = time.time()
        for i, path in enumerate(files):
            img = cv2.imread(path)
            if img is None:
                total -= 1
                continue
                
            # Simulate the resizing that happens in app.py now
            h, w = img.shape[:2]
            MAX_DIM = 1280
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            
            result = process_image(img, os.path.basename(path), clf, cfg)
            
            if result["label"] == expected_label:
                correct += 1
            if result.get("segmentation_fallback", False):
                fallbacks += 1
                
            if (i+1) % 50 == 0 or (i+1) == len(files):
                print(f"  Processed {i+1}/{len(files)} {expected_label} images...")
                
        t1 = time.time()
        if total == 0:
            return 0, 0
            
        print(f"  --> {expected_label.upper()}: {correct}/{total} correct ({(correct/total)*100:.1f}%)")
        print(f"  --> {fallbacks}/{total} images triggered the full-tray fallback.")
        print(f"  --> Time taken: {t1-t0:.1f}s ({(t1-t0)/total*1000:.1f}ms per image)\n")
        return correct, total

    print("--- Testing GOOD meat ---")
    good_correct, good_total = process_folder(good_files, "good")
    
    print("--- Testing SPOILED meat ---")
    spoiled_correct, spoiled_total = process_folder(spoiled_files, "spoiled")
    
    overall_correct = good_correct + spoiled_correct
    overall_total = good_total + spoiled_total
    
    print("==================================================")
    print("  OVERALL PIPELINE RESULTS")
    print("==================================================")
    if overall_total > 0:
        print(f"  Good Accuracy:    {good_correct}/{good_total} ({(good_correct/good_total)*100:.1f}%)")
        print(f"  Spoiled Accuracy: {spoiled_correct}/{spoiled_total} ({(spoiled_correct/spoiled_total)*100:.1f}%)")
        print(f"  Overall Accuracy: {overall_correct}/{overall_total} ({(overall_correct/overall_total)*100:.1f}%)")
    print("==================================================")

if __name__ == "__main__":
    val_d = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_freshness", "val")
    cfg_p = os.path.join(os.path.dirname(__file__), "..", "backend", "config.yaml")
    validate_full_pipeline(val_d, cfg_p)
