# Meat QC Pipeline

Real-time computer vision pipeline for automated meat quality control on a processing line.

```
camera → YOLO detector (bounding box + confidence)
       → freshness CNN   (good / spoiled + confidence %)
            ├── spoiled  → 🔴 DISCARD
            └── good     → size classifier (bbox → mm)
                               ├── small  → ⚙️  Grinding line
                               └── big    → 📦 Packing line
```

Every accepted piece gets a colour-coded bounding box on the live feed with
its freshness confidence and routing decision, and a row written to
`output/decisions_log.csv` for full traceability and audit.

---

## ✅ Current Model Performance

Trained on a **combined dataset** of 3,331 images (Kaggle Meat Freshness +
custom conveyor footage), evaluated on 831 held-out validation images:

| Metric | Value |
|---|---|
| Accuracy | **94.2%** |
| Precision | **95.7%** |
| Recall | **91.0%** |
| F1 Score | **0.933** |
| Spoilage detection rate | **96.7%** |

---

## 🚀 Quick Start — Reproduce Results from Scratch

Follow these steps **in order** to go from zero to a working 94%+ model.

### Step 1 — Clone the repo

```bash
git clone https://github.com/SoumiryaSarangi/project-meat-analysis.git
cd project-meat-analysis
```

### Step 2 — Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> **GPU (recommended):** If you have an NVIDIA GPU, install the CUDA version of
> PyTorch for 5× faster training. Replace the `torch` line in `requirements.txt`
> with the correct CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/).
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> ```
>
> Then verify with:
> ```bash
> python -c "import torch; print('CUDA:', torch.cuda.is_available())"
> ```

### Step 3 — Download the Kaggle dataset

1. Go to [Kaggle — Meat Freshness Image Dataset](https://www.kaggle.com/datasets/vinayakshanawad/meat-monitor-freshness-image-dataset)
2. Click **Download** → you get `archive.zip` (≈ 75 MB)
3. Place the zip anywhere on your machine (e.g. `Downloads/archive.zip`)

### Step 4 — Prepare the Kaggle dataset

```bash
python scripts/prepare_dataset.py --zip "C:/Users/YourName/Downloads/archive.zip"
```

This extracts the zip and sorts the images into:
```
data/dataset_freshness/
    train/  good/   (FRESH images)
            spoiled/ (HALF-FRESH + SPOILED images)
    val/    good/
            spoiled/
```

> **Optional:** Pass `--half-fresh good` if you want to treat borderline
> HALF-FRESH images as good rather than discarding them.

### Step 5 — (Optional) Add your own dataset to fix Domain Shift

If you have your own camera footage, mix it in so the model generalises to
your lighting and background. Organise your images into two folders:
- `path/to/Fresh/`   — photos of fresh / good meat
- `path/to/Spoiled/` — photos of spoiled meat

Then run:

```bash
python scripts/merge_datasets.py \
    --fresh  "path/to/Fresh" \
    --spoiled "path/to/Spoiled"
```

> **Why this matters:** A model trained only on Kaggle studio photos can fail
> badly on real conveyor footage (we observed 47% accuracy on new footage
> before merging, and 96.9% after retraining on the combined dataset).

### Step 6 — Train the freshness classifier

```bash
# CPU only (slow ~30 min):
python scripts/train_classifier.py --data data/dataset_freshness --epochs 30

# NVIDIA GPU (recommended, ~6-12 min):
python scripts/train_classifier.py --data data/dataset_freshness --epochs 30 --batch 64
```

The best checkpoint is saved automatically to `models/freshness_classifier.pt`.
You should see the val accuracy climb steadily toward 90%+.

### Step 7 — Validate the model

Run the pipeline on the entire validation folder:

```bash
python scripts/run_on_images.py \
    --images data/dataset_freshness/val \
    --out-dir output/test_annotated \
    --log-csv output/test_decisions_log.csv
```

Then get the detailed accuracy metrics (Precision, Recall, F1):

```bash
python scripts/analyze_results.py
```

To visually inspect the annotated images one by one:

```bash
python scripts/run_on_images.py --images data/dataset_freshness/val --show
# Press any key to advance, Q to quit
```

---

## 📁 Project Layout

```
config.yaml                 All tunable parameters (thresholds, device, paths)
requirements.txt            Python dependencies
models/
  freshness_classifier.pt   Trained CNN weights (saved automatically by training)
src/
  detector.py               YOLO wrapper → Detection(bbox, confidence)
  freshness_classifier.py   CNN → good/spoiled + confidence
  size_classifier.py        bbox pixels → mm → small/big
  tracker.py                Centroid tracker (process each belt piece once)
  router.py                 Decision → hardware signal (serial/MQTT/console)
  pipeline.py               Main real-time camera loop
scripts/
  prepare_dataset.py        Extract & organise the Kaggle zip
  merge_datasets.py         Mix additional images into train/val split
  run_on_images.py          Test runner: runs pipeline on a folder of images
  analyze_results.py        Reads the CSV log and prints accuracy metrics
  collect_data.py           Webcam capture tool for building your own dataset
  train_classifier.py       Trains the freshness CNN (MobileNetV2)
  train_yolo.py             Trains the YOLO meat detector
```

---

## ⚙️ Configuration (`config.yaml`)

All major parameters are in `config.yaml`. Key settings:

```yaml
freshness_classifier:
  device: "cuda"                   # "cuda" or "cpu"
  good_confidence_threshold: 0.60  # Safety dial: raise → stricter, lower → permissive

size_classifier:
  pixels_per_mm: 4.2               # Calibrate this for your camera!
  size_threshold_mm: 120           # Below = small (grind), above = big (pack)

routing:
  backend: "console"               # "console" | "serial" | "mqtt"
  serial_port: "COM3"              # For Arduino/ESP32 diverter
```

---

## 🎥 Run the Live Camera Pipeline

> **Note:** The live pipeline requires a trained YOLO detector
> (`models/yolo_meat_detector.pt`). Without it, use `run_on_images.py`
> for image-folder testing. See "Train YOLO detector" below.

```bash
# Live webcam (device 0):
python -m src.pipeline --config config.yaml

# Video file:
# Edit config.yaml: camera.source: "path/to/video.mp4"
python -m src.pipeline --config config.yaml
```

---

## 🔍 Train the YOLO Detector (for live camera use)

1. Go to [Roboflow Universe](https://universe.roboflow.com) → search **"meat detection"**
2. Export a dataset in **YOLOv11 format** → download the zip
3. Extract to `data/dataset_yolo/`
4. Train:

```bash
python scripts/train_yolo.py --data data/dataset_yolo/dataset.yaml --epochs 100
```

---

## 📏 Calibrate Size Measurement

The size classifier converts bounding-box pixels to millimetres using
`pixels_per_mm` in `config.yaml`. To calibrate for your camera:

1. Place a flat object of **known width** (e.g. a 100mm card) on the belt.
2. Run the detector on one frame and read the bounding box width in pixels.
3. Set `pixels_per_mm = bbox_width_px / 100` in `config.yaml`.
4. Set `size_threshold_mm` to whatever size separates "grind" from "pack"
   in your specific process.

---

## 🔌 Connect to Hardware

`src/router.py` is the one module to adapt per site. Three backends are ready:

| Backend | Description |
|---|---|
| `console` | Logs decisions only — safe default for testing |
| `serial` | Sends a command byte to an Arduino/ESP32 driving a relay/diverter |
| `mqtt` | Publishes decisions to a broker for PLC/SCADA integration |

Fill in the `TODO`s in your chosen backend, then set `routing.backend`
in `config.yaml`.

---

## 📝 Notes on Production Accuracy

- **Lighting matters more than model choice.** Inconsistent belt lighting will
  hurt the classifier far more than swapping architectures. Use fixed, diffuse,
  consistent-colour-temperature lighting over the inspection zone.
- **`good_confidence_threshold` is your safety dial.** Raising it discards
  more borderline pieces (safer) at the cost of more false rejects.
- **The tracker prevents double-counting.** The same physical piece is only
  classified once as it crosses multiple camera frames.
- **Add your own footage!** Use `scripts/collect_data.py` to capture images
  from your production camera and `scripts/merge_datasets.py` to mix them
  into the training set. This is the single biggest lever for accuracy on
  your specific setup.
