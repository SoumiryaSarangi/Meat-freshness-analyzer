# Meat QC Pipeline

> **🆕 Web App is live!** Upload a phone photo → get freshness verdict + routing decision.
> See [Web App Quick Start](#-web-app-quick-start) below.

---

## 🌐 Web App Quick Start

A FastAPI server lets anyone upload photos (from phone or laptop) and get:
- ✅ **Good** / ❌ **Spoiled** with confidence %
- Routing decision: **Discard** / **Grinding** / **Packing**
- Size method: **Measured** (reference card found) or **Estimated** (fallback)

### Why the architecture changed from conveyor pipeline

| Conveyor Pipeline | Web Upload |
|---|---|
| YOLO detector (needs training data) | GrabCut segmentation (works instantly, no training) |
| Fixed `pixels_per_mm` (camera at known height) | Reference-card per-photo calibration (place a credit/ID card next to the meat) |
| Real-time camera loop + hardware router | Per-request HTTP: one image → one JSON response |

**Trade-offs:** GrabCut is a generic segmenter with no meat-specific knowledge — it may struggle with complex backgrounds. Reference-card detection is shape-based (aspect ratio ≈ 1.586) and could match other rectangular objects. `size_threshold_mm=120` is a placeholder carried over from the conveyor design and has not been validated against a real product-line size cutoff.

### Run the web server

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in a browser.
From a phone on the same Wi-Fi: **http://\<your-machine-ip\>:8000**

### How to use size calibration

Place a standard **credit / debit / ID card** (85.60 mm wide) flat next to the meat in the photo. The server detects the card's shape, derives pixels-per-mm for that specific photo, and marks the size result as **"Measured"**. Without a card, it falls back to a frame-area estimate marked **"Estimated"**.

### Known limitations

- GrabCut segments by colour/texture contrast — unusual backgrounds or very dark meat may cause mis-segmentation (a fallback to the whole image is applied automatically).
- Card detection is contour-based; other rectangular objects (cutting board edge, phone case) could be mistaken for the card.
- `size_threshold_mm = 120` is unvalidated — treat size routing as indicative until confirmed against your actual product sizes.
- The existing 94.2% accuracy confusion matrix was validated on cropped dataset images, not on segmented crops from phone uploads. Re-evaluate on representative upload photos before production use.

### Next steps

- Validate reference-card detection on real phone photos in varied lighting.
- Collect upload-style photos, run them through the pipeline, and build a new confusion matrix to measure real-world accuracy in this deployment mode.

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
models/
  freshness_classifier.pt     Trained CNN weights (auto-saved by train_classifier.py)
src/
  freshness_classifier.py     CNN classifier class — imported by run_on_images.py
  size_classifier.py          Pixel-based size class — imported by run_on_images.py
  __init__.py
scripts/                      ← Everything needed to reproduce training + validation
  prepare_dataset.py          Extract & organise the Kaggle zip
  merge_datasets.py           Mix additional images into train/val split
  collect_data.py             Webcam capture tool for building your own dataset
  train_classifier.py         Trains the freshness CNN (MobileNetV2)
  run_on_images.py            Test runner: runs pipeline on a folder of images
  analyze_results.py          Reads the CSV log and prints accuracy metrics
backend/                      ← Web app (FastAPI server)
  app.py                      FastAPI server — POST /api/classify, GET /
  pipeline.py                 Per-image orchestrator (segment→classify→size→decide)
  segmentation.py             GrabCut meat region isolation
  size_estimator.py           Reference-card calibration + big/small call
  freshness_classifier.py     Same architecture as src/ — loads existing weights
  config.yaml                 Web pipeline config (device, thresholds, card dims)
  requirements.txt            Backend Python dependencies
  test_pipeline.py            Mandatory test suite (all 3 tests pass)
frontend/
  index.html                  Single-file mobile-first UI (drag-drop + result cards)
legacy/                       ← Retired conveyor belt code (not used by web app)
  pipeline_conveyor.py        Original real-time camera loop
  detector.py                 YOLO wrapper (detector was never trained)
  router.py                   Serial / MQTT hardware signal stubs
  tracker.py                  Centroid tracker for belt pieces
  train_yolo.py               YOLO training script
  config_conveyor.yaml        Original full conveyor config
  README.md                   Explains what's here and how to revive it
```

---

## ⚙️ Configuration (`backend/config.yaml`)

All web pipeline parameters are in `backend/config.yaml`:

```yaml
freshness_classifier:
  device: "cuda"                   # "cuda" or "cpu"
  good_confidence_threshold: 0.60  # Safety dial: raise → stricter, lower → permissive

size_classifier:
  size_threshold_mm: 120           # ⚠️ Unvalidated placeholder
  card_width_mm: 85.60             # ID-1 standard card width
  card_aspect_tolerance: 0.18      # ±tolerance for perspective skew
```

---

## 📝 Notes on Production Accuracy

- **Lighting matters more than model choice.** Inconsistent belt lighting will
  hurt the classifier far more than swapping architectures. Use fixed, diffuse,
  consistent-colour-temperature lighting over the inspection zone.
- **`good_confidence_threshold` is your safety dial.** Raising it discards
  more borderline pieces (safer) at the cost of more false rejects.
- **Add your own footage!** Use `scripts/collect_data.py` to capture images
  from your production camera and `scripts/merge_datasets.py` to mix them
  into the training set. This is the single biggest lever for accuracy on
  your specific setup.
- **Data Augmentation (Future Scope):** The `scripts/train_classifier.py` script 
  already uses PyTorch `transforms` to randomly flip, rotate, and color-jitter 
  images during training to teach the model to ignore camera angles and minor 
  lighting shifts. To squeeze out another 1-2% accuracy in the future, you can 
  add `RandomCrop` and `GaussianBlur` to those transforms if your real camera 
  frequently suffers from motion blur or zoom inconsistencies.
