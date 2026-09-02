# Meat QC Pipeline

Upload a photo of meat (from phone or laptop) and get an instant freshness verdict (**good** or **spoiled**) with confidence percentage, plus a routing decision (**discard** / **grinding** / **packing**) based on estimated size. Powered by a MobileNetV2 classifier trained on 3,300+ labeled images, served by a FastAPI backend with a mobile-first drag-and-drop UI.

---

## 🌐 Run the Web App

```bash
# 1. Install backend dependencies
cd backend
pip install -r requirements.txt

# 2. Start the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in a browser.
From a phone on the same Wi-Fi: **http://\<your-machine-ip\>:8000**

Upload one or more photos. For each photo the server returns:
- ✅ **Good** / ❌ **Spoiled** with confidence %
- Routing decision: **Discard** / **Grinding** / **Packing**
- Size method: **Measured** (reference card found) or **Estimated** (fallback)

### Size calibration

Place a standard **credit / debit / ID card** (85.60 mm wide) flat next to the meat in the photo. The server detects the card's rectangular shape, derives pixels-per-mm for that specific photo, and marks the size result as **"Measured"**. Without a card it falls back to a frame-area estimate marked **"Estimated"**.

### How the architecture works

This project was originally designed for a real-time conveyor belt with a YOLO detector and fixed-height camera. The deployment target changed to phone/laptop uploads, so the architecture was reworked:

| Original (conveyor) | Current (web upload) |
|---|---|
| YOLO detector (needs training data) | GrabCut segmentation (works instantly, no training) |
| Fixed `pixels_per_mm` (camera at known height) | Reference-card per-photo calibration |
| Real-time camera loop + hardware router | Per-request HTTP: one image → one JSON response |

The retired conveyor code is preserved in `legacy/` for reference.

---

## ⚠️ Known Limitations

- **GrabCut segmentation** segments by colour/texture contrast — unusual backgrounds or very dark meat may cause mis-segmentation (a fallback to the whole image is applied automatically).
- **Card detection** is contour-based; other rectangular objects (cutting board edge, phone case) could be mistaken for the reference card.
- **`size_threshold_mm = 120`** is an unvalidated placeholder — treat size routing as indicative until confirmed against your actual product sizes.
- **Accuracy figures are unverified against data leakage.** The 94.2% accuracy below was measured on a val split produced by `merge_datasets.py`, which does a plain random shuffle before splitting. If the source dataset contains near-duplicate frames (e.g. augmented copies of the same photo), some may land on both sides of the train/val boundary, inflating the reported accuracy. Treat these numbers as an upper bound pending a proper leakage audit.
- **Validation was on cropped dataset images**, not on segmented crops from phone uploads. Real-world accuracy on arbitrary phone photos may differ.
- **Multi-piece tray detection is a heuristic.** When the uploaded photo contains multiple pieces (e.g. a tray of diced chunks), the app automatically routes the batch to grinding instead of attempting a per-piece size measurement. The detection uses a distance-transform peak-counting algorithm — not an exact count — tuned using synthetic test images. The sensitivity threshold (`multi_piece_dist_threshold: 0.4` in `backend/config.yaml`) should be re-tuned against your own real tray photos before relying on it in production. Lower values (e.g. `0.3`) catch more subtle separations; higher values (e.g. `0.5`) reduce false positives on single pieces with uneven surfaces. True per-piece automatic measurement on tray photos is a larger future upgrade.

---

## 📊 Model Performance (unverified — see caveat above)

Trained on a **combined dataset** of 3,331 images (Kaggle Meat Freshness + custom footage), evaluated on 831 held-out validation images:

| Metric | Value |
|---|---|
| Accuracy | 94.2% |
| Precision | 95.7% |
| Recall | 91.0% |
| F1 Score | 0.933 |
| Spoilage detection rate | 96.7% |

---

## 🚀 Reproduce Training from Scratch

Follow these steps **in order** to go from a fresh clone to a trained model.

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
> PyTorch for 5× faster training:
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

1. Go to [Kaggle — Meat Freshness Image Dataset](https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset)
2. Click **Download** → you get `archive.zip` (≈ 75 MB)
3. Place the zip anywhere on your machine (e.g. `Downloads/archive.zip`)

### Step 4 — Prepare the dataset

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
> HALF-FRESH images as good rather than spoiled.

### Step 5 — (Optional) Add your own images to fix domain shift

If your photos look different from the Kaggle dataset (different lighting,
background, camera angle), mix in your own labeled images:

```bash
python scripts/merge_datasets.py \
    --fresh  "path/to/Fresh" \
    --spoiled "path/to/Spoiled"
```

> **Note:** `merge_datasets.py` does a plain `random.shuffle` before splitting
> 80/20 into train/val. If your source images include augmented near-duplicates,
> this can cause data leakage across the split boundary.

### Step 6 — Train the freshness classifier

```bash
# CPU only (slow, ~30 min):
python scripts/train_classifier.py --data data/dataset_freshness --epochs 30

# NVIDIA GPU (recommended, ~6-12 min):
python scripts/train_classifier.py --data data/dataset_freshness --epochs 30 --batch 64
```

The best checkpoint is saved automatically to `models/freshness_classifier.pt`.

### Step 7 — Validate the model

Run the classifier on the entire validation folder:

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
scripts/                      ← Everything needed to reproduce training + validation
  prepare_dataset.py          Extract & organise the Kaggle zip
  merge_datasets.py           Mix additional images into train/val split
  collect_data.py             Webcam capture tool for building your own dataset
  train_classifier.py         Trains the freshness CNN (MobileNetV2)
  run_on_images.py            Freshness evaluation on a folder of images
  analyze_results.py          Reads the CSV log and prints accuracy metrics
backend/                      ← Web app (FastAPI server)
  app.py                      FastAPI server — POST /api/classify, GET /
  pipeline.py                 Per-image orchestrator (segment→classify→size→decide)
  segmentation.py             GrabCut meat region isolation
  size_estimator.py           Reference-card calibration + big/small call
  freshness_classifier.py     MobileNetV2 classifier (also used by scripts/)
  config.yaml                 Web pipeline config (device, thresholds, card dims)
  requirements.txt            Backend Python dependencies
  test_pipeline.py            Mandatory test suite (3 tests)
frontend/
  index.html                  Single-file mobile-first UI (drag-drop + result cards)
legacy/                       ← Retired conveyor belt code (not used by web app)
  pipeline_conveyor.py        Original real-time camera loop
  detector.py                 YOLO wrapper (detector was never trained)
  router.py                   Serial / MQTT hardware signal stubs
  tracker.py                  Centroid tracker for belt pieces
  train_yolo.py               YOLO training script
  config_conveyor.yaml        Original full conveyor config
  requirements.txt            Conveyor-only dependencies (ultralytics, pyserial, paho-mqtt)
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
  fallback_area_threshold: 0.35    # Area fraction above which fallback = "big"
  multi_piece_dist_threshold: 0.4  # Piece-count sensitivity (see Known Limitations)
```

---

## 📝 Tips

- **Lighting matters more than model choice.** Inconsistent lighting will hurt the classifier far more than swapping architectures. Use fixed, diffuse, consistent-colour-temperature lighting.
- **`good_confidence_threshold` is your safety dial.** Raising it discards more borderline pieces (safer) at the cost of more false rejects.
- **Add your own photos!** Use `scripts/collect_data.py` to capture images from your camera and `scripts/merge_datasets.py` to mix them into the training set. This is the single biggest lever for accuracy on your specific setup.
- **Data Augmentation (Future Scope):** The `scripts/train_classifier.py` script already uses PyTorch `transforms` to randomly flip, rotate, and colour-jitter images during training. To squeeze out another 1-2% accuracy, you can add `RandomCrop` and `GaussianBlur` to those transforms if your photos frequently suffer from motion blur or zoom inconsistencies.

---

## 📷 Camera Capture & HTTPS

The web app supports two input methods — file upload and live camera capture — both feeding the same classification pipeline.

### HTTP vs HTTPS — what works where

| Scenario | File Upload | Camera Capture |
|---|---|---|
| `http://localhost:8000` (desktop) | ✅ | ✅ |
| `http://<lan-ip>:8000` (phone on same Wi-Fi) | ✅ | ❌ browser blocks |
| `https://` any origin | ✅ | ✅ |

**Why?** Browsers enforce that `getUserMedia` (camera access) requires a [Secure Context](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts). `localhost` is an exemption; any other plain-HTTP origin is not.

### Testing on a real phone before production

The fastest path is an HTTPS tunnel — no certificate setup needed:

```bash
# Install ngrok once: https://ngrok.com/download
ngrok http 8000
```

Open the `https://xxxxx.ngrok-free.app` URL it prints on your phone. The rear camera will open by default.

### Production deployment

For a permanently reachable service, you need real HTTPS — either:
- A reverse proxy (nginx / Caddy) in front of uvicorn with a TLS certificate (Let's Encrypt is free), or
- A hosting platform that provides HTTPS automatically (Railway, Render, Fly.io, etc.).

Plain `http://` on a public IP will get camera permission silently refused by the browser — this is the browser's security policy, not a bug in this app.

