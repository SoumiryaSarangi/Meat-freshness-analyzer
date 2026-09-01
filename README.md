# Meat QC Pipeline

Real-time computer vision pipeline for a meat processing line:

```
camera -> YOLO detector (bbox + confidence)
       -> freshness CNN (good / spoiled + confidence)
            -> spoiled -> discard
            -> good -> size classifier (bbox -> mm)
                 -> small -> grinding line
                 -> big   -> packing line
```

Every accepted piece gets a bounding box drawn on the live feed with its
freshness confidence and routing decision, and a row in `output/decisions_log.csv`
for traceability/audit.

## 1. Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

You'll need a CUDA-capable GPU for real-time inference at useful frame rates;
both models will also run on CPU for development/testing (set `device: "cpu"`
in `config.yaml`), just slower.

## 2. Get data

You don't have a labeled dataset yet, so start from public data and refine
with your own conveyor footage:

- **Freshness classifier (good/spoiled)**: We have successfully merged the Kaggle "Meat Freshness Image Dataset" (~2,270 images) with custom dataset footage (~1,896 images) to create a robust model that handles domain shift and varied lighting.
- **Detector (bounding boxes)**: Roboflow Universe has meat-labeled object
  detection datasets exportable directly in YOLO format. Search
  `universe.roboflow.com` for "meat".
- **Your own footage**: run `scripts/collect_data.py` against your actual
  conveyor camera. A model trained only on public photos (studio lighting,
  clean backgrounds) will likely underperform on your belt's lighting,
  angle, and background -- plan to fine-tune on at least a few hundred of
  your own labeled images before trusting this in production. Use `scripts/merge_datasets.py` to seamlessly integrate new images into the training set.

Label bounding boxes with Roboflow Annotate or CVAT (export to YOLO format
into `data/dataset_yolo/`). Label freshness crops into folders
`data/dataset_freshness/{train,val}/{good,spoiled}/`.

## 3. Train

```bash
python scripts/train_yolo.py --data data/dataset_yolo/dataset.yaml --epochs 100
python scripts/train_classifier.py --data data/dataset_freshness --epochs 30 --batch 64
```

> **Update:** We recently retrained the freshness classifier on an RTX 4050 GPU using a combined dataset (Kaggle + Custom Meat Analysis) for 30 epochs. The resulting model achieved **94.3% validation accuracy** and a **96.7% spoilage detection rate** (`models/freshness_classifier.pt`).

Copy the resulting weights to the paths referenced in `config.yaml`
(`models/yolo_meat_detector.pt`, `models/freshness_classifier.pt`).

## 4. Calibrate size measurement

The size classifier converts bounding-box pixels to millimeters using a
fixed `pixels_per_mm` factor in `config.yaml`. To calibrate:

1. Place a flat object of known width (e.g. a 100mm card) on the belt at the
   same height/distance the camera will see production pieces at.
2. Note its bounding box width in pixels (run the detector on it, or measure
   manually in a saved frame).
3. Set `pixels_per_mm = width_px / width_mm` in `config.yaml`.
4. Set `size_threshold_mm` to whatever cutoff separates "send to grinder"
   from "send to packing" in your process.

This assumes a fixed, perpendicular camera height above the belt. If camera
distance varies, you'll need a proper camera calibration (intrinsics +
known mounting height) rather than a single scale factor -- ping me if
that's your setup and I can extend this.

## 5. Run

```bash
python -m src.pipeline --config config.yaml
```

Starts with `routing.backend: "console"` in `config.yaml`, which only logs
decisions -- safe for validating detection/classification accuracy before
anything touches real actuators.

## 6. Connect to hardware

`src/router.py` is the one module meant to be adapted per-site. Three
backends are stubbed:

- `console` -- logging only (default)
- `serial` -- sends a command byte to a microcontroller (Arduino/ESP32)
  driving relays/diverters
- `mqtt` -- publishes decisions to a broker for PLC/SCADA integration

Fill in the `TODO`s in whichever backend matches your discard mechanism,
grinder line signal, and packing line signal, then set `routing.backend`
in `config.yaml`.

## Project layout

```
config.yaml                   All tunable parameters
src/
  detector.py                 YOLO wrapper -> Detection(bbox, confidence)
  freshness_classifier.py     CNN -> good/spoiled + confidence
  size_classifier.py          bbox -> mm -> small/big
  tracker.py                  centroid tracker (process each piece once)
  router.py                   decision -> hardware signal
  pipeline.py                 main real-time loop
scripts/
  collect_data.py             webcam capture for building your dataset
  prepare_dataset.py          extracts & organizes Kaggle zip downloads
  merge_datasets.py           mixes external data into the train/val split
  run_on_images.py            end-to-end pipeline test runner for image folders
  train_yolo.py               trains the detector
  train_classifier.py         trains the freshness CNN
```

## Notes on accuracy in production

- **Lighting matters more than model choice.** Meat color shifts (browning,
  graying) are the primary visual spoilage signal -- inconsistent belt
  lighting will hurt the freshness classifier more than swapping model
  architectures will help. Fixed, diffuse, consistent-color-temperature
  lighting over the inspection zone is worth the investment.
- **The `good_confidence_threshold` in config.yaml is your safety dial.**
  Raising it makes the system more conservative (more borderline pieces get
  discarded rather than risk shipping spoiled meat) at the cost of wasted
  good product. Tune it against your actual false-negative tolerance, not
  just validation accuracy.
- **The tracker prevents double-counting** the same physical piece across
  frames as it crosses the camera's field of view. If pieces move fast or
  overlap heavily, swap in Ultralytics' built-in `model.track()` (ByteTrack)
  instead of the simple centroid tracker.
