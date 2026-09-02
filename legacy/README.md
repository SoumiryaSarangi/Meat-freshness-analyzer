# Legacy — Conveyor Belt Pipeline

This folder contains code from the **original real-time conveyor-belt design**
that was superseded when the project pivoted to an upload-based web app.

These files are kept for reference and in case you want to revive the
conveyor pipeline in the future. **None of these files are imported or used
by the current web app.**

## What's here and why it was retired

| File | Original purpose | Why retired |
|---|---|---|
| `pipeline_conveyor.py` | Main real-time camera loop | Replaced by `backend/app.py` (request-based) |
| `detector.py` | YOLO meat detector wrapper | YOLO was never trained; web app uses GrabCut instead |
| `router.py` | Serial / MQTT hardware signalling | Not needed for web uploads |
| `tracker.py` | Centroid tracker (process each belt piece once) | Not needed without a live camera feed |
| `train_yolo.py` | YOLO training script | YOLO training was never completed |
| `config_conveyor.yaml` | Full conveyor pipeline config | Replaced by `backend/config.yaml` |

## To revive the conveyor pipeline

1. Train the YOLO detector:
   ```bash
   python legacy/train_yolo.py --data data/dataset_yolo/dataset.yaml --epochs 100
   ```
2. Copy `legacy/config_conveyor.yaml` back to `config.yaml` and fill in
   the correct `pixels_per_mm` calibration for your camera.
3. Run:
   ```bash
   python -m legacy.pipeline_conveyor --config config.yaml
   ```
