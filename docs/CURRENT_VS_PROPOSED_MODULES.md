# Current vs. Proposed Modules Specification

**Project:** Marbl — AI Meat Quality Control & Predictive Shelf-Life Inspection Pipeline  
**Document:** System Module Operational Matrix  

---

## 1. Operational Status Overview

Marbl combines operational production modules with actively researched advanced features. The matrix below defines what is currently active in the codebase versus proposed or staged modules.

| Subsystem / Feature | Module Scope | Status | Technical Implementation | Operational Maturity | Target Horizon |
|---|---|---|---|---|---|
| **Batch Intake Registration** | Module 1 | ✅ **Fully Operational** | Species selection, post-mortem $pH_u$, cold-chain transit history, packaging atmosphere | Production Ready | Active (v1.0) |
| **Ratkowsky Kinetic Shelf-Life Engine** | Module 1 | ✅ **Fully Operational** | Square-root growth rate law integrated with Cardinal pH and packaging atmosphere multipliers | Production Ready | Active (v1.0) |
| **Instant Client Mathematical Mirror** | Module 1 | ✅ **Fully Operational** | Pure JavaScript kinetic implementation in `shared.js` enabling 0ms UI slider updates | Production Ready | Active (v1.0) |
| **SQLite Batches Vault** | Module 1 | ✅ **Fully Operational** | Relational provenance schema storing timestamps, transit conditions, and shelf-life predictions | Production Ready | Active (v1.0) |
| **MobileNetV2 Freshness Classifier** | Module 2 | ✅ **Fully Operational** | PyTorch CNN trained on 3,331 images (94.2% validation accuracy; 96.8% spoilage recall) | Production Ready | Active (v1.0) |
| **GrabCut Foreground Isolation** | Module 2 | ✅ **Fully Operational** | Iterative energy minimization isolating meat contours from trays and cutting boards | Production Ready | Active (v1.0) |
| **Edge-Color Macro Guard** | Module 2 | ✅ **Fully Operational** | Border colorimetry check bypassing GrabCut on close-up cuts to prevent false over-cropping | Production Ready | Active (v1.0) |
| **ISO ID-1 Reference Card Sizing** | Module 2 | ✅ **Fully Operational** | Contour polygon detection of credit-card fiducial targets (85.60 mm) for sub-mm pixel calibration | Production Ready | Active (v1.0) |
| **Automated Routing Decision Logic** | Module 2 | ✅ **Fully Operational** | Rule-based triage matrix routing cuts to `discard`, `grinding`, or `packing` based on freshness and size | Production Ready | Active (v1.0) |
| **Client-Side Upload Optimizer** | Module 2 | ✅ **Fully Operational** | Browser `<canvas>` pre-scaler downsampling large 12–48MP mobile photos to 1280px (98% bandwidth reduction) | Production Ready | Active (v1.0) |
| **Visual Annotation Overlay Engine** | Module 2 | ✅ **Fully Operational** | OpenCV server-side bounding box, status pill, and dimension overlay renderer | Production Ready | Active (v1.0) |
| **Visible Fat Colorimetry** | Module 3 | ✅ **Fully Operational** | GrabCut-masked HSV/LAB color thresholding calculating surface lipid percentage | Production Ready | Active (v1.0) |
| **Deep Semantic Fat Segmentation** | Module 3 | 🔬 *Under Development* | MobileNetV3-UNet / SegFormer architecture; 600-image dataset currently being annotated | Research Prototype | Phase 5 (Q1 2027) |
| **Biogenic Amine Biosensor Strip** | Hardware | 🔬 *Staged / Emulated* | Electrochemical DAO enzyme reaction model, UART/BLE handshake protocol, and UI hardware monitor | Firmware Prototype | Phase 5 (Q2 2027) |
| **Automated Conveyor Diverter** | Hardware | 🔬 *Under Development* | Industrial pneumatic pusher / reject gate triggered via GPIO / MQTT message broker | Hardware Design | Phase 6 (Q2 2027) |

---

## 2. Module Details & Progression Path

### Module 1: Intake & Predictive Shelf-Life Assessment
- **Current State**: Operational in both backend (`shelf_life_engine.py`) and frontend (`shared.js`). Provides immediate predictions of remaining shelf-life days with 95% confidence bounds based on thermal history, post-mortem $pH_u$, and packaging atmosphere.
- **Proposed Enhancements**: Integration with automated IoT Bluetooth temperature loggers (e.g., Dallas 1-Wire or Sensirion NFC tags) attached to meat shipping containers to ingest continuous time-temperature curves directly instead of manual transit logs.

### Module 2: Visual Inspection & Physical Size Routing
- **Current State**: Operational in `backend/freshness_classifier.py`, `backend/segmentation.py`, `backend/size_estimator.py`, and `backend/annotate.py`. Employs MobileNetV2 with GrabCut segmentation and ISO ID-1 reference card calibration.
- **Proposed Enhancements**: Upgrading segmentation from GrabCut to YOLOv8-seg or Mask R-CNN to natively isolate multiple overlapping cuts on commercial sorting trays without heuristic distance transforms.

### Module 3: Visible Fat & Marbling Analysis
- **Current State**: Operational in `backend/fat_analyzer.py` via adaptive HSV/LAB color thresholding within the segmented foreground mask.
- **Proposed Enhancements**: Deep semantic segmentation using MobileNetV3-UNet to discriminate subcutaneous fat cap from intramuscular marbling flecks, enabling automated USDA marbling grade estimation (Prime, Choice, Select).

### Hardware Extension: Biogenic Amine Biosensing
- **Current State**: Mathematical model and UI integration completed. The frontend displays the sensor status, real-time potential, current, and Amine Index, cross-referencing visual predictions against volatile nitrogen / amine accumulation.
- **Proposed Enhancements**: Physical deployment of screen-printed carbon electrodes (SPCE) modified with Diamine Oxidase (DAO) connected to an LMP91000 analog front-end and ESP32 microcontroller over WebSerial or BLE.

