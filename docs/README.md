# Marbl Technical Documentation Library

Welcome to the comprehensive technical documentation repository for **Marbl**, the AI-powered meat freshness quality control and predictive shelf-life inspection pipeline.

---

## Technical Specifications & Whitepapers

| Document | Focus Area | Description |
|---|---|---|
| [**Master Complete System Documentation**](COMPLETE_SYSTEM_DOCUMENTATION.md) | **All-in-One Master Report** | **Single master document answering all 14 project components: workflow, actual screenshots, MobileNetV2 methodology, evaluation, metrology, kinetics, fat analysis, biosensors, validation, timeline, and literature review** |
| [**System Architecture & End-to-End Workflow**](SYSTEM_WORKFLOW_AND_ARCHITECTURE.md) | System Design & Flow | Decoupled client/API/vision/storage architecture diagram and complete lifecycle workflow with explicit rejection branch |
| [**Current vs. Proposed Modules Matrix**](CURRENT_VS_PROPOSED_MODULES.md) | Operational Matrix | Comprehensive status comparison between active production components and staged/future features |
| [**Model Training & Evaluation Report**](MODEL_TRAINING_AND_EVALUATION.md) | AI & Deep Learning | MobileNetV2 architecture, augmentations, 75/25 split, hyperparameters, 94.2% accuracy, confusion matrix, and latency benchmarks |
| [**Size Estimation & Metrological Validation**](SIZE_ESTIMATION_METROLOGY.md) | Computer Vision Metrology | ISO/IEC 7810 ID-1 standard calibration, sub-millimeter formulation, cardless fallback, and digital caliper validation (N=50, MAE 2.1mm) |
| [**Predictive Shelf-Life Assessment Methodology**](PREDICTIVE_SHELF_LIFE_METHODOLOGY.md) | Food Science & Kinetics | Full technical whitepaper on Ratkowsky square-root law, Cardinal pH multipliers, cold-chain thermal integration, and viva voce defense guide |
| [**Visible Fat & Marbling Analysis**](VISIBLE_FAT_ANALYSIS.md) | Colorimetry & Segmentation | Operational HSV/LAB lipid quantification pipeline, proposed MobileNetV3-UNet semantic marbling model, and evaluation criteria |
| [**Dataset Annotation Guide: Fat Segmentation**](fat_annotation_guide.md) | Data Engineering | Practical LabelMe and CVAT polygonal annotation guide for creating 3-class semantic masks |
| [**Biogenic Amine Biosensor Concept**](BIOGENIC_AMINE_SENSOR.md) | Hardware & Biosensing | Screen-printed Diamine Oxidase (DAO) electrochemical strip principle, LMP91000/ESP32 hardware, and dual-modal safety cross-validation |
| [**Experimental Validation Plan & Protocols**](EXPERIMENTAL_VALIDATION_PLAN.md) | Quality Assurance | Certified laboratory protocols benchmarking ISO 4833-1 plate counts, Mitutoyo calipers, incubator trials, and AOAC Soxhlet extraction |
| [**Project Timeline & Academic Literature Review**](PROJECT_TIMELINE_AND_LITERATURE.md) | Roadmap & Research | Multi-phase roadmap (Aug 2026 – Jun 2027) and literature review covering predictive microbiology, computer vision, and biosensors |

---

## Visual Assets Directory

All architectural diagrams, interface screenshots, and sensor renders are cataloged in [`docs/images/`](images/):
- `ui_desktop_intake.png`: Desktop intake console with continuous $pH_u$ slider and shelf-life projection.
- `ui_viewfinder_dropzone.jpg`: Industrial viewfinder upload interface with corner reticles.
- `ui_classification_results.jpg`: Multi-image batch inspection cards showing single-piece and multi-piece routing.
- `sensor_strip.png`: Screen-printed electrochemical sensor strip.

