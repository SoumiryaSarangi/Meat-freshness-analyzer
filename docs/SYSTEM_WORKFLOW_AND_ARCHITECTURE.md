# System Architecture & End-to-End Workflow

**Project:** Marbl — AI Meat Quality Control & Predictive Shelf-Life Inspection Pipeline  
**Document:** System Architecture & Workflow Specification  

---

## 1. System Architecture Overview

Marbl is designed as a decoupled, high-performance system connecting the edge browser interface, an asynchronous FastAPI application server, computer vision pipelines, deep learning inference engines, and persistent SQLite database storage.

```mermaid
graph TB
    subgraph Client_Layer ["Client Tier (Browser / Mobile / Terminal)"]
        UI["Responsive Web UI<br/>(desktop.html / mobile.html)"]
        JS["Client Kinetics & State Engine<br/>(shared.js)"]
        Canvas["HTML5 Canvas<br/>(Pre-Scale & Overlays)"]
    end

    subgraph API_Layer ["API & Orchestration Tier (FastAPI / Uvicorn)"]
        Router["HTTP Endpoints<br/>(/api/classify, /api/batches, /health)"]
        BatchCtrl["Batch Controller &<br/>Provenance Logger"]
        AsyncPool["ThreadPoolExecutor<br/>(Concurrent Batch Processing)"]
    end

    subgraph Processing_Layer ["Core Analytics & Vision Tier"]
        KinEngine["Kinetics Engine<br/>(shelf_life_engine.py)"]
        EXIF["EXIF & Geometric Normalization<br/>(1280px Scale)"]
        SegFilter{"Edge-Color Heuristic<br/>(Full Tray vs. Single Cut)"}
        GrabCut["Iterative GrabCut Segmentation<br/>(Unsupervised Masking)"]
        CV_Size["Reference Card Calibrator<br/>(ISO ID-1 Contour Detection)"]
        FatCV["Colorimetric Fat Analyzer<br/>(HSV/LAB Thresholding)"]
        MobileNet["MobileNetV2 Classifier<br/>(PyTorch / TorchScript)"]
    end

    subgraph Storage_Layer ["Persistence Tier"]
        DB[("SQLite Batches Vault<br/>(data/marbl.db)")]
        Weights[("Model Weights<br/>(freshness_classifier.pt)")]
    end

    UI -->|"HTTP POST /api/classify"| Router
    UI -->|"HTTP POST /api/batches"| Router
    JS <-->|"Instant Mathematical Calculation"| UI
    Canvas -->|"Pre-Scaled Image Blobs"| UI

    Router --> AsyncPool
    Router --> BatchCtrl
    BatchCtrl --> DB

    AsyncPool --> EXIF
    EXIF --> SegFilter
    SegFilter -->|"Single Piece"| GrabCut
    SegFilter -->|"Full Tray / Macro"| MobileNet
    GrabCut --> MobileNet
    GrabCut --> FatCV
    EXIF --> CV_Size
    MobileNet --> Router
    CV_Size --> Router
    FatCV --> Router
    Weights -.->|"Loads Weights"| MobileNet
```

### Architectural Tier Breakdown

1. **Client Tier (`frontend/`)**:
   - **`desktop.html`**: Dual-column industrial interface for desktop workstations and fixed QA desks. Incorporates real-time intake kinetics, the industrial viewfinder dropzone, and inspection results.
   - **`mobile.html`**: Handheld terminal view designed for loading docks and cold stores, featuring camera-first capture, batch registration, and touch-optimized results.
   - **`shared.js`**: Shared client-side library containing the interactive Ratkowsky kinetic shelf-life engine, client-side canvas pre-scaler (`optimizeImageForUpload`), and API communication helpers.

2. **API & Orchestration Tier (`backend/app.py`)**:
   - Built on **FastAPI** with ASGI runner **Uvicorn**.
   - Asynchronous request handling with thread pool delegation (`ThreadPoolExecutor`) for CPU/GPU-bound computer vision tasks.
   - Endpoints:
     - `POST /api/classify`: Multi-image batch inspection.
     - `POST /api/batches`: Batch provenance and intake logging.
     - `GET /api/batches`: Historical audit trail retrieval.
     - `GET /health`: Liveness probe and PyTorch model status.

3. **Core Analytics & Vision Tier**:
   - **`shelf_life_engine.py`**: Server-side Ratkowsky square-root microbial growth model with Cardinal pH multipliers and packaging atmosphere inhibition factors.
   - **`segmentation.py`**: Unsupervised GrabCut background isolation with edge-color heuristic safeguarding against macro tray over-segmentation.
   - **`freshness_classifier.py`**: MobileNetV2 convolutional neural network delivering binary freshness decisions (`good` vs. `spoiled`) with calibrated confidence scores.
   - **`size_estimator.py`**: Sub-millimeter metric estimation using ISO/IEC 7810 ID-1 reference card calibration targets, with automatic area fallback when no card is present.
   - **`annotate.py`**: Visual annotation renderer drawing bounding boxes, status pills, and dimensions on output images.

4. **Persistence Tier (`data/marbl.db`)**:
   - SQLite relational store logging batch identifiers, species, slaughter timestamps, cold-chain profiles, predicted shelf-life days, and visual inspection summaries.

---

## 2. End-to-End System Workflow & Decision Logic

The complete inspection journey follows a deterministic, food-safety-first pipeline from initial meat arrival to final disposition.

```mermaid
flowchart TD
    Start(["Meat Batch Arrival at Facility"]) --> BReg["1. Batch Registration<br/>(Batch ID, Species, Slaughter Date, Transit Temp, Packaging)"]
    
    BReg --> KinCalc["2. Kinetic Shelf-Life Engine<br/>(Ratkowsky Law + Cardinal pH + Packaging Multiplier)"]
    
    KinCalc --> CCPCheck{"Cold Chain Exceeded<br/>or Shelf-Life ≤ 0?"}
    CCPCheck -- "Yes (Violated)" --> RejectIntake["❌ REJECT AT INTAKE<br/>Cold-chain failure flag / Logged to DB"]
    
    CCPCheck -- "No (Safe)" --> VisualScan["3. Optical Inspection Acquisition<br/>(Camera / Upload with ID-1 Reference Card)"]
    
    VisualScan --> CanvasScale["Client Pre-Scale to 1280px<br/>(Optimized Payload)"]
    CanvasScale --> Decode["4. Backend EXIF Normalize & Decode"]
    
    Decode --> EdgeCheck{"Edge-Color Heuristic<br/>>20% border meat-colored?"}
    EdgeCheck -- "Yes (Macro/Full Tray)" --> SkipSeg["Bypass GrabCut<br/>(Use Full Image)"]
    EdgeCheck -- "No (Isolated Cut)" --> GrabCut["GrabCut 5-Iteration<br/>Foreground Masking"]
    
    SkipSeg --> Classify["5. MobileNetV2 Freshness Inference"]
    GrabCut --> Classify
    GrabCut --> FatAnalysis["6. Visible Fat Analysis<br/>(HSV/LAB Colorimetry)"]
    
    Decode --> CardDetect["7. ID-1 Reference Card Detection<br/>(Polygon contour extraction)"]
    CardDetect --> SizeCalc["Compute Metric Sizing (mm)"]
    
    Classify --> QualityBranch{"Freshness<br/>Verdict?"}
    
    QualityBranch -- "SPOILED<br/>(Confidence ≥ 60%)" --> Discard["❌ DISCARD DECISION<br/>Red bounding box / Food safety reject / Halt line"]
    
    QualityBranch -- "GOOD<br/>(Confidence > 60%)" --> SizeBranch{"Physical Size &<br/>Piece Count"}
    
    SizeBranch -- "Multi-Piece Cluster<br/>or Size < 120mm" --> Grinding["🔄 ROUTE: GRINDING<br/>Minced meat / Burger patties / Sausages"]
    
    SizeBranch -- "Single Piece<br/>and Size ≥ 120mm" --> Packing["📦 ROUTE: PACKING<br/>Primal retail trays / Steaks / Vacuum packaging"]
    
    Discard --> LogAudit["8. Log Inspection Results & Annotated Overlays to DB"]
    Grinding --> LogAudit
    Packing --> LogAudit
    RejectIntake --> LogAudit
    
    LogAudit --> EndNode(["Batch Processed & Archived"])
```

### Decision Logic Rules

| Output Class | Condition | Downstream Operational Action |
|---|---|---|
| **DISCARD** | MobileNetV2 `spoiled` verdict OR confidence on `good` < 60% | Meat is rejected immediately. Red bounding box displayed. Product is diverted to inedible waste or rendering. |
| **GRINDING** | Freshness is `good`, AND (longest dimension < 120 mm OR multi-piece cluster detected) | Meat is high quality but too small or irregular for retail steak trays. Routed to grinding line for mince, burgers, or sausage production. |
| **PACKING** | Freshness is `good`, longest dimension ≥ 120 mm, and single cut | Prime cut meeting retail specifications. Routed to automated tray wrapping, vacuum skin packaging, or prime labeling. |

