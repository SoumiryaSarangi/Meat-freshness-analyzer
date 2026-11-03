"""
FastAPI server — Meat QC web pipeline.

Endpoints:
    GET  /                      Serves the frontend UI (../frontend/index.html)
    POST /api/classify          Multi-file upload → JSON results

Usage:
    cd backend
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000 in a browser, or from a phone on the same
network: http://<your-machine-ip>:8000
"""

import asyncio
import base64
import concurrent.futures
import io
import os
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

from freshness_classifier import FreshnessClassifier
from pipeline import process_image
from annotate import annotate_image
import database as db
from shelf_life_engine import predict_shelf_life

# Fat analyser — imported lazily so the server starts even if
# segmentation_models_pytorch is not installed (the module guards itself).
try:
    from fat_analyser import FatAnalyser
    _FAT_ANALYSER_AVAILABLE = True
except ImportError:
    _FAT_ANALYSER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
CONFIG_PATH = BACKEND_DIR / "config.yaml"


def decode_upload_bytes(contents: bytes) -> np.ndarray | None:
    """Decode raw upload bytes to a BGR ndarray, normalising EXIF orientation.

    Gallery photos from phones frequently embed an EXIF orientation tag
    instead of physically rotating pixels — cv2.imdecode ignores this tag,
    producing sideways/upside-down images that confuse GrabCut and the
    freshness classifier.  PIL's exif_transpose() fixes that before we
    ever pass pixels to OpenCV.

    Returns None if the bytes cannot be decoded as an image.
    """
    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img = ImageOps.exif_transpose(pil_img)  # no-op if no EXIF orientation tag
        pil_img = pil_img.convert("RGB")
        
        # Resize huge photos to 800px max (optimal balance for 224px CNN / 256px U-Net & fast web cards)
        MAX_DIM = 800
        w, h = pil_img.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        # Fall back to cv2 for any format PIL cannot open (e.g. corrupted)
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            MAX_DIM = 800
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
_classifier: FreshnessClassifier | None = None
_fat_analyser = None   # FatAnalyser | None  (type not annotated to avoid import error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at server startup and initialize SQLite database."""
    global _classifier, _fat_analyser
    
    # Initialize SQLite tables
    try:
        db.init_db()
        print("[app] SQLite database initialized at data/marbl.db")
    except Exception as exc:
        print(f"[app] ERROR initializing SQLite database: {exc}")

    # utf-8-sig strips the BOM that Windows text editors sometimes write
    with open(CONFIG_PATH, encoding='utf-8-sig') as f:
        cfg = yaml.safe_load(f)
    app.state.cfg = cfg

    fc = cfg["freshness_classifier"]
    weights = str(BACKEND_DIR / fc["weights"]) if not os.path.isabs(fc["weights"]) \
        else fc["weights"]
    # Resolve relative path from the project root if the ../models/ pattern is used
    if not os.path.exists(weights):
        weights = str((BACKEND_DIR / Path(fc["weights"])).resolve())

    _classifier = FreshnessClassifier(
        weights_path=weights,
        input_size=int(fc.get("input_size", 224)),
        device=fc.get("device", "cpu"),
        class_names=fc.get("class_names", ("good", "spoiled")),
        good_confidence_threshold=float(fc.get("good_confidence_threshold", 0.60)),
    )

    # ── Module 3: Fat Analyser ─────────────────────────────────────────────────
    fa_cfg = cfg.get("fat_analyser", {})
    if fa_cfg.get("enabled", False):
        if not _FAT_ANALYSER_AVAILABLE:
            print("[app] WARNING: fat_analyser.enabled=true but 'segmentation_models_pytorch' "
                  "is not installed. Fat analysis disabled. "
                  "Install with: pip install segmentation-models-pytorch")
        else:
            fa_weights = fa_cfg.get("weights", "../models/fat_segmenter.pt")
            fa_weights_path = str(BACKEND_DIR / fa_weights) if not os.path.isabs(fa_weights) \
                else fa_weights
            if not os.path.exists(fa_weights_path):
                fa_weights_path = str((BACKEND_DIR / Path(fa_weights)).resolve())
            if os.path.exists(fa_weights_path):
                _fat_analyser = FatAnalyser(
                    weights_path=fa_weights_path,
                    input_size=int(fa_cfg.get("input_size", 256)),
                    device=fa_cfg.get("device", "cpu"),
                    fat_thresholds=fa_cfg.get("fat_thresholds"),
                )
                print("[app] Fat analyser loaded. Module 3 (visible fat coverage) is active.")
            else:
                print(f"[app] WARNING: fat_analyser.enabled=true but weights not found at "
                      f"'{fa_weights_path}'. Run scripts/train_fat_segmenter.py first.")
    else:
        print("[app] Fat analyser disabled (fat_analyser.enabled: false in config.yaml).")

    print("[app] Models loaded. Server ready.")
    yield
    print("[app] Shutdown.")


app = FastAPI(
    title="Meat QC API",
    description="Upload meat photos; get freshness classification + routing decisions.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

MOBILE_UA_KEYWORDS = ("iphone", "android", "ipad", "mobile")

@app.get("/", include_in_schema=False)
async def index(request: Request):
    """Serve the appropriate frontend based on User-Agent or ?view override."""
    view = request.query_params.get("view")
    if view not in ("mobile", "desktop"):
        ua = request.headers.get("user-agent", "").lower()
        view = "mobile" if any(k in ua for k in MOBILE_UA_KEYWORDS) else "desktop"
    
    filename = "mobile.html" if view == "mobile" else "desktop.html"
    file_path = FRONTEND_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"frontend/{filename} not found")
    
    return FileResponse(str(file_path), media_type="text/html")


# ---------------------------------------------------------------------------
# Module 1: Batch Registration & Shelf-Life API
# ---------------------------------------------------------------------------

@app.post("/api/shelf-life/estimate")
async def estimate_shelf_life_endpoint(payload: dict = Body(...)):
    """Calculate predictive shelf life in real time as user edits intake form."""
    try:
        pred = predict_shelf_life(
            meat_type=payload.get("meat_type", "beef"),
            initial_ph=float(payload.get("initial_ph", 5.6)),
            receiving_temp_c=float(payload.get("receiving_temp_c", 2.0)),
            time_since_slaughter_hrs=float(payload.get("time_since_slaughter_hrs", 24.0)),
            transport_temp_avg_c=float(payload.get("transport_temp_avg_c", 2.0)),
            transport_duration_hrs=float(payload.get("transport_duration_hrs", 4.0)),
            transport_excursion=bool(payload.get("transport_excursion", False)),
            packaging_type=payload.get("packaging_type", "air"),
        )
        return JSONResponse(content={
            "shelf_life_days": pred.shelf_life_days,
            "shelf_life_min_days": pred.shelf_life_min_days,
            "shelf_life_max_days": pred.shelf_life_max_days,
            "confidence_score": pred.confidence_score,
            "recommended_temp_c": pred.recommended_temp_c,
            "recommended_conditions": pred.recommended_conditions,
            "warnings": pred.warnings,
            "growth_rate_per_day": pred.growth_rate_per_day,
            "current_log_cfu": pred.current_log_cfu,
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Shelf-life calculation error: {exc}")


@app.post("/api/batch/create")
async def create_batch_endpoint(payload: dict = Body(...)):
    """Register incoming meat batch (Module 1) and calculate shelf life."""
    required = ["meat_type", "initial_ph", "receiving_temp_c", "initial_weight_val", "packaging_type"]
    missing = [k for k in required if k not in payload or payload[k] is None]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required intake fields: {', '.join(missing)}")

    # 1. Compute predictive shelf life
    pred = predict_shelf_life(
        meat_type=payload.get("meat_type", "beef"),
        initial_ph=float(payload.get("initial_ph", 5.6)),
        receiving_temp_c=float(payload.get("receiving_temp_c", 2.0)),
        time_since_slaughter_hrs=float(payload.get("time_since_slaughter_hrs", 24.0)),
        transport_temp_avg_c=float(payload.get("transport_temp_avg_c", 2.0)),
        transport_duration_hrs=float(payload.get("transport_duration_hrs", 4.0)),
        transport_excursion=bool(payload.get("transport_excursion", False)),
        packaging_type=payload.get("packaging_type", "air"),
    )

    # 2. Prepare database record
    batch_data = {
        "batch_id": payload.get("batch_id") or db.generate_batch_id(),
        "meat_type": payload["meat_type"],
        "initial_ph": float(payload["initial_ph"]),
        "receiving_temp_c": float(payload["receiving_temp_c"]),
        "time_since_slaughter_hrs": float(payload.get("time_since_slaughter_hrs", 24.0)),
        "transport_temp_avg_c": float(payload.get("transport_temp_avg_c", 2.0)),
        "transport_duration_hrs": float(payload.get("transport_duration_hrs", 4.0)),
        "transport_excursion": bool(payload.get("transport_excursion", False)),
        "packaging_type": payload["packaging_type"],
        "initial_weight_val": float(payload["initial_weight_val"]),
        "initial_weight_unit": payload.get("initial_weight_unit", "kg"),
        "shelf_life_days": pred.shelf_life_days,
        "shelf_life_min_days": pred.shelf_life_min_days,
        "shelf_life_max_days": pred.shelf_life_max_days,
        "confidence_score": pred.confidence_score,
        "recommended_temp_c": pred.recommended_temp_c,
        "warnings": pred.warnings,
        "status": "intake_complete",
    }

    try:
        created = db.create_batch(batch_data)
        return JSONResponse(content={"batch": created, "shelf_life": {
            "shelf_life_days": pred.shelf_life_days,
            "shelf_life_min_days": pred.shelf_life_min_days,
            "shelf_life_max_days": pred.shelf_life_max_days,
            "recommended_temp_c": pred.recommended_temp_c,
            "recommended_conditions": pred.recommended_conditions,
            "warnings": pred.warnings,
        }})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database save error: {exc}")


@app.get("/api/batch/{batch_id}")
async def get_batch_endpoint(batch_id: str):
    """Retrieve details for a specific batch."""
    batch = db.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    return JSONResponse(content={"batch": batch})


@app.get("/api/batches")
async def list_batches_endpoint():
    """List recent registered batches."""
    batches = db.list_batches(limit=50)
    return JSONResponse(content={"batches": batches})


@app.get("/api/batch/{batch_id}/records")
async def get_batch_records_endpoint(batch_id: str):
    """Get all visual inspection records for a batch."""
    records = db.get_batch_inspection_records(batch_id)
    return JSONResponse(content={"records": records})


# ---------------------------------------------------------------------------
# Module 2 & 3: Visual Inspection API (Linked to Active Batch)
# ---------------------------------------------------------------------------

@app.post("/api/classify")
async def classify(
    files: list[UploadFile] = File(...),
    batch_id: str | None = Form(None),
):
    """Classify one or more uploaded meat photos.

    Returns a JSON object with a 'results' list, one entry per uploaded file,
    in the same order as the files were uploaded. If batch_id is provided,
    results are archived into the database under that batch.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    cfg = app.state.cfg

    # Read uploaded file bytes concurrently
    upload_payloads = []
    for upload in files:
        raw = await upload.read()
        upload_payloads.append((upload.filename or "upload", raw))

    def _process_single_image(item):
        fname, raw_bytes = item
        img = decode_upload_bytes(raw_bytes)
        if img is None:
            return {
                "filename": fname,
                "error": "Could not decode image. Please upload a valid JPEG/PNG.",
            }

        try:
            res = process_image(
                image_bgr=img,
                filename=fname,
                classifier=_classifier,
                cfg=cfg,
                fat_analyser=_fat_analyser,
            )
        except Exception as exc:
            return {
                "filename": fname,
                "error": f"Processing error: {exc}",
            }

        # Annotate
        try:
            ann_bytes = annotate_image(img, res)
            if ann_bytes is not None:
                res["annotated_image_base64"] = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(ann_bytes).decode("ascii")
                )
            else:
                res["annotated_image_base64"] = None
        except Exception:
            res["annotated_image_base64"] = None

        return res

    # Run image batch concurrently across thread pool
    loop = asyncio.get_running_loop()
    workers = min(4, len(upload_payloads) or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [loop.run_in_executor(executor, _process_single_image, item) for item in upload_payloads]
        results = await asyncio.gather(*futures)

    # Archive to SQLite in sequential order if batch_id was provided
    if batch_id:
        for r in results:
            if not r.get("error"):
                r["batch_id"] = batch_id
                try:
                    db.add_inspection_record(batch_id, r)
                except Exception as db_exc:
                    print(f"[app] Warning: failed to save record to DB: {db_exc}")

    return JSONResponse(content={"results": results, "batch_id": batch_id})


@app.get("/health")
async def health():
    """Quick liveness check."""
    return {"status": "ok", "model_loaded": _classifier is not None,
            "fat_analyser_loaded": _fat_analyser is not None}
