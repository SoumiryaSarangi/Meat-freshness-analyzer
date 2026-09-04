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

import base64
import io
import os
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

from freshness_classifier import FreshnessClassifier
from pipeline import process_image
from annotate import annotate_image

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
        
        # Resize huge gallery photos to speed up GrabCut (~20s -> ~0.2s)
        # Size measurement is scale-invariant (relative to card or image area)
        MAX_DIM = 1280
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
            MAX_DIM = 1280
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return img

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
_classifier: FreshnessClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at server startup."""
    global _classifier
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


@app.post("/api/classify")
async def classify(files: list[UploadFile] = File(...)):
    """Classify one or more uploaded meat photos.

    Returns a JSON object with a 'results' list, one entry per uploaded file,
    in the same order as the files were uploaded.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    cfg = app.state.cfg
    results = []

    for upload in files:
        raw = await upload.read()
        image = decode_upload_bytes(raw)

        if image is None:
            results.append({
                "filename": upload.filename,
                "error": "Could not decode image. Please upload a valid JPEG/PNG.",
            })
            continue

        try:
            result = process_image(
                image_bgr=image,
                filename=upload.filename or "upload",
                classifier=_classifier,
                cfg=cfg,
            )
        except Exception as exc:
            results.append({
                "filename": upload.filename,
                "error": f"Processing error: {exc}",
            })
            continue

        # ── Annotate: draw results onto a resized copy of the image ──
        try:
            ann_bytes = annotate_image(image, result)
            if ann_bytes is not None:
                result["annotated_image_base64"] = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(ann_bytes).decode("ascii")
                )
            else:
                result["annotated_image_base64"] = None
        except Exception:
            result["annotated_image_base64"] = None

        results.append(result)

    return JSONResponse(content={"results": results})


@app.get("/health")
async def health():
    """Quick liveness check."""
    return {"status": "ok", "model_loaded": _classifier is not None}
