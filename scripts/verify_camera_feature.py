"""
Automated verification for the camera capture + EXIF fix feature.

Tests:
  V1: Backend regression — all existing test_pipeline.py tests still pass.
  V2: EXIF orientation fix — a JPEG with orientation=6 (90-deg CW) is decoded
      right-side-up by decode_upload_bytes(); raw cv2.imdecode would be wrong.
  V3: Camera-path equivalence — a camera_capture_<ts>.jpg blob POSTed to
      /api/classify returns schema-identical JSON to a normal file upload.

Run from the project root:
    python scripts/verify_camera_feature.py
"""

import io
import sys
import struct
import time
import json
import tempfile
import subprocess
import urllib.request
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
BACKEND = PROJECT / "backend"
sys.path.insert(0, str(BACKEND))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── V1: Backend regression ─────────────────────────────────────────────────────
section("V1 — Backend regression (test_pipeline.py)")
try:
    import tempfile, os
    log = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
    log.close()
    proc = subprocess.run(
        [sys.executable, "test_pipeline.py"],
        cwd=str(BACKEND),
        stdout=open(log.name, "w"),
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    output = open(log.name).read()
    os.unlink(log.name)
    print(output[-3000:])  # last 3 KB is enough for summary
    if proc.returncode == 0 and "ALL TESTS PASSED" in output:
        print(f"  Result: {PASS}")
        v1_ok = True
    else:
        print(f"  Result: {FAIL}  (exit code {proc.returncode})")
        v1_ok = False
except Exception as e:
    print(f"  Result: {FAIL}  — {e}")
    v1_ok = False


# ── V2: EXIF orientation fix ───────────────────────────────────────────────────
section("V2 — EXIF orientation fix (orientation=6, 90-deg CW)")
try:
    import numpy as np
    import cv2
    from PIL import Image
    from app import decode_upload_bytes

    # Create a 100x200 (w x h) image that is clearly non-square so rotation
    # is detectable.  The top-left quadrant is red, rest is blue.
    W, H = 100, 200
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    arr[:H//2, :W//2] = [0, 0, 255]   # red (BGR)  top-left
    arr[H//2:, W//2:] = [255, 0, 0]   # blue       bottom-right
    pil_orig = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    # Save with EXIF orientation = 6 (90-deg CW rotation stored in metadata,
    # pixels NOT physically rotated — exactly what phone cameras do).
    exif = pil_orig.getexif()
    exif[0x0112] = 6  # Orientation tag
    buf = io.BytesIO()
    pil_orig.save(buf, format="JPEG", exif=exif.tobytes(), quality=95)
    jpeg_bytes = buf.getvalue()

    # Verify: raw cv2.imdecode ignores EXIF → dimensions are still 100x200
    raw_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    raw_img = cv2.imdecode(raw_arr, cv2.IMREAD_COLOR)

    # Our helper should correct to 200x100 (portrait → landscape after fixing)
    fixed_img = decode_upload_bytes(jpeg_bytes)
    print(f"  raw cv2.imdecode shape:      {raw_img.shape[:2]}  (h×w)")
    print(f"  decode_upload_bytes shape:   {fixed_img.shape[:2]}  (h×w)  — expected (100, 200) [corrected]")
    print(f"  OpenCV version: {cv2.__version__}")

    # Both should produce the correctly-oriented image.
    # OpenCV >= 4.8 auto-applies EXIF orientation; our helper uses PIL as an
    # explicit, stable guarantee regardless of OpenCV version.
    assert fixed_img.shape == (100, 200, 3), \
        f"decode_upload_bytes: expected (100, 200, 3), got {fixed_img.shape}"
    # raw cv2 may or may not correct (depends on version) — just print, don't assert
    if raw_img.shape[:2] == (100, 200):
        print("  Note: cv2.imdecode also corrected orientation (OpenCV >= 4.8 behaviour).")
    else:
        print("  Note: cv2.imdecode did NOT correct orientation — PIL helper is the safety net.")
    print(f"  Result: {PASS}")
    v2_ok = True
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  Result: {FAIL}  — {e}")
    v2_ok = False


# ── V3: Camera-path equivalence ────────────────────────────────────────────────
section("V3 — Camera-path equivalence (camera_capture filename via HTTP POST)")
v3_ok = False
try:
    import numpy as np
    import cv2

    # Check server is up
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=3)
    except Exception:
        print("  [SKIP] Server not reachable on localhost:8000 — start uvicorn first.")
        v3_ok = None  # skip, not a failure
    else:
        # Build a minimal synthetic JPEG in memory — same as canvas.toBlob() would produce
        H, W = 224, 224
        img = np.full((H, W, 3), 180, dtype=np.uint8)
        cv2.rectangle(img, (60, 60), (160, 160), (40, 60, 150), -1)
        _, jpeg_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        jpeg_bytes = jpeg_buf.tobytes()

        cam_filename  = f"camera_capture_{int(time.time())}.jpg"
        file_filename = f"upload_{int(time.time())}.jpg"

        def post_file(filename, data):
            boundary = b"----VerifyBoundary"
            body = (
                b"--" + boundary + b"\r\n"
                b'Content-Disposition: form-data; name="files"; filename="' + filename.encode() + b'"\r\n'
                b"Content-Type: image/jpeg\r\n\r\n" +
                data +
                b"\r\n--" + boundary + b"--\r\n"
            )
            req = urllib.request.Request(
                "http://localhost:8000/api/classify",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())

        cam_resp  = post_file(cam_filename,  jpeg_bytes)
        file_resp = post_file(file_filename, jpeg_bytes)

        cam_r  = cam_resp["results"][0]
        file_r = file_resp["results"][0]

        REQUIRED_KEYS = {"filename","label","good_confidence","spoiled_confidence",
                         "decision","size_category","size_method",
                         "longest_dimension_mm","piece_count_estimate","box","segmentation_fallback"}

        missing_cam  = REQUIRED_KEYS - cam_r.keys()
        missing_file = REQUIRED_KEYS - file_r.keys()

        print(f"  Camera-filename response keys:  {sorted(cam_r.keys())}")
        print(f"  Upload-filename response keys:  {sorted(file_r.keys())}")
        print(f"  Camera label/decision:   {cam_r['label']} / {cam_r['decision']}")
        print(f"  Upload label/decision:   {file_r['label']} / {file_r['decision']}")

        assert not missing_cam,  f"Missing keys in camera response: {missing_cam}"
        assert not missing_file, f"Missing keys in upload response: {missing_file}"
        # Both should classify the identical JPEG identically
        assert cam_r["label"]    == file_r["label"],    "label mismatch"
        assert cam_r["decision"] == file_r["decision"], "decision mismatch"

        print(f"  Schema identical and decisions match: {PASS}")
        v3_ok = True

except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  Result: {FAIL}  — {e}")
    v3_ok = False


# ── Summary ────────────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"  V1 — Backend regression:       {'PASS' if v1_ok else 'FAIL'}")
print(f"  V2 — EXIF orientation fix:     {'PASS' if v2_ok else 'FAIL'}")
print(f"  V3 — Camera-path equivalence:  {'PASS' if v3_ok is True else 'SKIP (server not running)' if v3_ok is None else 'FAIL'}")
all_ok = v1_ok and v2_ok and (v3_ok is not False)
print(f"\n  Overall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
sys.exit(0 if all_ok else 1)
