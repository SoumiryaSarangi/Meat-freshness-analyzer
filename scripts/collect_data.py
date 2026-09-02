"""
Data collection utility.

Public datasets (see README) are a fine starting point, but a model trained
only on them will likely underperform on your own camera and lighting. Use
this script to capture images directly from a webcam or camera feed to
fine-tune on.

Usage:
    python scripts/collect_data.py --source 0 --out data/raw --interval 0.5

Controls while running:
    SPACE - capture a frame immediately (in addition to the timed interval)
    q     - quit

After collecting, sort your images into:
  data/dataset_freshness/good/     -- fresh meat
  data/dataset_freshness/spoiled/  -- spoiled meat
Then use merge_datasets.py to add them to your train/val split.
"""

import argparse
import os
import time

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="camera index or video/stream URL")
    parser.add_argument("--out", default="data/raw", help="output directory for captured frames")
    parser.add_argument("--interval", type=float, default=1.0,
                         help="seconds between automatic captures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    print("Collecting frames. SPACE = capture now, q = quit.")
    last_capture = 0.0
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imshow("Data collection (SPACE=capture, q=quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        now = time.time()
        should_capture = (now - last_capture >= args.interval) or key == ord(" ")

        if should_capture:
            filename = os.path.join(args.out, f"frame_{count:05d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            last_capture = now
            print(f"Saved {filename}")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Captured {count} frames to {args.out}")


if __name__ == "__main__":
    main()
