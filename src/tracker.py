"""
Lightweight centroid tracker.

On a moving conveyor, the same physical piece of meat is detected across many
consecutive frames. Without tracking, the pipeline would classify and route
it multiple times. This tracker assigns a stable ID to each piece based on
centroid proximity between frames, so the pipeline can process each piece
exactly once (on first confident sighting) and ignore repeat detections
until it disappears off-frame.

This is intentionally simple (no Kalman filter / re-ID) -- sufficient for a
single-lane conveyor with one piece width of separation. For overlapping or
fast-moving items, consider ByteTrack (bundled with Ultralytics: model.track()).
"""

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared_frames: int = 10, max_match_distance_px: float = 75):
        self.next_id = 0
        self.objects: "OrderedDict[int, Tuple[int, int]]" = OrderedDict()
        self.disappeared: "OrderedDict[int, int]" = OrderedDict()
        self.processed_ids: set = set()  # IDs already sent downstream for classification
        self.max_disappeared_frames = max_disappeared_frames
        self.max_match_distance_px = max_match_distance_px

    def _register(self, centroid: Tuple[int, int]) -> int:
        object_id = self.next_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.next_id += 1
        return object_id

    def _deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]
        self.processed_ids.discard(object_id)

    def update(self, centroids: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """Call once per frame with the centroids of this frame's detections.
        Returns a mapping of object_id -> centroid for currently visible objects,
        in the same order as the input list (so callers can zip back to detections)."""
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared_frames:
                    self._deregister(object_id)
            return {}

        if len(self.objects) == 0:
            assigned = {}
            for c in centroids:
                oid = self._register(c)
                assigned[oid] = c
            return assigned

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))
        input_centroids = np.array(centroids)

        distances = np.linalg.norm(
            object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :], axis=2
        )

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        assigned: Dict[int, Tuple[int, int]] = {}

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_match_distance_px:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = centroids[col]
            self.disappeared[object_id] = 0
            assigned[object_id] = centroids[col]
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared_frames:
                self._deregister(object_id)

        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            oid = self._register(centroids[col])
            assigned[oid] = centroids[col]

        return assigned

    def mark_processed(self, object_id: int) -> None:
        self.processed_ids.add(object_id)

    def is_processed(self, object_id: int) -> bool:
        return object_id in self.processed_ids
