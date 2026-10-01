from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from .database import FaceDatabase, Match
from .embedder import FaceEmbedder

log = logging.getLogger(__name__)

MAX_WIDTH = 640  # frames are downscaled to this width before detection, for speed


def recognize(
    frame: np.ndarray,
    embedder: FaceEmbedder,
    db: FaceDatabase,
    threshold: float,
    min_face_size: int = 0,
) -> list[Match]:
    """Detect every face in a BGR frame and match it against the database."""
    scale = 1.0
    small = frame
    if frame.shape[1] > MAX_WIDTH:
        scale = frame.shape[1] / MAX_WIDTH
        small = cv2.resize(frame, (MAX_WIDTH, int(frame.shape[0] / scale)))

    matches = []
    for face in embedder.embed(small):
        x, y, w, h = (int(v * scale) for v in face.box)
        if w < min_face_size or h < min_face_size:
            continue
        name, distance = db.match(face.embedding, threshold)
        matches.append(Match(name, distance, (x, y, w, h)))
    return matches


class RecognitionWorker(threading.Thread):
    """Runs recognition off the UI thread so the camera feed never freezes."""

    def __init__(self, embedder: FaceEmbedder, db: FaceDatabase, threshold: float, min_face_size: int):
        super().__init__(daemon=True)
        self.embedder = embedder
        self.db = db
        self.threshold = threshold
        self.min_face_size = min_face_size

        self._cond = threading.Condition()
        self._pending: np.ndarray | None = None
        self._stopped = False
        self._busy = False

        self.results: list[Match] = []
        self.results_id = 0          # increments every time new results land
        self.results_time = 0.0

    @property
    def busy(self) -> bool:
        return self._busy or self._pending is not None

    def submit(self, frame: np.ndarray) -> None:
        """Queue a frame. If one is already waiting, it is replaced (we only want the latest)."""
        with self._cond:
            self._pending = frame.copy()
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify()

    def run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                frame, self._pending = self._pending, None
                self._busy = True
            try:
                matches = recognize(frame, self.embedder, self.db, self.threshold, self.min_face_size)
            except Exception:
                log.exception("Recognition failed on a frame")
                matches = []
            with self._cond:
                self.results = matches
                self.results_time = time.monotonic()
                self.results_id += 1
                self._busy = False
