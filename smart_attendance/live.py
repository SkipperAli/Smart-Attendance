from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np

from . import ui
from .attendance_log import AttendanceLog
from .database import FaceDatabase
from .embedder import FaceEmbedder
from .recognition import RecognitionWorker

log = logging.getLogger(__name__)

WINDOW = "Smart Attendance"
RESULT_TTL = 1.5   # seconds a face box stays on screen after a scan
TOAST_TTL = 2.5


def open_camera(index: int) -> cv2.VideoCapture:
    # DirectShow opens much faster than the default backend on Windows
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}. Try --camera 1 if you have more than one.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def run_live(
    embedder: FaceEmbedder,
    db: FaceDatabase,
    attendance: AttendanceLog,
    threshold: float,
    camera_index: int,
    scan_interval: float,
    min_face_size: int,
    start_in_auto: bool = True,
) -> None:
    worker = RecognitionWorker(embedder, db, threshold, min_face_size)
    worker.start()
    cap = open_camera(camera_index)

    auto = start_in_auto
    last_submit = 0.0
    seen_results = 0
    toast: tuple[str, tuple, float] | None = None
    roster = db.students
    date_label = attendance.day.strftime("%a, %d %b %Y")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("Camera stopped returning frames.")
                break
            frame = cv2.flip(frame, 1)  # mirror view feels natural on a webcam
            now = time.monotonic()

            if auto and not worker.busy and now - last_submit >= scan_interval:
                worker.submit(frame)
                last_submit = now

            # Handle newly finished scans
            if worker.results_id != seen_results:
                seen_results = worker.results_id
                newly_marked = []
                for m in worker.results:
                    if m.name and attendance.mark(m.name, m.distance, datetime.now()):
                        newly_marked.append(m.name)
                        log.info("Marked %s present (distance %.3f)", m.name, m.distance)
                if newly_marked:
                    toast = (f"Marked present: {', '.join(newly_marked)}", ui.GREEN, now)
                elif not auto and not worker.results:
                    toast = ("No face found, try again", ui.AMBER, now)
                elif not auto and all(m.name is None for m in worker.results):
                    toast = ("Face not recognised", ui.RED, now)

            view = frame.copy()
            if now - worker.results_time <= RESULT_TTL * (1 if auto else 2):
                for m in worker.results:
                    ui.draw_match(view, m, already_marked=bool(m.name) and attendance.is_marked(m.name))

            ui.draw_header(view, "Smart Attendance", f"{len(roster)} enrolled  |  model {embedder.model_name}",
                           "auto" if auto else "manual", worker.busy)
            ui.draw_footer(view, "SPACE scan    A toggle auto    Q quit")
            if toast and now - toast[2] <= TOAST_TTL:
                ui.draw_toast(view, toast[0], toast[1])

            panel = ui.sidebar(view.shape[0], attendance.records, roster, date_label)
            cv2.imshow(WINDOW, np.hstack([view, panel]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            if key == ord("a"):
                auto = not auto
                toast = (f"Auto scan {'on' if auto else 'off'}", ui.AMBER, now)
            if key == ord(" ") and not worker.busy:
                worker.submit(frame)
                last_submit = now
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
