from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import cv2

from . import ui
from .database import IMAGE_EXTS
from .live import open_camera

WINDOW = "Smart Attendance - Enroll"


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w .-]", "", name).strip(" .")
    if not cleaned:
        raise ValueError(f"'{name}' is not a usable name.")
    return cleaned


def next_index(folder: Path) -> int:
    nums = [int(p.stem) for p in folder.glob("*") if p.stem.isdigit()]
    return max(nums, default=0) + 1


def import_photos(students_dir: Path, name: str, photos: list[Path]) -> list[Path]:
    folder = students_dir / safe_name(name)
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    i = next_index(folder)
    for photo in photos:
        if not photo.is_file() or photo.suffix.lower() not in IMAGE_EXTS:
            print(f"  skipped {photo} (not an image)")
            continue
        dest = folder / f"{i:02d}{photo.suffix.lower()}"
        shutil.copy2(photo, dest)
        saved.append(dest)
        i += 1
    return saved


def capture_photos(students_dir: Path, name: str, shots: int, camera_index: int) -> list[Path]:
    """Open the webcam and save `shots` photos when exactly one face is visible."""
    folder = students_dir / safe_name(name)
    folder.mkdir(parents=True, exist_ok=True)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = open_camera(camera_index)
    saved: list[Path] = []
    flash_until = 0.0
    i = next_index(folder)
    try:
        while len(saved) < shots:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 6, minSize=(80, 80))

            view = frame.copy()
            for (x, y, w, h) in faces:
                ui.corner_box(view, (x, y, w, h), ui.GREEN if len(faces) == 1 else ui.AMBER)

            ui.draw_header(view, f"Enrolling: {name}", f"photo {len(saved) + 1} of {shots}", "manual", False)
            if len(faces) == 0:
                hint = "No face detected - face the camera"
            elif len(faces) > 1:
                hint = "Only one person in frame, please"
            else:
                hint = "SPACE capture    Q cancel    (tilt your head a little between shots)"
            ui.draw_footer(view, hint)
            if time.monotonic() < flash_until:
                ui.draw_toast(view, f"Saved {len(saved)}/{shots}")

            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and len(faces) == 1:
                dest = folder / f"{i:02d}.jpg"
                cv2.imwrite(str(dest), frame)
                saved.append(dest)
                i += 1
                flash_until = time.monotonic() + 0.8
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return saved
