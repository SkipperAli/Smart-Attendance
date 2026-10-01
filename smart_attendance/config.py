from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Cosine-distance thresholds DeepFace uses for each model.
# Lower distance = more similar. A face matches if distance <= threshold.
DEFAULT_THRESHOLDS = {
    "VGG-Face": 0.68,
    "Facenet": 0.40,
    "Facenet512": 0.30,
    "ArcFace": 0.68,
    "SFace": 0.593,
    "GhostFaceNet": 0.65,
}


@dataclass
class Settings:
    students_dir: Path = ROOT / "students"
    attendance_dir: Path = ROOT / "attendance"
    cache_dir: Path = ROOT / ".cache"
    model_name: str = "Facenet512"
    detector_backend: str = "opencv"
    threshold: float | None = None   # None -> model default from DEFAULT_THRESHOLDS
    camera_index: int = 0
    scan_interval: float = 0.8       # seconds between scans in auto mode
    min_face_size: int = 60          # ignore detections smaller than this (px)

    @property
    def effective_threshold(self) -> float:
        if self.threshold is not None:
            return self.threshold
        return DEFAULT_THRESHOLDS.get(self.model_name, 0.40)
