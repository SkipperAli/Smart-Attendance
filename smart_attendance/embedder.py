from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

ImageInput = Union[str, np.ndarray]


@dataclass
class Face:
    embedding: np.ndarray
    box: tuple[int, int, int, int]  # x, y, w, h


class FaceEmbedder:
    """Thin wrapper around DeepFace: image in, list of (embedding, box) out."""

    def __init__(self, model_name: str, detector_backend: str):
        # Imported lazily: TensorFlow takes a few seconds to load.
        from deepface import DeepFace

        self._deepface = DeepFace
        self.model_name = model_name
        self.detector_backend = detector_backend

    def warm_up(self) -> None:
        """Load model weights now (downloads them on first run)."""
        self._deepface.build_model(self.model_name)

    def embed(self, image: ImageInput) -> list[Face]:
        """Return one Face per detected face. Empty list if no face is found."""
        try:
            reps = self._deepface.represent(
                img_path=image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True,
            )
        except ValueError:
            # DeepFace raises ValueError when no face is detected.
            return []

        faces = []
        for rep in reps:
            area = rep["facial_area"]
            box = (int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"]))
            faces.append(Face(np.asarray(rep["embedding"], dtype=np.float32), box))
        return faces
