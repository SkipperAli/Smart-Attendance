from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_students(students_dir: Path) -> dict[str, list[Path]]:
    """Find enrolled students.

    Two layouts are supported, and can be mixed:
      students/Ali Abbas.jpg            -> one photo
      students/Ali Abbas/01.jpg, 02.jpg -> several photos (more accurate)
    """
    roster: dict[str, list[Path]] = {}
    if not students_dir.exists():
        return roster

    for entry in sorted(students_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            images = sorted(p for p in entry.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if images:
                roster.setdefault(entry.name, []).extend(images)
        elif entry.suffix.lower() in IMAGE_EXTS:
            roster.setdefault(entry.stem, []).append(entry)
    return roster


def cosine_distances(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine distance between each row of `matrix` and `vector`."""
    m = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    v = vector / np.linalg.norm(vector)
    return 1.0 - m @ v


@dataclass
class Match:
    name: str | None      # None = unknown face
    distance: float       # distance to the closest enrolled face
    box: tuple[int, int, int, int]


class FaceDatabase:
    """Embeddings of every enrolled photo, with nearest-neighbour matching."""

    def __init__(self, names: list[str], embeddings: np.ndarray):
        self.names = names
        self.embeddings = embeddings

    @property
    def students(self) -> list[str]:
        return sorted(set(self.names))

    def __len__(self) -> int:
        return len(self.names)

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str | None, float]:
        """Closest student and their distance; name is None if above threshold."""
        if not self.names:
            return None, float("inf")
        distances = cosine_distances(self.embeddings, embedding)
        best = int(np.argmin(distances))
        distance = float(distances[best])
        return (self.names[best] if distance <= threshold else None), distance

    @classmethod
    def build(
        cls,
        students_dir: Path,
        embed_photo: Callable[[Path], np.ndarray | None],
        cache_file: Path | None = None,
    ) -> "FaceDatabase":
        """Embed every enrolled photo, reusing cached embeddings for unchanged files.

        `embed_photo` returns the embedding of the main face in a photo,
        or None if no face was found.
        """
        cache: dict[tuple, np.ndarray | None] = {}
        if cache_file and cache_file.exists():
            try:
                cache = pickle.loads(cache_file.read_bytes())
            except Exception:  # corrupt or incompatible cache: just rebuild
                log.warning("Embedding cache unreadable, rebuilding it.")

        roster = discover_students(students_dir)
        names: list[str] = []
        vectors: list[np.ndarray] = []
        fresh_cache: dict[tuple, np.ndarray | None] = {}
        computed = 0

        for name, photos in roster.items():
            for photo in photos:
                stat = photo.stat()
                key = (str(photo.relative_to(students_dir)), stat.st_mtime_ns, stat.st_size)
                if key in cache:
                    vector = cache[key]
                else:
                    vector = embed_photo(photo)
                    computed += 1
                    if vector is None:
                        log.warning("No face found in %s, skipping it.", photo)
                fresh_cache[key] = vector
                if vector is not None:
                    names.append(name)
                    vectors.append(np.asarray(vector, dtype=np.float32))

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(pickle.dumps(fresh_cache))

        log.info("Face database: %d photos of %d students (%d newly embedded).",
                 len(names), len(set(names)), computed)
        matrix = np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)
        return cls(names, matrix)
