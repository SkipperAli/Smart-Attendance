"""Unit tests for matching, caching and the attendance log.

These use fake embeddings, so they run in under a second without TensorFlow.
"""
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from smart_attendance.attendance_log import AttendanceLog
from smart_attendance.database import FaceDatabase, cosine_distances, discover_students


def touch(path: Path, content: bytes = b"img") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_discover_supports_files_and_folders(tmp_path):
    touch(tmp_path / "Sara.jpg")
    touch(tmp_path / "Ali Abbas" / "01.jpg")
    touch(tmp_path / "Ali Abbas" / "02.png")
    touch(tmp_path / "Ali Abbas" / "notes.txt")
    touch(tmp_path / ".cache" / "x.jpg")
    touch(tmp_path / "README.md")

    roster = discover_students(tmp_path)
    assert set(roster) == {"Sara", "Ali Abbas"}
    assert len(roster["Ali Abbas"]) == 2


def test_cosine_distance():
    m = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    d = cosine_distances(m, np.array([2, 0], dtype=np.float32))
    assert np.allclose(d, [0, 1, 2])


def test_match_picks_closest_and_respects_threshold():
    db = FaceDatabase(["ali", "ali", "sara"], np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype=np.float32))
    name, dist = db.match(np.array([0.95, 0.05, 0], dtype=np.float32), threshold=0.3)
    assert name == "ali" and dist < 0.01

    name, dist = db.match(np.array([0, 0, 1], dtype=np.float32), threshold=0.3)
    assert name is None and dist == pytest.approx(1.0)


def test_empty_database_matches_nobody():
    db = FaceDatabase([], np.zeros((0, 1), dtype=np.float32))
    assert db.match(np.ones(3), 0.3) == (None, float("inf"))


def test_build_uses_cache_and_skips_faceless_photos(tmp_path):
    students = tmp_path / "students"
    touch(students / "ali" / "01.jpg", b"a")
    touch(students / "sara.jpg", b"s")
    touch(students / "nobody.jpg", b"n")
    cache = tmp_path / "cache.pkl"

    calls = []

    def embed(path):
        calls.append(path.name)
        return None if path.stem == "nobody" else np.random.rand(4)

    db = FaceDatabase.build(students, embed, cache)
    assert db.students == ["ali", "sara"]
    assert len(calls) == 3

    calls.clear()
    FaceDatabase.build(students, embed, cache)
    assert calls == []  # everything served from cache, including the faceless photo

    touch(students / "sara.jpg", b"changed")  # edited photo gets re-embedded
    FaceDatabase.build(students, embed, cache)
    assert calls == ["sara.jpg"]


def test_attendance_marks_once_per_day(tmp_path):
    day = date(2026, 10, 1)
    log = AttendanceLog(tmp_path, day)
    assert log.mark("Ali", 0.12, datetime(2026, 10, 1, 9, 0, 5))
    assert not log.mark("Ali", 0.10)
    assert log.mark("Sara", 0.2, datetime(2026, 10, 1, 8, 55, 0))

    assert [r.name for r in log.records] == ["Sara", "Ali"]  # sorted by time

    reopened = AttendanceLog(tmp_path, day)  # survives a restart
    assert reopened.is_marked("Ali")
    assert not reopened.mark("Ali", 0.1)
    assert (tmp_path / "2026-10-01.csv").read_text().splitlines()[0] == "name,time,distance"
