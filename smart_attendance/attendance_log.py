from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

FIELDS = ["name", "time", "distance"]


@dataclass
class Record:
    name: str
    time: str
    distance: float


class AttendanceLog:
    """One CSV per day (attendance/YYYY-MM-DD.csv). Each student is marked once per day."""

    def __init__(self, directory: Path, day: date | None = None):
        self.directory = directory
        self.day = day or date.today()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._records = {r.name: r for r in self.read(self.day)}

    @property
    def path(self) -> Path:
        return self.file_for(self.day)

    def file_for(self, day: date) -> Path:
        return self.directory / f"{day.isoformat()}.csv"

    @property
    def records(self) -> list[Record]:
        return sorted(self._records.values(), key=lambda r: r.time)

    def is_marked(self, name: str) -> bool:
        return name in self._records

    def mark(self, name: str, distance: float, when: datetime | None = None) -> bool:
        """Record a student as present. Returns False if already marked today."""
        if name in self._records:
            return False
        record = Record(name, (when or datetime.now()).strftime("%H:%M:%S"), round(distance, 4))
        new_file = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({"name": record.name, "time": record.time, "distance": record.distance})
        self._records[name] = record
        return True

    def read(self, day: date) -> list[Record]:
        path = self.file_for(day)
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as f:
            return [Record(row["name"], row["time"], float(row["distance"])) for row in csv.DictReader(f)]
