"""Smart Attendance - face-recognition attendance from your webcam.

    python attendance.py run                    start live attendance
    python attendance.py enroll "Ali Abbas"     add a student with the webcam
    python attendance.py scan class.jpg         take attendance from a photo
    python attendance.py report                 who is present / absent today
    python attendance.py students               list enrolled students
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

# Quieten TensorFlow's start-up noise before it gets imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from smart_attendance.attendance_log import AttendanceLog
from smart_attendance.config import DEFAULT_THRESHOLDS, Settings
from smart_attendance.database import FaceDatabase, discover_students

log = logging.getLogger("smart_attendance")


# ---------------------------------------------------------------- helpers

def load_engine(settings: Settings, rebuild: bool = False):
    """Load the model and build (or update) the face database."""
    from smart_attendance.embedder import FaceEmbedder

    roster = discover_students(settings.students_dir)
    if not roster:
        sys.exit(
            f"No students enrolled yet.\n"
            f"  Add one with:  python attendance.py enroll \"Your Name\"\n"
            f"  or drop photos into {settings.students_dir}{os.sep}"
        )

    print(f"Loading {settings.model_name} (first run downloads the weights)...")
    embedder = FaceEmbedder(settings.model_name, settings.detector_backend)
    embedder.warm_up()

    def embed_photo(path: Path):
        faces = embedder.embed(str(path))
        if not faces:
            return None
        # Enrollment photos should contain one person; use the largest face if not.
        return max(faces, key=lambda f: f.box[2] * f.box[3]).embedding

    cache = settings.cache_dir / f"embeddings-{settings.model_name}-{settings.detector_backend}.pkl"
    if rebuild and cache.exists():
        cache.unlink()
    db = FaceDatabase.build(settings.students_dir, embed_photo, cache)
    if not len(db):
        sys.exit("None of the enrolled photos contain a detectable face. Use clear, front-facing photos.")
    print(f"Ready: {len(db.students)} students, {len(db)} photos.")
    return embedder, db


# ---------------------------------------------------------------- commands

def cmd_run(args, settings: Settings) -> None:
    from smart_attendance.live import run_live

    embedder, db = load_engine(settings, args.rebuild)
    attendance = AttendanceLog(settings.attendance_dir)
    run_live(
        embedder, db, attendance,
        threshold=settings.effective_threshold,
        camera_index=settings.camera_index,
        scan_interval=settings.scan_interval,
        min_face_size=settings.min_face_size,
        start_in_auto=not args.manual,
    )
    print(f"\n{len(attendance.records)} of {len(db.students)} present. Saved to {attendance.path}")


def cmd_enroll(args, settings: Settings) -> None:
    from smart_attendance.enroll import capture_photos, import_photos

    if args.photos:
        saved = import_photos(settings.students_dir, args.name, [Path(p) for p in args.photos])
    else:
        saved = capture_photos(settings.students_dir, args.name, args.shots, settings.camera_index)
    if saved:
        print(f"Saved {len(saved)} photo(s) for {args.name} in {saved[0].parent}")
    else:
        print("Nothing saved.")


def cmd_scan(args, settings: Settings) -> None:
    import cv2
    from smart_attendance import ui
    from smart_attendance.recognition import recognize

    embedder, db = load_engine(settings, args.rebuild)
    attendance = AttendanceLog(settings.attendance_dir)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in args.images:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"{image_path}: could not read image")
            continue
        matches = recognize(frame, embedder, db, settings.effective_threshold)
        print(f"\n{image_path}: {len(matches)} face(s)")
        for m in matches:
            ui.draw_match(frame, m, already_marked=False)
            if m.name is None:
                print(f"  ?  unknown              closest distance {m.distance:.3f}")
                continue
            if args.no_mark:
                status = "recognised"
            elif attendance.mark(m.name, m.distance):
                status = "marked present"
            else:
                status = "already marked today"
            print(f"  +  {m.name:<20} distance {m.distance:.3f}  {status}")
        if out_dir:
            dest = out_dir / f"{Path(image_path).stem}_scanned.jpg"
            cv2.imwrite(str(dest), frame)
            print(f"  annotated image -> {dest}")


def cmd_report(args, settings: Settings) -> None:
    day = date.fromisoformat(args.date) if args.date else date.today()
    attendance = AttendanceLog(settings.attendance_dir, day)
    roster = sorted(discover_students(settings.students_dir))
    records = attendance.records
    present = {r.name for r in records}

    print(f"\nAttendance for {day:%A, %d %B %Y}")
    print("-" * 40)
    if not records and not attendance.path.exists():
        print("No attendance taken on this day.")
    for r in records:
        print(f"  present  {r.name:<22} {r.time}")
    for name in roster:
        if name not in present:
            print(f"  absent   {name}")
    print("-" * 40)
    print(f"{len(present)} present, {len([n for n in roster if n not in present])} absent")


def cmd_students(args, settings: Settings) -> None:
    roster = discover_students(settings.students_dir)
    if not roster:
        print("No students enrolled.")
        return
    for name, photos in roster.items():
        print(f"  {name:<24} {len(photos)} photo(s)")
    print(f"\n{len(roster)} students")


# ---------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="attendance.py",
        description="Face-recognition attendance with DeepFace + OpenCV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="Facenet512", choices=sorted(DEFAULT_THRESHOLDS),
                   help="face recognition model (default: Facenet512)")
    p.add_argument("--detector", default="opencv",
                   help="face detector: opencv (fast), ssd, mtcnn, retinaface, yunet... (default: opencv)")
    p.add_argument("--threshold", type=float,
                   help="max cosine distance for a match (default: the model's own threshold)")
    p.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    p.add_argument("--students", type=Path, help="students folder (default: ./students)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start live attendance from the webcam")
    run.add_argument("--manual", action="store_true", help="start in manual mode (press SPACE to scan)")
    run.add_argument("--interval", type=float, default=0.8, help="seconds between auto scans")
    run.add_argument("--rebuild", action="store_true", help="re-embed every enrolled photo")
    run.set_defaults(func=cmd_run)

    enroll = sub.add_parser("enroll", help="add a student (webcam capture, or import photos)")
    enroll.add_argument("name", help='student name, e.g. "Ali Abbas"')
    enroll.add_argument("photos", nargs="*", help="photo files to import instead of using the webcam")
    enroll.add_argument("--shots", type=int, default=5, help="photos to capture (default: 5)")
    enroll.set_defaults(func=cmd_enroll)

    scan = sub.add_parser("scan", help="take attendance from one or more photos")
    scan.add_argument("images", nargs="+")
    scan.add_argument("--out", help="folder to save annotated images")
    scan.add_argument("--no-mark", action="store_true", help="only recognise, don't record attendance")
    scan.add_argument("--rebuild", action="store_true", help="re-embed every enrolled photo")
    scan.set_defaults(func=cmd_scan)

    report = sub.add_parser("report", help="present / absent list for a day")
    report.add_argument("--date", help="YYYY-MM-DD (default: today)")
    report.set_defaults(func=cmd_report)

    students = sub.add_parser("students", help="list enrolled students")
    students.set_defaults(func=cmd_students)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
    )
    settings = Settings(
        model_name=args.model,
        detector_backend=args.detector,
        threshold=args.threshold,
        camera_index=args.camera,
    )
    if args.students:
        settings.students_dir = args.students
    if getattr(args, "interval", None):
        settings.scan_interval = args.interval
    try:
        args.func(args, settings)
    except KeyboardInterrupt:
        print("\nStopped.")
    except RuntimeError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
