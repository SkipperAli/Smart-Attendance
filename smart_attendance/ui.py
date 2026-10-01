from __future__ import annotations

import cv2
import numpy as np

from .attendance_log import Record
from .database import Match

# Colours are BGR
GREEN = (94, 197, 34)
TEAL = (170, 178, 20)
RED = (68, 68, 239)
AMBER = (11, 158, 245)
WHITE = (245, 245, 245)
MUTED = (160, 160, 160)
PANEL = (32, 27, 24)
PANEL_LINE = (58, 52, 48)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
SIDEBAR_WIDTH = 280


def text(img, s, org, scale=0.5, color=WHITE, thickness=1, font=FONT):
    cv2.putText(img, s, org, font, scale, color, thickness, cv2.LINE_AA)


def text_size(s, scale=0.5, thickness=1, font=FONT) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(s, font, scale, thickness)
    return w, h


def translucent_rect(img, pt1, pt2, color, alpha=0.6):
    x1, y1 = max(pt1[0], 0), max(pt1[1], 0)
    x2, y2 = min(pt2[0], img.shape[1]), min(pt2[1], img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def corner_box(img, box, color, thickness=2):
    """Bracket-style corners instead of a full rectangle."""
    x, y, w, h = box
    length = max(12, min(w, h) // 4)
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)


def draw_match(img, match: Match, already_marked: bool) -> None:
    if match.name is None:
        color, label, sub = RED, "Unknown", f"dist {match.distance:.2f}"
    else:
        color = TEAL if already_marked else GREEN
        label = match.name
        sub = ("present" if already_marked else "match") + f"  {match.distance:.2f}"

    corner_box(img, match.box, color)
    x, y, w, h = match.box
    lw, lh = text_size(label, 0.6, 1, FONT_BOLD)
    sw, sh = text_size(sub, 0.42)
    tag_w = max(lw, sw) + 16
    tag_h = lh + sh + 18
    top = y - tag_h - 6 if y - tag_h - 6 > 0 else y + h + 6
    cv2.rectangle(img, (x, top), (x + tag_w, top + tag_h), color, -1, cv2.LINE_AA)
    text(img, label, (x + 8, top + lh + 6), 0.6, (20, 20, 20), 1, FONT_BOLD)
    text(img, sub, (x + 8, top + lh + sh + 12), 0.42, (40, 40, 40))


def draw_header(img, title: str, subtitle: str, mode: str, scanning: bool) -> None:
    translucent_rect(img, (0, 0), (img.shape[1], 46), (15, 15, 15), 0.55)
    text(img, title, (14, 30), 0.7, WHITE, 1, FONT_BOLD)
    tw, _ = text_size(title, 0.7, 1, FONT_BOLD)
    text(img, subtitle, (14 + tw + 14, 29), 0.45, MUTED)

    pill = f"{mode.upper()}" + ("  scanning" if scanning else "")
    pw, _ = text_size(pill, 0.45)
    x2 = img.shape[1] - 12
    color = GREEN if mode == "auto" else AMBER
    cv2.rectangle(img, (x2 - pw - 34, 12), (x2, 36), color, -1, cv2.LINE_AA)
    cv2.circle(img, (x2 - pw - 20, 24), 4, (20, 20, 20), -1, cv2.LINE_AA)
    text(img, pill, (x2 - pw - 10, 29), 0.45, (20, 20, 20))


def draw_footer(img, hints: str) -> None:
    h = img.shape[0]
    translucent_rect(img, (0, h - 34), (img.shape[1], h), (15, 15, 15), 0.55)
    text(img, hints, (14, h - 12), 0.45, WHITE)


def draw_toast(img, message: str, color=GREEN) -> None:
    w, h = text_size(message, 0.6, 1, FONT_BOLD)
    x = (img.shape[1] - w) // 2
    y = img.shape[0] - 70
    cv2.rectangle(img, (x - 16, y - h - 14), (x + w + 16, y + 12), color, -1, cv2.LINE_AA)
    text(img, message, (x, y), 0.6, (20, 20, 20), 1, FONT_BOLD)


def sidebar(height: int, records: list[Record], roster: list[str], date_label: str) -> np.ndarray:
    panel = np.full((height, SIDEBAR_WIDTH, 3), PANEL, dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, height), PANEL_LINE, 1)

    text(panel, "ATTENDANCE", (18, 32), 0.5, MUTED)
    text(panel, date_label, (18, 54), 0.5, WHITE)

    present, total = len(records), len(roster)
    text(panel, f"{present}", (18, 104), 1.4, GREEN, 2, FONT_BOLD)
    nw, _ = text_size(f"{present}", 1.4, 2, FONT_BOLD)
    text(panel, f"/ {total} present", (24 + nw, 104), 0.55, MUTED)

    # progress bar
    bar_w = SIDEBAR_WIDTH - 36
    cv2.rectangle(panel, (18, 118), (18 + bar_w, 124), PANEL_LINE, -1)
    if total:
        cv2.rectangle(panel, (18, 118), (18 + int(bar_w * min(present / total, 1)), 124), GREEN, -1)

    y = 156
    cv2.line(panel, (18, y - 14), (SIDEBAR_WIDTH - 18, y - 14), PANEL_LINE, 1)
    marked = {r.name for r in records}
    rows = [(r.name, r.time, True) for r in records] + [(n, "--:--", False) for n in roster if n not in marked]

    max_rows = (height - y - 40) // 26
    for name, when, here in rows[:max_rows]:
        cv2.circle(panel, (24, y + 6), 5, GREEN if here else PANEL_LINE, -1, cv2.LINE_AA)
        label = name if len(name) <= 20 else name[:19] + "..."
        text(panel, label, (38, y + 11), 0.5, WHITE if here else MUTED)
        tw, _ = text_size(when, 0.45)
        text(panel, when, (SIDEBAR_WIDTH - 18 - tw, y + 11), 0.45, MUTED)
        y += 26
    if len(rows) > max_rows:
        text(panel, f"+ {len(rows) - max_rows} more", (38, y + 11), 0.45, MUTED)
    return panel
