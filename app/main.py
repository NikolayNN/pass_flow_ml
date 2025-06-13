"""Minimal People Counter using YOLOv8 + DeepSORT
================================================
Draws bounding boxes, a base point at the bottom‑center of each box, traces the point, and counts
entries/exits across a user‑defined reference line — no smoothing or hysteresis.

Now supports:
* `--output` – choose the folder for processed video & CSV;
* `--trail-len` – length of the visual track (history) per object;
* `--point-radius` – radius of the base‑point circle.
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# ── Tracker initialization ────────────────────────────────────────────
tracker = DeepSort(max_age=30, n_init=1, max_cosine_distance=0.4)

# ── Global counters ───────────────────────────────────────────────────
counter_in = 0
counter_out = 0
counted_ids_in: set[int] = set()
counted_ids_out: set[int] = set()

# ── Geometry helpers ──────────────────────────────────────────────────

def intersect_line(p1: tuple[int, int], p2: tuple[int, int], line: list[int]) -> bool:
    x1, y1, x2, y2 = line

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (
        ccw(p1, (x1, y1), (x2, y2)) != ccw(p2, (x1, y1), (x2, y2))
        and ccw(p1, p2, (x1, y1)) != ccw(p1, p2, (x2, y2))
    )


def is_vertical(line: list[int]) -> bool:
    x1, y1, x2, y2 = line
    return abs(x1 - x2) > abs(y1 - y2)

# ── Core routine ──────────────────────────────────────────────────────

def run(
    source: str,
    weights: str,
    out_dir: str,
    csv_out: str,
    line_norm: list[float],
    trail_len: int = 2,
    point_radius: int = 4,
) -> None:
    """Process stream and save processed.mp4 + CSV into *out_dir*.

    * trail_len   – number of previous points kept and drawn for each track;
    * point_radius – visual size of the base‑point dot.
    """
    global counter_in, counter_out

    os.makedirs(out_dir, exist_ok=True)

    video_path = os.path.join(out_dir, "processed.mp4")
    csv_path = csv_out if os.path.isabs(csv_out) else os.path.join(out_dir, csv_out)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    model = YOLO(weights)

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Reference line in pixels
    line = [
        int(line_norm[0] * W),
        int(line_norm[1] * H),
        int(line_norm[2] * W),
        int(line_norm[3] * H),
    ]
    vertical = is_vertical(line)

    frame_idx = 0
    rows: list[list[int]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── Detection ────────────────────────────────────────────────
        results = model(frame, conf=0.4)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        detections = [
            ([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], float(conf), "person")
            for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes)
            if int(cls) == 0  # keep only persons
        ]

        # ── Tracking ────────────────────────────────────────────────
        tracks = tracker.update_tracks(detections, frame=frame)

        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 2)

        for tr in tracks:
            if not tr.is_confirmed():
                continue

            tid = tr.track_id
            l, t, r_, b = map(int, tr.to_ltrb())

            # Bounding box
            cv2.rectangle(frame, (l, t), (r_, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (l, t - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Base point (bottom‑center)
            cx = (l + r_) // 2
            cy = b
            cv2.circle(frame, (cx, cy), point_radius, (0, 0, 255), -1)

            # History of trail_len points
            tr.hist = getattr(tr, "hist", []) + [(cx, cy)]
            if len(tr.hist) > trail_len:
                tr.hist = tr.hist[-trail_len:]

            # Draw trail segments
            for p1, p2 in zip(tr.hist, tr.hist[1:]):
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # Counting on crossing
            if len(tr.hist) >= 2 and intersect_line(tr.hist[-2], tr.hist[-1], line):
                delta = (cx - tr.hist[-2][0]) if vertical else (cy - tr.hist[-2][1])
                if delta < 0 and tid not in counted_ids_in:
                    counter_in += 1
                    counted_ids_in.add(tid)
                elif delta > 0 and tid not in counted_ids_out:
                    counter_out += 1
                    counted_ids_out.add(tid)

        # Overlay counts
        cv2.putText(frame, f"IN:  {counter_in}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 160, 0), 2)
        cv2.putText(frame, f"OUT: {counter_out}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 160), 2)

        out.write(frame)
        rows.append([frame_idx, counter_in, counter_out, counter_in - counter_out])
        frame_idx += 1

    cap.release()
    out.release()

    pd.DataFrame(rows, columns=["frame", "in", "out", "total"]).to_csv(csv_path, index=False)

    print("Saved video →", video_path)
    print("Saved CSV   →", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal People Counter with YOLOv8 + DeepSORT")
    parser.add_argument("--source", required=True, help="video file or camera index; '0' for webcam")
    parser.add_argument("--weights", required=True, help="YOLOv8 .pt model file")
    parser.add_argument("--output", default="output", help="folder for processed.mp4 & default CSV")
    parser.add_argument("--save-csv", help="CSV filename or full path; default <output>/stats.csv")
    parser.add_argument("--trail-len", type=int, default=15, help="number of points in the visual track")
    parser.add_argument("--point-radius", type=int, default=4, help="radius of the base‑point circle")
    parser.add_argument(
        "--line", nargs=4, type=float, required=True, metavar=("x1", "y1", "x2", "y2"),
        help="normalized coordinates (0–1) of the reference line",
    )
    args = parser.parse_args()

    csv_name = args.save_csv or "stats.csv"

    run(
        source=args.source,
        weights=args.weights,
        out_dir=args.output,
        csv_out=csv_name,
        line_norm=args.line,
        trail_len=args.trail_len,
        point_radius=args.point_radius,
    )
