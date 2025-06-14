"""Minimal People Counter using YOLOv8 + DeepSORT
================================================
Draws bounding boxes, a base point at the bottom‑center of each box, traces the point, and counts
entries/exits across a user‑defined reference line — no smoothing or hysteresis.

Now supports:
* `--output` – choose the folder for processed video & CSV;
* `--trail-len` – length of the visual track (history) per object;
* `--point-radius` – radius of the base‑point circle;
* `--reverse` – swap in/out orientation.
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
    line2_norm: list[float] | None = None,
    trail_len: int = 2,
    point_radius: int = 4,
    min_frames: int = 10,
    min_disp: float = 20.0,
    reverse: bool = False,
) -> None:
    """Process stream and save processed.mp4 + CSV into *out_dir*.

    * trail_len   – number of previous points kept and drawn for each track;
    * point_radius – visual size of the base‑point dot.
    * min_frames   – minimum track length in frames for counting;
    * min_disp     – minimum displacement in pixels for counting.
    * line2_norm   – optional second line for strict A->B/B->A counting.
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
    # Precompute the normal vector of the line pointing to its left side
    vec = (line[2] - line[0], line[3] - line[1])
    line_normal = (-vec[1], vec[0])
    if reverse:
        line_normal = (-line_normal[0], -line_normal[1])

    line2 = None
    if line2_norm:
        line2 = [
            int(line2_norm[0] * W),
            int(line2_norm[1] * H),
            int(line2_norm[2] * W),
            int(line2_norm[3] * H),
        ]
    in_seq = ("A", "B")
    out_seq = ("B", "A")
    if reverse:
        in_seq, out_seq = out_seq, in_seq

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
        if line2:
            cv2.line(frame, (line2[0], line2[1]), (line2[2], line2[3]), (0, 255, 255), 2)

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

            # Track lifetime and start point for filtering
            if not hasattr(tr, "frames"):
                tr.frames = 0
                tr.start_pt = (cx, cy)
            tr.frames += 1

            # History of trail_len points
            tr.hist = getattr(tr, "hist", []) + [(cx, cy)]
            if len(tr.hist) > trail_len:
                tr.hist = tr.hist[-trail_len:]

            # Draw trail segments
            for p1, p2 in zip(tr.hist, tr.hist[1:]):
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # Draw the displacement vector from the first point
            cv2.arrowedLine(
                frame,
                tr.start_pt,
                (cx, cy),
                (0, 255, 255),
                2,
                tipLength=0.3,
            )

            # Counting on crossing
            if len(tr.hist) >= 2:
                p_prev, p_curr = tr.hist[-2], tr.hist[-1]
                if line2:
                    if intersect_line(p_prev, p_curr, line):
                        if getattr(tr, "last_cross", None) == "B":
                            disp = np.hypot(cx - tr.start_pt[0], cy - tr.start_pt[1])
                            if tr.frames >= min_frames and disp >= min_disp:
                                seq = ("B", "A")
                                if seq == in_seq and tid not in counted_ids_in:
                                    counter_in += 1
                                    counted_ids_in.add(tid)
                                elif seq == out_seq and tid not in counted_ids_out:
                                    counter_out += 1
                                    counted_ids_out.add(tid)
                            tr.last_cross = None
                        else:
                            tr.last_cross = "A"
                    elif intersect_line(p_prev, p_curr, line2):
                        if getattr(tr, "last_cross", None) == "A":
                            disp = np.hypot(cx - tr.start_pt[0], cy - tr.start_pt[1])
                            if tr.frames >= min_frames and disp >= min_disp:
                                seq = ("A", "B")
                                if seq == in_seq and tid not in counted_ids_in:
                                    counter_in += 1
                                    counted_ids_in.add(tid)
                                elif seq == out_seq and tid not in counted_ids_out:
                                    counter_out += 1
                                    counted_ids_out.add(tid)
                            tr.last_cross = None
                        else:
                            tr.last_cross = "B"
                else:
                    if intersect_line(p_prev, p_curr, line):
                        disp = np.hypot(cx - tr.start_pt[0], cy - tr.start_pt[1])
                        if tr.frames >= min_frames and disp >= min_disp:
                            v = (cx - tr.start_pt[0], cy - tr.start_pt[1])
                            dot = v[0] * line_normal[0] + v[1] * line_normal[1]
                            if dot > 0 and tid not in counted_ids_in:
                                counter_in += 1
                                counted_ids_in.add(tid)
                            elif dot < 0 and tid not in counted_ids_out:
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
    parser.add_argument("--min-frames", type=int, default=10, help="minimum track length in frames to count")
    parser.add_argument("--min-displacement", type=float, default=20.0, help="minimum displacement in pixels to count")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="swap the in/out orientation of the reference line",
    )
    parser.add_argument(
        "--line", nargs=4, type=float, required=True, metavar=("x1", "y1", "x2", "y2"),
        help="normalized coordinates (0–1) of the reference line",
    )
    parser.add_argument(
        "--line2",
        nargs=4,
        type=float,
        metavar=("x1", "y1", "x2", "y2"),
        help="optional second reference line for A->B/B->A counting",
    )
    args = parser.parse_args()

    csv_name = args.save_csv or "stats.csv"

    run(
        source=args.source,
        weights=args.weights,
        out_dir=args.output,
        csv_out=csv_name,
        line_norm=args.line,
        line2_norm=args.line2,
        trail_len=args.trail_len,
        point_radius=args.point_radius,
        min_frames=args.min_frames,
        min_disp=args.min_displacement,
        reverse=args.reverse,
    )
