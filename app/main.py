"""People Counter with YOLOv8 + DeepSORT
================================================
▶ Версия 7 — стабилизация трека для предотвращения двойного счёта

Добавлено:
- Сглаживание координат маркера с помощью скользящего среднего
- Гистерезис по направлению (min_shift), чтобы избежать повторных срабатываний при скачках

"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Инициализация трекера
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.4
)

# Глобальные счётчики
counter_in = 0
counter_out = 0
counted_ids_in, counted_ids_out = set(), set()

# Настройки визуализации
POINT_COLOR_BG = (0, 255, 255)
POINT_COLOR_FG = (0, 0, 0)
POINT_RADIUS = 15
CROSS_SIZE = 12
TRAIL_COLOR = (0, 255, 255)
DRAW_TRAIL = True
TRAIL_LEN = 30

# Стабилизация
SMOOTH_WINDOW = 5
MIN_SHIFT_PX = 10

# ── Вспомогательные функции ─────────────────────────────────────
def intersect_line(p1, p2, line):
    x1, y1, x2, y2 = line
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (
        ccw(p1, (x1, y1), (x2, y2)) != ccw(p2, (x1, y1), (x2, y2))
        and ccw(p1, p2, (x1, y1)) != ccw(p1, p2, (x2, y2))
    )

def is_vertical(line):
    x1, y1, x2, y2 = line
    return abs(x1 - x2) > abs(y1 - y2)

# ── Основная функция ───────────────────────────────────────────
def run(src, weights, csv_out, line_norm, h_align, v_align, debug=False):
    global counter_in, counter_out

    model = YOLO(weights)

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))

    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter("output/processed.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    line = [
        int(line_norm[0] * W),
        int(line_norm[1] * H),
        int(line_norm[2] * W),
        int(line_norm[3] * H),
    ]
    vertical = is_vertical(line)

    frame_idx = 0
    rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=0.4)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        det_list = []
        for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes):
            if int(cls) != 0:
                continue
            det_list.append((
                [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                float(conf),
                "person"
            ))

        tracks = tracker.update_tracks(det_list, frame=frame)

        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), TRAIL_COLOR, 2)

        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            l, t, r, b = map(int, tr.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cx = l if h_align == "left" else r if h_align == "right" else (l + r) // 2
            cy = t if v_align == "top" else b if v_align == "bottom" else (t + b) // 2
            cy = np.clip(cy, 0, H - 1)

            if debug and not hasattr(tr, "_dbg"):
                print(f"Track {tid} marker=({cx},{cy}) align=[{h_align},{v_align}]")
                tr._dbg = True

            # Сглаживание траектории
            tr.smooth_hist = getattr(tr, "smooth_hist", []) + [(cx, cy)]
            if len(tr.smooth_hist) > SMOOTH_WINDOW:
                tr.smooth_hist.pop(0)
            smoothed = np.mean(tr.smooth_hist, axis=0).astype(int)
            cx, cy = smoothed

            # Рисуем маркер
            cv2.circle(frame, (cx, cy), POINT_RADIUS, POINT_COLOR_BG, -1)
            cv2.circle(frame, (cx, cy), POINT_RADIUS, POINT_COLOR_FG, 2)
            cv2.line(frame, (cx - CROSS_SIZE, cy), (cx + CROSS_SIZE, cy), POINT_COLOR_FG, 2)
            cv2.line(frame, (cx, cy - CROSS_SIZE), (cx, cy + CROSS_SIZE), POINT_COLOR_FG, 2)

            tr.hist = getattr(tr, "hist", []) + [(cx, cy)]
            if len(tr.hist) > TRAIL_LEN:
                tr.hist.pop(0)
            if DRAW_TRAIL and len(tr.hist) > 1:
                for i in range(1, len(tr.hist)):
                    cv2.line(frame, tr.hist[i - 1], tr.hist[i], TRAIL_COLOR, 1)

            if len(tr.hist) > 1 and intersect_line(tr.hist[-2], tr.hist[-1], line):
                d = (cx - tr.hist[-2][0]) if vertical else (cy - tr.hist[-2][1])
                if abs(d) < MIN_SHIFT_PX:
                    continue
                if d < 0 and tid not in counted_ids_in:
                    counter_in += 1
                    counted_ids_in.add(tid)
                elif d > 0 and tid not in counted_ids_out:
                    counter_out += 1
                    counted_ids_out.add(tid)

        cv2.putText(frame, f"IN: {counter_in}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        cv2.putText(frame, f"OUT: {counter_out}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)

        out.write(frame)
        rows.append([frame_idx, counter_in, counter_out, counter_in - counter_out])
        frame_idx += 1

    cap.release()
    out.release()
    pd.DataFrame(rows, columns=["frame", "in", "out", "total"]).to_csv(csv_out, index=False)
    print(f"Saved CSV: {csv_out}\nSaved video: output/processed.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="People Counter with stable tracking (YOLOv8)")
    parser.add_argument("--source", required=True, help="путь к видео или камере")
    parser.add_argument("--weights", required=True, help=".pt‑файл модели YOLOv8")
    parser.add_argument("--save-csv", required=True, help="файл для статистики по кадрам")
    parser.add_argument("--line", nargs=4, type=float, required=True, metavar=("x1", "y1", "x2", "y2"), help="нормированные координаты линии (0–1)")
    parser.add_argument("--h-align", choices=["left", "center", "right"], default="center", help="горизонтальное выравнивание маркера")
    parser.add_argument("--v-align", choices=["top", "center", "bottom"], default="center", help="вертикальное выравнивание маркера")
    parser.add_argument("--debug", action="store_true", help="печать координат первого маркера")
    args = parser.parse_args()

    run(
        args.source,
        args.weights,
        args.save_csv,
        args.line,
        args.h_align,
        args.v_align,
        debug=args.debug,
    )
