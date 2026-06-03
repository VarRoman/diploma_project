#!/usr/bin/env python3
"""
redetect_wh.py — Phase F: перезапуск YOLO з захопленням ПОВНОЇ геометрії
bbox (w_box, h_box) + conf, а не лише ширини.

МОТИВАЦІЯ. Поточний detect_infos/detection_data.json містить лише w_box.
Аналіз показав: шум w_box корелює з ГОРИЗОНТАЛЬНОЮ швидкістю м'яча в кадрі
(corr|res|,|du|=+0.22 проти |dv|=+0.04) і має напрямкову складову (рух
праворуч роздуває w на ~+1 px). Це сигнатура motion-blur: розмиття розтягує
рамку ВЗДОВЖ руху. Вісь, ПЕРПЕНДИКУЛЯРНА руху, лишається чистим діаметром.
Щоб нею скористатись для глибини, потрібен h_box — звідси цей re-run.

ВИХІД: JSON-список (1 запис/кадр), сумісний із detection_data.json
(ті самі ключі frame, ball_detected, x_pos, y_pos, w_box) ПЛЮС h_box, conf.
Так старий пайплайн працює без змін, а новий може читати h_box/conf.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out_json", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=1920)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.model))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"[err] не відкрити відео: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sys.stderr.write(f"[i] {args.video.name} frames={total} model={args.model.name}\n")

    out = []
    fidx = 0  # 1-based, як у detection_data.json
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1
        res = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf,
                            iou=args.iou, verbose=False, stream=False)
        rec = {"frame": fidx, "ball_detected": False,
               "x_pos": None, "y_pos": None,
               "w_box": None, "h_box": None, "conf": None}
        if res and len(res[0].boxes) > 0:
            b = res[0].boxes
            confs = b.conf.cpu().numpy()
            i = int(np.argmax(confs))
            x, y, w, h = b.xywh[i].cpu().numpy().tolist()
            rec.update(ball_detected=True, x_pos=x, y_pos=y,
                       w_box=w, h_box=h, conf=float(confs[i]))
        out.append(rec)
        if fidx % 100 == 0:
            sys.stderr.write(f"  ... {fidx}/{total}\n")

    cap.release()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    n_det = sum(1 for r in out if r["ball_detected"])
    sys.stderr.write(f"[ok] {fidx} кадрів, {n_det} детекцій -> {args.out_json}\n")


if __name__ == "__main__":
    main()
