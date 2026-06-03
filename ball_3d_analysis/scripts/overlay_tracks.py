#!/usr/bin/env python3
"""
Накласти результат IMMTracker (JSONL з run_segment.py) на відео.

Малює на кожному кадрі:
  • сиру YOLO-детекцію (жовте коло + bbox; червоне, якщо детекцію відкинуто
    гейтом — з підписом причини);
  • 3D-позиції треків, спроєктовані НАЗАД у кадр (Confirmed — зелене,
    Tentative — сіре), з track_id, метровими X/Y/Z і слідом останніх позицій;
  • стовпчики ймовірностей режимів IMM (ballistic / hit / bounce) для
    домінантного треку — щоб ОКОМ бачити, чи прокидаються фізичні режими;
  • каркас майданчика й сітки, спроєктований тими ж K,R,t — візуальна
    перевірка, що калібрування й глибина адекватні.

Калібрування береться з header (--save_calibration) АБО перераховується
self-contained тими ж 4 копланарними кутами, що й у run_segment.

Запуск:
  ~/miniconda3/envs/Volleyball_diploma/bin/python3 scripts/overlay_tracks.py \
      --video ../data/videos/Japan_vs_Poland_ultrashort.mp4 \
      --jsonl /tmp/ab_default_check.jsonl \
      --out /tmp/overlay.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2

# self-contained калібрування (ті самі точки, що у run_segment / модулі)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from get_court_position_methods import (  # noqa: E402
    calibrate_camera, pts_real_3d, pts_video_2d,
)

COURT_X, COURT_Z, NET_H = 9.0, 18.0, 2.43


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    return lines[0], lines[1:]


def build_calibration(header):
    """Повертає (K, R, camera_pos) — з header.calibration якщо є, інакше
    перерахунок тими ж 4 копланарними кутами."""
    cal = header.get("calibration")
    if cal and "K" in cal and "R" in cal and "camera_pos" in cal:
        K = np.array(cal["K"], np.float64)
        R = np.array(cal["R"], np.float64)
        cam = np.array(cal["camera_pos"], np.float64).reshape(3)
        return K, R, cam
    # перерахунок: фокусна з tracker.focal_override або K-дефолт 5805 (бродкаст)
    tk = header.get("tracker", {})
    f = tk.get("focal_override") or 5805.0
    iw, ih = header.get("frame_size", [1920, 1080])
    K = np.array([[f, 0, iw / 2.0], [0, f, ih / 2.0], [0, 0, 1.0]], np.float32)
    dist = np.zeros((4, 1), np.float32)
    R, _tvec, cam = calibrate_camera(pts_real_3d, pts_video_2d, K, dist)
    return np.array(K, np.float64), np.array(R, np.float64), \
        np.array(cam, np.float64).reshape(3)


def project(pts_world, K, R, cam):
    """(N,3) світові → (N,2) пікселі, узгоджено з оберненою моделлю
    get_3d_position: P = cam + Z_c·(Rᵀ·d_cam) ⇒ q = R·(P−cam),
    u = cx + fx·q₀/q₂. (calibrate_camera робить Y-flip для хіральності,
    тож cv2.projectPoints з tvec тут НЕ підходить.) Повертає (px(N,2),
    z_cam(N,)) — z_cam>0 = точка перед камерою."""
    pw = np.asarray(pts_world, np.float64).reshape(-1, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    q = (R @ (pw - cam.reshape(1, 3)).T).T          # (N,3)
    zc = q[:, 2]
    zc_safe = np.where(np.abs(zc) < 1e-9, 1e-9, zc)
    u = cx + fx * q[:, 0] / zc_safe
    v = cy + fy * q[:, 1] / zc_safe
    return np.stack([u, v], axis=1), zc


def draw_court(img, K, R, cam):
    corners = np.array([[0, 0, 0], [COURT_X, 0, 0],
                        [COURT_X, 0, COURT_Z], [0, 0, COURT_Z]], float)
    net = np.array([[0, 0, COURT_Z / 2], [0, NET_H, COURT_Z / 2],
                    [COURT_X, NET_H, COURT_Z / 2], [COURT_X, 0, COURT_Z / 2]],
                   float)
    cpx, cz = project(corners, K, R, cam)
    npx, nz = project(net, K, R, cam)
    if np.all(cz > 0):
        pts = cpx.astype(int)
        for i in range(4):
            cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]),
                     (90, 90, 90), 2, cv2.LINE_AA)
    if np.all(nz > 0):
        p = npx.astype(int)
        cv2.line(img, tuple(p[1]), tuple(p[2]), (200, 200, 0), 2, cv2.LINE_AA)
        cv2.line(img, tuple(p[0]), tuple(p[1]), (200, 200, 0), 1, cv2.LINE_AA)
        cv2.line(img, tuple(p[3]), tuple(p[2]), (200, 200, 0), 1, cv2.LINE_AA)


def draw_mode_bars(img, mu, org):
    """Стовпчики IMM: ballistic(сірий) / hit(червоний) / bounce(синій)."""
    labels = ["bal", "hit", "bnc"]
    cols = [(180, 180, 180), (0, 0, 255), (255, 120, 0)]
    x0, y0 = org
    for i, (lab, c) in enumerate(zip(labels, cols)):
        h = int(60 * float(mu[i])) if i < len(mu) else 0
        y = y0 + i * 22
        cv2.rectangle(img, (x0, y + 14), (x0 + 60, y + 18), (60, 60, 60), -1)
        cv2.rectangle(img, (x0, y + 14), (x0 + max(1, h), y + 18), c, -1)
        cv2.putText(img, f"{lab} {float(mu[i]) if i < len(mu) else 0:.2f}",
                    (x0 + 66, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1,
                    cv2.LINE_AA)


def draw_coord_panel(img, tid, Xw, zcam, dz, fresh):
    """Помітне табло координат домінантного треку: світові X/Y/Z (м),
    дистанція до камери Z_cam (м) і її покадрова зміна ΔZ_cam (м) —
    індикатор джиттера глибини. ΔZ_cam забарвлюється: зелений <0.3 м,
    жовтий 0.3-1.0, червоний >1.0."""
    x0, y0 = 20, 70
    w, h = 330, 168
    ov = img.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (90, 90, 90), 1)

    tag = "DETECTED" if fresh else "COASTING"
    tagc = (0, 220, 0) if fresh else (0, 165, 255)
    cv2.putText(img, f"track #{tid}  [{tag}]", (x0 + 12, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, tagc, 2, cv2.LINE_AA)
    # світові координати (X=ширина 0-9, Y=висота, Z=довжина 0-18)
    cv2.putText(img, f"X={Xw[0]:6.2f} m  (width 0-9)", (x0 + 12, y0 + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Y={Xw[1]:6.2f} m  (height)", (x0 + 12, y0 + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Z={Xw[2]:6.2f} m  (length 0-18)", (x0 + 12, y0 + 106),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    # дистанція до камери + джиттер
    adz = abs(dz)
    dzc = (0, 220, 0) if adz < 0.3 else (0, 215, 215) if adz < 1.0 \
        else (0, 60, 255)
    cv2.putText(img, f"Z_cam={zcam:6.2f} m  (depth)", (x0 + 12, y0 + 134),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"dZ_cam={dz:+5.2f} m/frame", (x0 + 12, y0 + 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, dzc, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Overlay IMMTracker JSONL на відео")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--trail", type=int, default=25,
                    help="Скільки останніх позицій малювати слідом (default 25)")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Обмежити к-сть кадрів (0 = всі)")
    args = ap.parse_args()

    header, recs = load_jsonl(args.jsonl)
    by_frame = {r["frame_0based"]: r for r in recs if r.get("type") == "frame"}
    K, R, cam = build_calibration(header)
    # Phase F (padding sweep): підпис ефективного діаметра D_eff і похідного
    # «порожнього відступу» padding=(D_eff−0.21)/2 на сторону (см). Z_c∝D_eff,
    # тож це число прямо масштабує всю глибину — показуємо в HUD, щоб
    # порівнювати відео різних padding-значень ОКОМ.
    d_eff = float(header.get("tracker", {}).get("ball_diameter", 0.21))
    pad_cm = (d_eff - 0.21) / 2.0 * 100.0
    fr_range = header.get("frame_range", [0, len(recs)])
    f_start = int(fr_range[0])
    fps = float(header.get("fps", 50.0))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"[err] не відкрити відео: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))

    trails = {}      # track_id -> list of (px,py)
    prev_zcam = {}   # track_id -> попередня дистанція до камери (для ΔZ_cam)
    n = 0
    f_idx = f_start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rec = by_frame.get(f_idx)
        # каркас майданчика
        draw_court(frame, K, R, cam)

        if rec is not None:
            # сира детекція
            rd = rec.get("raw_detection") or {}
            if rd.get("detected"):
                u, v = int(rd["u"]), int(rd["v"])
                wb = float(rd.get("w_box", 0))
                rr = rec.get("reject_reason")
                col = (0, 255, 255) if rr is None else (0, 0, 255)
                cv2.circle(frame, (u, v), 5, col, -1, cv2.LINE_AA)
                if wb > 0:
                    s = int(wb / 2)
                    cv2.rectangle(frame, (u - s, v - s), (u + s, v + s),
                                  col, 1, cv2.LINE_AA)
                if rr:
                    cv2.putText(frame, f"REJECT: {rr}", (u + 8, v),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                cv2.LINE_AA)

            # треки
            tracks = rec.get("tracks", [])
            dom = None
            dom_info = None  # (Xw, z_cam, fresh)
            for t in tracks:
                xp = t["x_post"]
                Xw = np.array([xp[0], xp[3], xp[6]], float)
                px, zc = project(Xw, K, R, cam)
                if zc[0] <= 0:
                    continue
                p = tuple(px[0].astype(int))
                conf = t["state"] == "Confirmed"
                col = (0, 220, 0) if conf else (140, 140, 140)
                tid = t["track_id"]
                trails.setdefault(tid, []).append(p)
                trails[tid] = trails[tid][-args.trail:]
                for a, b in zip(trails[tid], trails[tid][1:]):
                    cv2.line(frame, a, b, col, 1, cv2.LINE_AA)
                cv2.circle(frame, p, 7, col, 2, cv2.LINE_AA)
                cv2.putText(frame,
                            f"#{tid} X{Xw[0]:.1f} Y{Xw[1]:.1f} Z{Xw[2]:.1f}",
                            (p[0] + 9, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, col, 1, cv2.LINE_AA)
                if conf and (dom is None or t["hits"] > dom["hits"]):
                    dom = t
                    dom_info = (Xw, float(zc[0]),
                                t["time_since_update"] == 0)

            # Панель координат + IMM-режими домінантного Confirmed-треку
            if dom is not None and dom_info is not None:
                Xw, zcam, fresh = dom_info
                tid = dom["track_id"]
                dz = zcam - prev_zcam.get(tid, zcam)
                prev_zcam[tid] = zcam
                draw_coord_panel(frame, tid, Xw, zcam, dz, fresh)
                draw_mode_bars(frame, dom.get("mu", [1, 0, 0]), (28, 250))

        # HUD
        nconf = rec.get("n_confirmed", 0) if rec else 0
        cv2.putText(frame, f"frame {f_idx}  t={f_idx / fps:.2f}s  conf={nconf}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(frame,
                    f"D_eff={d_eff:.3f}m  padding={pad_cm:.1f}cm/side",
                    (w - 470, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 230, 230), 2, cv2.LINE_AA)
        vw.write(frame)
        n += 1
        f_idx += 1
        if args.max_frames and n >= args.max_frames:
            break
        if n % 200 == 0:
            sys.stderr.write(f"  ... {n} кадрів\n")

    cap.release()
    vw.release()
    sys.stderr.write(f"[ok] {n} кадрів -> {args.out}\n")


if __name__ == "__main__":
    main()
