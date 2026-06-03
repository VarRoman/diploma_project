#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Інтерактивний пікер калібрувальних точок для СТАТИЧНОЇ камери трансляції.

Контекст: відео — офіційна трансляція (професійна камера з НЕВІДОМИМИ
інтринсиками). Фокусна f=1300 у коді була від камери Pixel 9 і до цього
відео не стосується. Єдине принципове джерело f — геометрія корту: клікаємо
точки з відомими 3D-координатами, далі оцінюємо f, що мінімізує репроєкцію.

Камера статична (без зуму/панорамування) → точка корту має ОДНАКОВИЙ піксель
у кожному кадрі. Тому якщо гравець перекрив точку — перемкни кадр (n/p) і
клікни її там, де видно: піксель той самий.

Світова СК (як у get_court_position_methods): X — ширина (0..9 м),
Z — довжина (0..18 м), Y — вгору (підлога = 0). "Ближня" лінія (Z=0) —
ВНИЗУ кадру (ближче до камери), "дальня" (Z=18) — ВГОРІ кадру.

КЕРУВАННЯ:
  ліва кнопка миші  — поставити поточну цільову точку (підсвічена жовтим)
  колесо миші       — зум in/out під курсором
  +/-               — зум in/out по центру
  стрілки / ПКМ-драг — пан (зсув видимої області)
  n / p             — наступний / попередній кадр (+10)
  N / P (Shift)     — стрибок на +100 / -100 кадрів
  u                 — undo (прибрати останню поставлену точку)
  s                 — skip (позначити поточну точку як невидиму, пропустити)
  j                 — перейти до точки за номером (запит у терміналі)
  w                 — записати JSON і ПРОДОВЖИТИ
  q / ESC           — записати JSON і вийти

Вивід (JSON): для кожної точки label, world [X,Y,Z], pixel [u,v] або null.
Наприкінці робиться швидка оцінка f (sweep репроєкції) для миттєвого фідбеку.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def build_point_plan(net_height: float):
    """
    Список калібрувальних точок у порядку клікання.
    Кожен елемент: (label, X, Y, Z, hint).
    Підлогові точки (Y=0) — по периметру/лініях; дві верхні точки сітки
    (Y=net_height) дають ВЕРТИКАЛЬНИЙ референс, що різко покращує оцінку f
    (ламає копланарну виродженість 4 кутів).
    """
    H = net_height
    return [
        # --- Підлога (Y=0): ближня лінія Z=0 внизу кадру ---
        ("near-left corner",        0.0, 0.0,  0.0, "ближній лівий кут (низ-ліво)"),
        ("near-right corner",       9.0, 0.0,  0.0, "ближній правий кут (низ-право)"),
        # --- Дальня лінія Z=18 вгорі кадру ---
        ("far-right corner",        9.0, 0.0, 18.0, "дальній правий кут (верх-право)"),
        ("far-left corner",         0.0, 0.0, 18.0, "дальній лівий кут (верх-ліво)"),
        # --- Ближня лінія атаки Z=6 (3 м від центру до ближньої сторони) ---
        ("near attack L (Z=6)",     0.0, 0.0,  6.0, "ближня лінія атаки ∩ ЛІВА бічна"),
        ("near attack R (Z=6)",     9.0, 0.0,  6.0, "ближня лінія атаки ∩ ПРАВА бічна"),
        # --- Центральна лінія Z=9 (під сіткою) ---
        ("center line L (Z=9)",     0.0, 0.0,  9.0, "центральна лінія ∩ ЛІВА бічна (основа сітки зліва)"),
        ("center line R (Z=9)",     9.0, 0.0,  9.0, "центральна лінія ∩ ПРАВА бічна (основа сітки справа)"),
        # --- Дальня лінія атаки Z=12 ---
        ("far attack L (Z=12)",     0.0, 0.0, 12.0, "дальня лінія атаки ∩ ЛІВА бічна"),
        ("far attack R (Z=12)",     9.0, 0.0, 12.0, "дальня лінія атаки ∩ ПРАВА бічна"),
        # --- Верх сітки (Y=H) над основами — ВЕРТИКАЛЬНИЙ референс ---
        ("net TOP left (Y=%.2f)" % H,   0.0, H,  9.0, "верх стрічки сітки біля ЛІВОЇ антени (рівно над center line L)"),
        ("net TOP right (Y=%.2f)" % H,  9.0, H,  9.0, "верх стрічки сітки біля ПРАВОЇ антени (рівно над center line R)"),
    ]


class View:
    """Зум/пан із точним зворотним відображенням display→image."""
    def __init__(self, img_w, img_h, disp_w, disp_h):
        self.iw, self.ih = img_w, img_h
        self.dw, self.dh = disp_w, disp_h
        self.zoom = min(disp_w / img_w, disp_h / img_h)
        self.min_zoom = self.zoom
        self.off_x = 0.0
        self.off_y = 0.0
        self._clamp()

    def _clamp(self):
        vis_w = self.dw / self.zoom
        vis_h = self.dh / self.zoom
        self.off_x = float(np.clip(self.off_x, 0, max(0, self.iw - vis_w)))
        self.off_y = float(np.clip(self.off_y, 0, max(0, self.ih - vis_h)))

    def disp_to_img(self, dx, dy):
        return self.off_x + dx / self.zoom, self.off_y + dy / self.zoom

    def img_to_disp(self, ix, iy):
        return (ix - self.off_x) * self.zoom, (iy - self.off_y) * self.zoom

    def zoom_at(self, dx, dy, factor):
        ix, iy = self.disp_to_img(dx, dy)
        self.zoom = float(np.clip(self.zoom * factor, self.min_zoom, 40.0))
        # тримаємо точку під курсором на місці
        self.off_x = ix - dx / self.zoom
        self.off_y = iy - dy / self.zoom
        self._clamp()

    def pan(self, d_ix, d_iy):
        self.off_x += d_ix
        self.off_y += d_iy
        self._clamp()

    def render(self, img):
        vis_w = int(round(self.dw / self.zoom))
        vis_h = int(round(self.dh / self.zoom))
        x0 = int(round(self.off_x))
        y0 = int(round(self.off_y))
        x1 = min(self.iw, x0 + vis_w)
        y1 = min(self.ih, y0 + vis_h)
        crop = img[y0:y1, x0:x1]
        interp = cv2.INTER_NEAREST if self.zoom > 1.0 else cv2.INTER_AREA
        return cv2.resize(crop, (self.dw, self.dh), interpolation=interp)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", type=Path,
                   default=Path("detect_infos/calibration_points.json"))
    p.add_argument("--net_height", type=float, default=2.43,
                   help="Висота сітки (м). ЧОЛОВІЧИЙ FIVB=2.43, жіночий=2.24. "
                        "ПЕРЕВІР для цього матчу!")
    p.add_argument("--start_frame", type=int, default=300)
    p.add_argument("--disp_w", type=int, default=1600)
    p.add_argument("--disp_h", type=int, default=900)
    return p.parse_args(argv)


def quick_focal_estimate(world_pts, img_pts, img_w, img_h):
    """Швидкий sweep f: для кожного f робимо solvePnP і рахуємо середню
    репроєкцію. Повертає (best_f, best_err, table)."""
    if len(world_pts) < 4:
        return None, None, []
    wp = np.asarray(world_pts, np.float32)
    ip = np.asarray(img_pts, np.float32)
    dist = np.zeros((4, 1), np.float32)
    coplanar = np.allclose(wp[:, 1], 0.0)  # усі Y=0?
    flag = cv2.SOLVEPNP_ITERATIVE
    table = []
    best = (None, 1e18)
    for f in range(800, 9001, 100):
        K = np.array([[f, 0, img_w / 2.0], [0, f, img_h / 2.0], [0, 0, 1.0]],
                     np.float32)
        ok, rvec, tvec = cv2.solvePnP(wp, ip, K, dist, flags=flag)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(wp, rvec, tvec, K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - ip, axis=1)))
        table.append((f, err))
        if err < best[1]:
            best = (f, err)
    return best[0], best[1], table, coplanar


def main(argv=None):
    args = parse_args(argv)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.stderr.write(f"[e] не відкрив відео: {args.video}\n")
        return 1
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    iw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ih = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    plan = build_point_plan(args.net_height)
    n_pts = len(plan)
    picks = [None] * n_pts   # pixel (u,v) у КООРД. ОРИГІНАЛУ або None(skip)
    cur = 0                  # індекс поточної цільової точки

    frame_idx = max(0, min(args.start_frame, n_total - 1))

    def read_frame(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        return fr if ok else None

    frame = read_frame(frame_idx)
    if frame is None:
        sys.stderr.write("[e] не зчитав стартовий кадр\n")
        return 1

    view = View(iw, ih, args.disp_w, args.disp_h)
    win = "calib picker"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    state = {"pan_anchor": None}

    def on_mouse(event, x, y, flags, _param):
        nonlocal cur
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = view.disp_to_img(x, y)
            if 0 <= ix < iw and 0 <= iy < ih and cur < n_pts:
                picks[cur] = (float(ix), float(iy))
                # перейти до наступної НЕпоставленої точки
                cur = next((k for k in range(cur + 1, n_pts)
                            if picks[k] is None), n_pts)
        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.25 if flags > 0 else 0.8
            view.zoom_at(x, y, factor)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["pan_anchor"] = (x, y, view.off_x, view.off_y)
        elif event == cv2.EVENT_MOUSEMOVE and state["pan_anchor"] is not None:
            ax, ay, ox, oy = state["pan_anchor"]
            view.off_x = ox - (x - ax) / view.zoom
            view.off_y = oy - (y - ay) / view.zoom
            view._clamp()
        elif event == cv2.EVENT_RBUTTONUP:
            state["pan_anchor"] = None

    cv2.setMouseCallback(win, on_mouse)

    def draw():
        disp = view.render(frame)
        # позначки вже поставлених точок
        for k, pk in enumerate(picks):
            if pk is None:
                continue
            dx, dy = view.img_to_disp(*pk)
            if -20 <= dx <= args.disp_w + 20 and -20 <= dy <= args.disp_h + 20:
                color = (0, 200, 0)
                cv2.drawMarker(disp, (int(dx), int(dy)), color,
                               cv2.MARKER_CROSS, 18, 2)
                cv2.circle(disp, (int(dx), int(dy)), 2, color, -1)
                cv2.putText(disp, str(k + 1), (int(dx) + 6, int(dy) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # верхній банер
        n_done = sum(1 for p in picks if p is not None)
        banner_h = 64
        cv2.rectangle(disp, (0, 0), (args.disp_w, banner_h), (0, 0, 0), -1)
        if cur < n_pts:
            lbl, X, Y, Z, hint = plan[cur]
            t1 = f"[{cur+1}/{n_pts}] КЛІКНИ: {lbl}  world=({X:.1f},{Y:.2f},{Z:.1f})"
            t2 = f"  {hint}"
            cv2.putText(disp, t1, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp, t2, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(disp, "Усі точки оброблено. w=save+continue, q=save+quit",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)
        info = (f"frame {frame_idx}/{n_total-1}  zoom x{view.zoom:.1f}  "
                f"done {n_done}/{n_pts}  |  click  wheel=zoom  n/p frame  "
                f"u undo  s skip  j jump  w save  q quit")
        cv2.putText(disp, info, (8, args.disp_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(win, disp)

    def save_json():
        out = {
            "video": str(args.video),
            "net_height": args.net_height,
            "frame_size": [iw, ih],
            "convention": "X=width[0..9], Y=up[floor=0], Z=length[0..18]",
            "points": [
                {"label": plan[k][0],
                 "world": [plan[k][1], plan[k][2], plan[k][3]],
                 "pixel": list(picks[k]) if picks[k] is not None else None}
                for k in range(n_pts)
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        sys.stderr.write(f"[ok] збережено {args.out} "
                         f"({sum(1 for p in picks if p)} точок)\n")
        # миттєвий фідбек: оцінка f
        wp = [plan[k][1:4] for k in range(n_pts) if picks[k] is not None]
        ipx = [picks[k] for k in range(n_pts) if picks[k] is not None]
        res = quick_focal_estimate(wp, ipx, iw, ih)
        if res[0] is not None:
            bf, be, table, coplanar = res
            sys.stderr.write(
                f"[f] оцінка фокусної: f≈{bf} px, репроєкція≈{be:.2f} px "
                f"(N={len(wp)}, {'КОПЛАНАРНІ' if coplanar else 'є вертикальні'} точки)\n")
            # кілька значень кривої навколо мінімуму
            near = [t for t in table if abs(t[0] - bf) <= 600]
            sys.stderr.write("    f→err: " +
                             "  ".join(f"{f}:{e:.1f}" for f, e in near[::2]) + "\n")
        else:
            sys.stderr.write("[f] замало точок для оцінки f (потрібно ≥4)\n")

    sys.stderr.write(__doc__ + "\n")
    sys.stderr.write(f"[i] video={args.video.name} {iw}x{ih} frames={n_total} "
                     f"net_height={args.net_height} м (перевір ч/ж!)\n")

    while True:
        draw()
        key = cv2.waitKey(20) & 0xFFFF
        if key == 0xFFFF:
            continue
        ch = key & 0xFF
        if ch in (ord('q'), 27):           # q / ESC
            save_json()
            break
        elif ch == ord('w'):
            save_json()
        elif ch == ord('n'):
            frame_idx = min(n_total - 1, frame_idx + 10)
            frame = read_frame(frame_idx) if read_frame(frame_idx) is not None else frame
        elif ch == ord('p'):
            frame_idx = max(0, frame_idx - 10)
            frame = read_frame(frame_idx)
        elif ch == ord('N'):
            frame_idx = min(n_total - 1, frame_idx + 100)
            frame = read_frame(frame_idx)
        elif ch == ord('P'):
            frame_idx = max(0, frame_idx - 100)
            frame = read_frame(frame_idx)
        elif ch == ord('u'):               # undo останньої поставленої
            placed = [k for k in range(n_pts) if picks[k] is not None]
            if placed:
                last = placed[-1]
                picks[last] = None
                cur = last
        elif ch == ord('s'):               # skip поточної
            if cur < n_pts:
                picks[cur] = None
                cur = next((k for k in range(cur + 1, n_pts)
                            if picks[k] is None), n_pts)
        elif ch == ord('j'):               # jump до номера
            cv2.destroyWindow(win)
            try:
                tgt = int(input(f"перейти до точки № (1..{n_pts}): ")) - 1
                if 0 <= tgt < n_pts:
                    cur = tgt
            except Exception:
                pass
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(win, on_mouse)
        elif ch in (ord('+'), ord('=')):
            view.zoom_at(args.disp_w // 2, args.disp_h // 2, 1.25)
        elif ch in (ord('-'), ord('_')):
            view.zoom_at(args.disp_w // 2, args.disp_h // 2, 0.8)
        elif ch == 81:   # left arrow
            view.pan(-50 / view.zoom, 0)
        elif ch == 83:   # right
            view.pan(50 / view.zoom, 0)
        elif ch == 82:   # up
            view.pan(0, -50 / view.zoom)
        elif ch == 84:   # down
            view.pan(0, 50 / view.zoom)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
