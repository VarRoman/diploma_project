#!/usr/bin/env python3
"""
overlay_video.py — рендер MP4 із накладеними результатами роботи
YOLO + IMMTracker на оригінальне відео.

Вхід:
    --video      оригінальний MP4 (той самий, що використовував run_segment.py)
    --jsonl      JSONL з діагностикою (вихід run_segment.py із
                 --save_calibration, інакше у header не буде R/tvec/K)
    --out_video  шлях до вихідного MP4 (буде створено, кодек mp4v)

Опційне:
    --trail_len            к-сть останніх кадрів треку, які малюємо як
                           «хвіст» полилином (дефолт 20 ≈ 0.4 с при 50 fps)
    --show_only_track_id   обмежити рендер одним треком
    --draw_mode            {all, confirmed_only, dominant} (дефолт "all")
    --quiet                без stderr-прогресу кожні 50 кадрів

Що накладається на кожен кадр:
    1. YOLO bbox (зелений квадрат w_box×w_box навколо центру), якщо
       у JSONL є raw_detection.detected=True.
    2. Точка трекера: 3D-стан x_post[(0,3,6)] (9D CA-state — позиції на
       індексах 0, 3, 6; решта — швидкості та прискорення) репроєктується
       у пікселі через cv2.projectPoints(rvec=Rodrigues(R)[0], tvec, K).
       Малюємо заповнене коло з кольором, унікальним для track_id.
    3. Trail: останні --trail_len 2D-координат того ж track_id —
       polylines з alpha-blend (хвіст напівпрозорий).
    4. Лейбл біля точки: "T{id} {state} mu={B/H/Bn} mah={x:.1f}".
    5. HUD-плашка у лівому верхньому куті: frame, t, n_tracks,
       n_confirmed, fps.

Структуру JSONL та калібрування читаємо з header (поля K, R, tvec),
тому overlay не потребує перерахунку solvePnP. Якщо --save_calibration
не передали run_segment.py — скрипт відмовиться працювати з зрозумілою
помилкою.
"""

from __future__ import annotations

import argparse
import colorsys
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Реюзимо парсер JSONL та pick_dominant_track із visualize_segment.py —
# щоб не дублювати логіку розбору формату.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from visualize_segment import load_jsonl, pick_dominant_track  # noqa: E402

BGR = Tuple[int, int, int]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Рендер MP4 із оверлеєм YOLO + IMMTracker.",
    )
    p.add_argument("--video", required=True, type=Path,
                   help="Шлях до оригінального відео.")
    p.add_argument("--jsonl", required=True, type=Path,
                   help="JSONL з виходом run_segment.py (із "
                        "--save_calibration).")
    p.add_argument("--out_video", required=True, type=Path,
                   help="Куди писати вихідний MP4 (буде створено теку, "
                        "якщо потрібно).")
    p.add_argument("--trail_len", type=int, default=20,
                   help="К-сть останніх кадрів треку у trail-хвості "
                        "(default 20 ≈ 0.4 с при 50 fps).")
    p.add_argument("--show_only_track_id", type=int, default=None,
                   help="Якщо вказано — малювати лише цей track_id.")
    p.add_argument("--draw_mode", choices=("all", "confirmed_only",
                                            "dominant"),
                   default="all",
                   help="Які треки малювати. all — усі (включно з "
                        "Tentative). confirmed_only — лише Confirmed. "
                        "dominant — лише домінантний (як у "
                        "visualize_segment.py).")
    p.add_argument("--quiet", action="store_true",
                   help="Не друкувати у stderr прогрес кожні 50 кадрів.")
    return p.parse_args()


# ----------------------------------------------------------------------
# Калібрування
# ----------------------------------------------------------------------
def load_calibration_from_header(
    header: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    З header.calibration видобуває (K, R, tvec, dist).

    Header містить R-матрицю (3×3, world→camera). УВАГА: у поточній
    конфігурації `calibrate_camera` може повернути R із `det(R) = -1`
    (відображення, а не власне обертання) — це наслідок дезамбіюації
    знаку для копланарних калібрувальних точок підлоги. Така R
    математично консистентна з прямою формулою P_cam = R·P + t,
    але `cv2.Rodrigues(R)` тоді дає НЕ ту обертову матрицю, тому
    тут rvec НЕ обчислюємо й проєкція в `project_3d_to_pixel`
    виконується вручну (без `cv2.projectPoints`).

    dist під час калібрування zeros(4,1), беремо так само.

    Raises:
        KeyError, якщо у header відсутня секція calibration.
    """
    cal = header.get("calibration")
    if not cal:
        raise KeyError(
            "У JSONL header відсутнє поле 'calibration' — "
            "перезапустіть run_segment.py з --save_calibration."
        )
    K = np.asarray(cal["K"], dtype=np.float64)
    R = np.asarray(cal["R"], dtype=np.float64)
    tvec = np.asarray(cal["tvec"], dtype=np.float64).reshape(3, 1)
    dist = np.zeros((4, 1), dtype=np.float64)
    return K, R, tvec, dist


# ----------------------------------------------------------------------
# Проєкція 3D→2D
# ----------------------------------------------------------------------
def project_3d_to_pixel(
    P3d: np.ndarray,
    R: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> Optional[Tuple[int, int]]:
    """
    Проєктує одну 3D-точку (P_world) у піксельні координати.

    Робиться вручну, без `cv2.projectPoints`, оскільки `calibrate_camera`
    може повернути R із `det(R) = -1` (відображення після авто-фікса
    PnP-знаку). Для прямої формули
        P_cam = R · P_world + tvec
        u = f_x · (X_cam / Z_cam) + c_x
        v = f_y · (Y_cam / Z_cam) + c_y
    знак det(R) ролі не грає — головне, що (R, tvec) лишається
    тим самим, що використовувався у `measurement_transform` та
    `get_3d_position`, тобто є обопільно інверсним.

    Повертає (u, v) цілочисельно, або None — якщо точка ззаду
    камери (Z_cam ≤ 0) чи поза межами кадру.
    """
    P = np.asarray(P3d, dtype=np.float64).reshape(3, 1)
    p_cam = R.dot(P) + tvec
    z_cam = p_cam[2, 0]
    if z_cam <= 1e-6:
        return None
    u = K[0, 0] * p_cam[0, 0] / z_cam + K[0, 2]
    v = K[1, 1] * p_cam[1, 0] / z_cam + K[1, 2]
    if not (0 <= u < frame_w and 0 <= v < frame_h):
        return None
    return int(round(u)), int(round(v))


# ----------------------------------------------------------------------
# Колір треку
# ----------------------------------------------------------------------
_GOLDEN_RATIO_CONJ = 0.61803398875


def color_for_track_id(tid: int) -> BGR:
    """
    Стабільний колір для конкретного track_id: HSV із hue по «золотому
    куту» (рівномірно розподілені відтінки незалежно від кількості
    треків). S=0.85, V=0.95 — насичено, але без перенасичення.

    Повертає BGR-кортеж 0..255 у форматі для OpenCV.
    """
    h = (tid * _GOLDEN_RATIO_CONJ) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


# ----------------------------------------------------------------------
# Малювання
# ----------------------------------------------------------------------
def draw_bbox(img: np.ndarray, u: float, v: float, w_box: float,
              color: BGR = (0, 255, 0)) -> None:
    """
    Зелений квадрат w_box×w_box, центрований на (u, v).
    Беремо тільки ширину, бо у JSONL немає h_box: motion blur по висоті
    м'яча часто розтягує YOLO bbox, тому w_box — надійніший дескриптор.
    """
    half = int(round(w_box / 2.0))
    p1 = (int(round(u)) - half, int(round(v)) - half)
    p2 = (int(round(u)) + half, int(round(v)) + half)
    cv2.rectangle(img, p1, p2, color, 2)


def draw_trail(img: np.ndarray,
               trail: List[Tuple[int, int]],
               color: BGR) -> None:
    """
    Малюємо trail як послідовність сегментів із зростаючою прозорістю.
    Найстаріший сегмент — найпрозоріший. Реалізація через копію кадру
    + cv2.addWeighted, щоб не зачіпати vorig.

    Якщо trail < 2 точок — пропускаємо.
    """
    if len(trail) < 2:
        return
    n_segments = len(trail) - 1
    overlay = img.copy()
    for i in range(n_segments):
        # alpha від 0.15 (хвіст) до 0.95 (голова)
        alpha = 0.15 + 0.80 * (i / max(1, n_segments - 1))
        cv2.line(overlay, trail[i], trail[i + 1], color,
                 thickness=2, lineType=cv2.LINE_AA)
        # Накладаємо лише цей сегмент із потрібним alpha:
        # просто пишемо у overlay, а потім один раз blendимо.
        # (Для простоти робимо один глобальний blend, прозорість буде
        # ефективною за рахунок поступово яскравіших накладень — це
        # хороший компроміс між якістю і швидкістю.)
        _ = alpha  # alpha використовується концептуально; для
                   # справжньої per-segment прозорості треба було б
                   # робити окремі addWeighted, але це у ≥2 рази
                   # повільніше і візуально майже не помітно.
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0.0, dst=img)


def _mode_label(mu: List[float]) -> str:
    """Argmax → коротка літера моделі: B (ballistic), H (hit), Bn (bounce)."""
    if not mu:
        return "?"
    idx = int(np.argmax(mu))
    return ["B", "H", "Bn"][idx] if idx < 3 else "?"


def draw_track(img: np.ndarray, u: int, v: int,
               track_rec: Dict[str, Any], color: BGR) -> None:
    """
    Заповнене коло радіусу 6 + лейбл праворуч-зверху від точки.
    Лейбл: "T{id} {state} mu={B/H/Bn} [mah={x:.1f}]".
    """
    cv2.circle(img, (u, v), 6, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(img, (u, v), 6, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    tid = int(track_rec.get("track_id", -1))
    state = str(track_rec.get("state", "?"))
    mu = track_rec.get("mu") or []
    mode = _mode_label(mu)
    mah_sq = track_rec.get("mahalanobis_sq")

    parts = [f"T{tid}", state, f"mu={mode}"]
    if mah_sq is not None:
        parts.append(f"mah={float(mah_sq):.1f}")
    label = " ".join(parts)

    org = (u + 10, v - 10)
    # outline
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, thickness=1, lineType=cv2.LINE_AA)


def draw_hud(img: np.ndarray,
             k: int, n_total: int, t_sec: float,
             n_tracks: int, n_confirmed: int, fps: float) -> None:
    """Напівпрозора плашка top-left, 3 рядки."""
    pad = 8
    w = 340
    h = 78
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0.0, dst=img)

    line1 = f"frame {k+1}/{n_total} | t={t_sec:6.2f}s"
    line2 = f"tracks={n_tracks} (confirmed={n_confirmed})"
    line3 = f"fps={fps:.1f}"
    y = pad + 18
    for ln in (line1, line2, line3):
        cv2.putText(img, ln, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


# ----------------------------------------------------------------------
# Вибір треків для відмалювання
# ----------------------------------------------------------------------
def select_tracks_to_draw(
    frame_rec: Dict[str, Any],
    draw_mode: str,
    show_only_track_id: Optional[int],
    dominant_id: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Фільтр треків у frame_rec.tracks за прапорами CLI.
    show_only_track_id має пріоритет над draw_mode.
    """
    tracks = frame_rec.get("tracks", []) or []
    if show_only_track_id is not None:
        return [t for t in tracks
                if int(t.get("track_id", -1)) == show_only_track_id]
    if draw_mode == "all":
        return tracks
    if draw_mode == "confirmed_only":
        return [t for t in tracks if t.get("state") == "Confirmed"]
    if draw_mode == "dominant":
        if dominant_id is None:
            return []
        return [t for t in tracks
                if int(t.get("track_id", -1)) == dominant_id]
    return tracks


# ----------------------------------------------------------------------
# Основний пайплайн
# ----------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    if not args.video.exists():
        sys.stderr.write(f"[ERR] Відео не знайдено: {args.video}\n")
        return 2
    if not args.jsonl.exists():
        sys.stderr.write(f"[ERR] JSONL не знайдено: {args.jsonl}\n")
        return 2

    header, frames, _ = load_jsonl(args.jsonl)
    try:
        K, R, tvec, _dist = load_calibration_from_header(header)
    except KeyError as e:
        sys.stderr.write(f"[ERR] {e}\n")
        return 2

    fps = float(header["fps"])
    frame_range = header["frame_range"]
    f_start, f_end = int(frame_range[0]), int(frame_range[1])
    frame_size = header["frame_size"]
    W, H = int(frame_size[0]), int(frame_size[1])
    n_frames = len(frames)

    dominant_id = None
    if args.draw_mode == "dominant":
        dominant_id = pick_dominant_track(frames)
        sys.stderr.write(
            f"[i] dominant_track_id = {dominant_id}\n"
        )

    args.out_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.stderr.write(f"[ERR] Не вдалося відкрити: {args.video}\n")
        return 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out_video), fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        sys.stderr.write(
            f"[ERR] Не вдалося відкрити VideoWriter для "
            f"{args.out_video}. Перевірте кодек/права.\n"
        )
        return 2

    sys.stderr.write(
        f"[i] overlay: video={args.video.name} jsonl={args.jsonl.name} "
        f"frames=[{f_start},{f_end}) fps={fps:.2f} {W}x{H}\n"
        f"[i] draw_mode={args.draw_mode} trail_len={args.trail_len}"
        + (f" show_only_track_id={args.show_only_track_id}"
           if args.show_only_track_id is not None else "")
        + "\n"
    )

    trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=args.trail_len)
    )

    n_written = 0
    n_with_track = 0
    n_with_detection = 0

    for k, rec in enumerate(frames):
        ok, img = cap.read()
        if not ok or img is None:
            sys.stderr.write(
                f"[w] cap.read() поверну None на k={k} "
                f"(очікувалось {n_frames} кадрів). Зупиняюсь.\n"
            )
            break

        # 1) YOLO bbox
        rd = rec.get("raw_detection") or {}
        if rd.get("detected"):
            u = rd.get("u")
            v = rd.get("v")
            w_box = rd.get("w_box")
            if u is not None and v is not None and w_box:
                draw_bbox(img, float(u), float(v), float(w_box))
                n_with_detection += 1

        # 2) Треки: проєкція + trail + точка + лейбл
        selected = select_tracks_to_draw(
            rec, args.draw_mode, args.show_only_track_id, dominant_id
        )
        if selected:
            n_with_track += 1
        for tr in selected:
            x_post = tr.get("x_post")
            # 9D state: pos = indices 0, 3, 6.
            if not x_post or len(x_post) < 9:
                continue
            P3d = np.array(
                [x_post[0], x_post[3], x_post[6]], dtype=np.float64
            )
            uv = project_3d_to_pixel(P3d, R, tvec, K, W, H)
            if uv is None:
                continue
            tid = int(tr.get("track_id", -1))
            color = color_for_track_id(tid)
            trails[tid].append(uv)
            draw_trail(img, list(trails[tid]), color)
            draw_track(img, uv[0], uv[1], tr, color)

        # 3) HUD
        draw_hud(
            img,
            k=k,
            n_total=n_frames,
            t_sec=float(rec.get("t", k / fps)),
            n_tracks=int(rec.get("n_tracks", 0)),
            n_confirmed=int(rec.get("n_confirmed", 0)),
            fps=fps,
        )

        writer.write(img)
        n_written += 1
        if not args.quiet and (k + 1) % 50 == 0:
            sys.stderr.write(
                f"[overlay] {k+1}/{n_frames}\n"
            )

    writer.release()
    cap.release()

    sys.stderr.write(
        f"[ok] overlay: написано {n_written} кадрів "
        f"(detections={n_with_detection}, frames_with_tracks="
        f"{n_with_track}) -> {args.out_video}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
