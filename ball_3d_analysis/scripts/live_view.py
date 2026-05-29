#!/usr/bin/env python3
"""
live_view.py — інтерактивний переглядач відео з оверлеєм YOLO + IMMTracker.

На відміну від overlay_video.py (рендерить пасивний MP4), цей скрипт
показує кадри через cv2.imshow і дозволяє «наживо» спостерігати:

    - як росте trail трекера у часі (кожен крок додає точку);
    - де YOLO відловила/втратила м'яч (bbox з'являються/зникають);
    - як IMM-режим змінюється (мітка mu= у лейблі точки);
    - як трек переходить Tentative → Confirmed → Deleted.

Підтримує паузу/пошагове просування/seek назад — щоб зупинятися на
складних моментах (удари, оклюзії, перетин кадру) і вивчати поведінку
алгоритму візуально та евристично.

Контроли (кл-ші відображаються у HUD):
    SPACE       пауза / продовжити play
    n або →     наступний кадр (працює і у play, і у pause)
    p або ←     попередній кадр (працює у pause, до 240 кадрів назад
                із кешу; далі через cap.set — повільніше, але працює)
    J           seek назад на 30 кадрів (≈ 0.6 с)
    L           seek вперед на 30 кадрів
    HOME        перейти на початок
    END         перейти у кінець
    +  або =    прискорити (×1 → ×2 → ×4 → ×8)
    -  або _    уповільнити (×1 → ×0.5 → ×0.25)
    r           reset playback до початку, trails очистяться автоматично
    h           toggle HUD
    t           toggle trails
    b           toggle bbox (YOLO детекції)
    1           draw_mode = all (усі треки)
    2           draw_mode = confirmed_only
    3           draw_mode = dominant
    s           зберегти поточний кадр у файл (PNG поряд із JSONL)
    q  або ESC  вийти

Вікно автоматично масштабується. За замовчуванням висота вікна 720 px
(можна змінити через --window_height); відео ресайзиться зі збереженням
співвідношення сторін.

Використання:
    python scripts/live_view.py \\
        --video ../data/videos/Japan_vs_Poland_ultrashort.mp4 \\
        --jsonl logs/seg_ultrashort_full.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# Реюзимо парсер JSONL + усі функції малювання з overlay_video.py.
from visualize_segment import load_jsonl, pick_dominant_track  # noqa: E402
from overlay_video import (  # noqa: E402
    color_for_track_id,
    draw_bbox,
    draw_track,
    draw_trail,
    load_calibration_from_header,
    project_3d_to_pixel,
    select_tracks_to_draw,
)


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Інтерактивний переглядач відео з оверлеєм трекера.",
    )
    p.add_argument("--video", required=True, type=Path,
                   help="Шлях до оригінального відео.")
    p.add_argument("--jsonl", required=True, type=Path,
                   help="JSONL з виходом run_segment.py (із "
                        "--save_calibration).")
    p.add_argument("--trail_len", type=int, default=40,
                   help="К-сть останніх кадрів треку у trail-хвості "
                        "(default 40 ≈ 0.8 с при 50 fps). У live-режимі "
                        "ставимо вищий за overlay_video.py default 20, "
                        "бо при паузі хочемо бачити більше історії.")
    p.add_argument("--start_frame", type=int, default=0,
                   help="З якого індексу JSONL стартувати (default 0).")
    p.add_argument("--draw_mode", choices=("all", "confirmed_only",
                                            "dominant"),
                   default="all",
                   help="Стартовий режим відображення треків. "
                        "Можна перемикати наживо клавішами 1/2/3.")
    p.add_argument("--window_height", type=int, default=720,
                   help="Висота вікна перегляду у пікселях "
                        "(ширина обчислюється зі збереженням ratio). "
                        "Default 720.")
    p.add_argument("--cache_size", type=int, default=240,
                   help="К-сть кадрів у RAM-кеші для швидких ←/→. "
                        "Default 240 ≈ 4.8 с при 50 fps. Збільшення "
                        "коштує ~6 МБ/кадр RAM при 1920×1080.")
    p.add_argument("--snapshot_dir", type=Path, default=None,
                   help="Куди писати знімки клавішею 's'. Default — "
                        "тека з JSONL.")
    return p.parse_args()


# ----------------------------------------------------------------------
class FrameReader:
    """
    Обгортка над cv2.VideoCapture з LRU-кешем розкодованих кадрів.

    Випадки використання:
        get(idx) у послідовному порядку — швидко (sequential read);
        get(idx) "назад" — якщо у кеші, миттєво; інакше cap.set
        (повільно через ребуфер відеокодеку, але рідко).

    `idx` — індекс відносно JSONL-фрейму 0; реальна позиція = f_start + idx.
    """

    def __init__(self, video_path: Path, f_start: int,
                 n_frames: int, cache_size: int = 240):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Не вдалося відкрити {video_path}")
        self.f_start = int(f_start)
        self.n_frames = int(n_frames)
        self.cache_size = int(cache_size)
        # OrderedDict використовуємо як LRU: move_to_end на hit,
        # popitem(last=False) для FIFO-витіснення.
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        # Позиція "наступного кадру, який поверне cap.read()" — щоб
        # знати, чи потрібно дзвонити cap.set перед .read().
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_start)
        self._next_seq_idx = 0

    def get(self, idx: int) -> Optional[np.ndarray]:
        """
        Повертає кадр для логічного індексу idx (0..n_frames-1).
        None — якщо out-of-range або декодер відмовив.
        """
        if idx < 0 or idx >= self.n_frames:
            return None
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        # Cache miss: можливо потрібен seek.
        if idx != self._next_seq_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_start + idx)
            self._next_seq_idx = idx
        ok, img = self.cap.read()
        if not ok or img is None:
            return None
        self._next_seq_idx = idx + 1
        # Кешуємо копію, щоб подальший cv2.rectangle/etc. не мутував
        # дані всередині кеша.
        self._cache[idx] = img.copy()
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return self._cache[idx]

    def release(self) -> None:
        self.cap.release()


# ----------------------------------------------------------------------
def rebuild_trails(
    frames: List[Dict[str, Any]],
    K: np.ndarray,
    R: np.ndarray,
    tvec: np.ndarray,
    W: int,
    H: int,
    end_idx: int,
    trail_len: int,
) -> Dict[int, Deque[Tuple[int, int]]]:
    """
    Перебудовує trail-словник з нуля, дивлячись на проміжок
    [max(0, end_idx-trail_len+1) .. end_idx] включно.

    Виклик потрібен при будь-якому seek'у (назад або вперед), щоб не
    "запам'ятовувати" точки, які з точки зору поточної позиції ще не
    мали б існувати (для seek назад) або не накопичились (для seek
    вперед).

    Послідовний хід (наступний кадр) робиться окремо — інкрементально.
    """
    trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=trail_len)
    )
    if end_idx < 0:
        return trails
    start = max(0, end_idx - trail_len + 1)
    for k in range(start, end_idx + 1):
        rec = frames[k]
        for tr in rec.get("tracks", []) or []:
            x_post = tr.get("x_post") or []
            if len(x_post) < 6:
                continue
            P3d = np.array(
                [x_post[0], x_post[2], x_post[4]], dtype=np.float64
            )
            uv = project_3d_to_pixel(P3d, R, tvec, K, W, H)
            if uv is None:
                continue
            trails[int(tr.get("track_id", -1))].append(uv)
    return trails


def update_trails_step(
    trails: Dict[int, Deque[Tuple[int, int]]],
    rec: Dict[str, Any],
    K: np.ndarray,
    R: np.ndarray,
    tvec: np.ndarray,
    W: int,
    H: int,
) -> None:
    """Інкрементально додає точки нового кадру у trails (для forward step)."""
    for tr in rec.get("tracks", []) or []:
        x_post = tr.get("x_post") or []
        if len(x_post) < 6:
            continue
        P3d = np.array(
            [x_post[0], x_post[2], x_post[4]], dtype=np.float64
        )
        uv = project_3d_to_pixel(P3d, R, tvec, K, W, H)
        if uv is None:
            continue
        trails[int(tr.get("track_id", -1))].append(uv)


# ----------------------------------------------------------------------
def draw_live_hud(
    img: np.ndarray,
    *,
    current_idx: int,
    n_total: int,
    t_sec: float,
    n_tracks: int,
    n_confirmed: int,
    fps: float,
    paused: bool,
    speed: float,
    draw_mode: str,
    show_bbox: bool,
    show_trails: bool,
) -> None:
    """
    HUD у лівому верхньому куті, ширший за overlay_video.draw_hud — щоб
    помістити стани toggles та підказку про ключі. Прозорий бекдроп.
    """
    pad = 8
    w = 470
    h = 142
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0.0, dst=img)

    state_str = "PAUSE" if paused else "PLAY"
    color_state = (0, 200, 255) if paused else (0, 255, 0)
    cv2.putText(img, state_str, (pad, pad + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_state, 2, cv2.LINE_AA)

    bar_x = pad + 90
    bar_y = pad + 10
    bar_w = w - bar_x - pad
    bar_h = 8
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), -1)
    if n_total > 0:
        progress = min(1.0, max(0.0, (current_idx + 1) / n_total))
        cv2.rectangle(img,
                      (bar_x, bar_y),
                      (bar_x + int(bar_w * progress), bar_y + bar_h),
                      (200, 200, 255), -1)

    line2 = (f"frame {current_idx + 1}/{n_total} | t={t_sec:6.2f}s "
             f"| speed=x{speed:g}")
    line3 = (f"tracks={n_tracks} (confirmed={n_confirmed}) | fps_src={fps:.1f}")
    line4 = (f"draw_mode={draw_mode}  "
             f"[t]rails={'on' if show_trails else 'off'}  "
             f"[b]box={'on' if show_bbox else 'off'}")
    line5 = "keys: SPACE=play/pause  N/P=step  J/L=skip30  1/2/3=mode  q=quit"

    y = pad + 50
    for ln in (line2, line3, line4, line5):
        cv2.putText(img, ln, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 21


# ----------------------------------------------------------------------
def render_frame(
    img: np.ndarray,
    rec: Dict[str, Any],
    trails: Dict[int, Deque[Tuple[int, int]]],
    K: np.ndarray,
    R: np.ndarray,
    tvec: np.ndarray,
    W: int,
    H: int,
    *,
    draw_mode: str,
    dominant_id: Optional[int],
    show_only_track_id: Optional[int],
    show_bbox: bool,
    show_trails: bool,
) -> np.ndarray:
    """
    Малює оверлей на копії кадру (вихідне зображення з cap.read лишається
    чистим, бо знаходиться у кеші FrameReader і не повинно мутуватись).
    Повертає нове зображення з оверлеєм.
    """
    out = img.copy()

    if show_bbox:
        rd = rec.get("raw_detection") or {}
        if rd.get("detected"):
            u = rd.get("u")
            v = rd.get("v")
            w_box = rd.get("w_box")
            if u is not None and v is not None and w_box:
                draw_bbox(out, float(u), float(v), float(w_box))

    selected = select_tracks_to_draw(
        rec, draw_mode, show_only_track_id, dominant_id
    )
    for tr in selected:
        x_post = tr.get("x_post") or []
        if len(x_post) < 6:
            continue
        P3d = np.array(
            [x_post[0], x_post[2], x_post[4]], dtype=np.float64
        )
        uv = project_3d_to_pixel(P3d, R, tvec, K, W, H)
        if uv is None:
            continue
        tid = int(tr.get("track_id", -1))
        color = color_for_track_id(tid)
        if show_trails and tid in trails:
            draw_trail(out, list(trails[tid]), color)
        draw_track(out, uv[0], uv[1], tr, color)

    return out


# ----------------------------------------------------------------------
# Розпізнавання клавіш — cross-platform трохи кострубата справа.
# cv2.waitKeyEx повертає extended код:
#   - звичайні ASCII клавіші: 0..255 (літери, цифри, пробіл, ESC=27);
#   - стрілки на Linux X11: 65361 (←), 65362 (↑), 65363 (→), 65364 (↓);
#   - стрілки на Windows: 2424832 (←), 2490368 (↑), 2555904 (→), 2621440 (↓);
#   - HOME/END на Linux: 65360 / 65367.
#
# УВАГА: 65361 & 0xFF = 81 = 'Q' — якщо просто маскувати нижній байт,
# стрілка ВЛІВО прочитається як 'Q' (quit). Тому *спочатку* перевіряємо
# спеціальні коди (> 255), і лише якщо не збіг — fall-back на ASCII.
_KEY_LEFT = {65361, 2424832}
_KEY_RIGHT = {65363, 2555904}
_KEY_HOME = {65360, 2359296}
_KEY_END = {65367, 2293760}
# Стрілки UP/DOWN зарезервовані під майбутній zoom/speed-set, але поки
# не використовуємо — лишаємо коменти для очевидності.
_KEY_UP = {65362, 2490368}
_KEY_DOWN = {65364, 2621440}


# ----------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    if not args.video.exists():
        sys.stderr.write(f"[ERR] Відео не знайдено: {args.video}\n")
        return 2
    if not args.jsonl.exists():
        sys.stderr.write(f"[ERR] JSONL не знайдено: {args.jsonl}\n")
        return 2

    header, frames, _summary = load_jsonl(args.jsonl)
    try:
        K, R, tvec, _dist = load_calibration_from_header(header)
    except KeyError as e:
        sys.stderr.write(f"[ERR] {e}\n")
        return 2

    fps_src = float(header["fps"])
    frame_range = header["frame_range"]
    f_start = int(frame_range[0])
    frame_size = header["frame_size"]
    W, H = int(frame_size[0]), int(frame_size[1])
    n_frames = len(frames)

    if n_frames == 0:
        sys.stderr.write("[ERR] JSONL не містить кадрів.\n")
        return 2

    dominant_id = pick_dominant_track(frames)
    sys.stderr.write(f"[i] dominant_track_id = {dominant_id}\n")

    reader = FrameReader(args.video, f_start, n_frames,
                         cache_size=args.cache_size)

    # Стартовий стан
    current_idx = max(0, min(n_frames - 1, args.start_frame))
    paused = True   # стартуємо у паузі, щоб користувач встиг побачити кадр 1
    speed = 1.0
    draw_mode = args.draw_mode
    show_bbox = True
    show_trails = True
    show_hud = True
    trail_len = args.trail_len
    speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    # Початкові trails: до start_frame включно
    trails = rebuild_trails(frames, K, R, tvec, W, H,
                            end_idx=current_idx,
                            trail_len=trail_len)

    snapshot_dir = args.snapshot_dir or args.jsonl.parent

    # Вікно
    win_name = "live_view: IMMTracker overlay"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    target_h = int(args.window_height)
    target_w = int(round(W * target_h / H))
    cv2.resizeWindow(win_name, target_w, target_h)

    sys.stderr.write(
        f"[i] live_view: {args.video.name}  {n_frames} frames @ "
        f"{fps_src:.1f} fps  {W}x{H} → window {target_w}x{target_h}\n"
        f"[i] контроли: SPACE=play/pause, N/→=next, P/←=prev, "
        f"J/L=skip30, +/-=speed, 1/2/3=mode, t/b/h=toggles, "
        f"s=snapshot, r=reset, q/ESC=quit\n"
    )

    last_step_time = time.perf_counter()

    def goto(new_idx: int, *, reset_trails: bool = True) -> None:
        nonlocal current_idx, trails
        new_idx = max(0, min(n_frames - 1, new_idx))
        if new_idx == current_idx and not reset_trails:
            return
        current_idx = new_idx
        if reset_trails:
            trails = rebuild_trails(frames, K, R, tvec, W, H,
                                    end_idx=current_idx,
                                    trail_len=trail_len)

    while True:
        rec = frames[current_idx]
        img = reader.get(current_idx)
        if img is None:
            sys.stderr.write(
                f"[w] cap.read() повернув None для idx={current_idx}\n"
            )
            break

        display = render_frame(
            img, rec, trails, K, R, tvec, W, H,
            draw_mode=draw_mode,
            dominant_id=dominant_id,
            show_only_track_id=None,
            show_bbox=show_bbox,
            show_trails=show_trails,
        )

        if show_hud:
            draw_live_hud(
                display,
                current_idx=current_idx,
                n_total=n_frames,
                t_sec=float(rec.get("t", current_idx / fps_src)),
                n_tracks=int(rec.get("n_tracks", 0)),
                n_confirmed=int(rec.get("n_confirmed", 0)),
                fps=fps_src,
                paused=paused,
                speed=speed,
                draw_mode=draw_mode,
                show_bbox=show_bbox,
                show_trails=show_trails,
            )

        cv2.imshow(win_name, display)

        # Розрахунок delay для waitKey:
        # - у pause: чекаємо вічно (0)
        # - у play: 1000 / fps_src / speed мс
        if paused:
            delay = 0
        else:
            delay = max(1, int(round(1000.0 / fps_src / speed)))
        key = cv2.waitKeyEx(delay)
        if key == -1:
            # таймаут — продовжуємо play
            if not paused:
                if current_idx >= n_frames - 1:
                    paused = True  # дійшли до кінця
                else:
                    current_idx += 1
                    update_trails_step(trails, frames[current_idx],
                                       K, R, tvec, W, H)
            continue

        # --- 1) СПЕЦІАЛЬНІ КЛАВІШІ (стрілки/HOME/END) — спочатку, бо
        #        їх low-byte збігається з ASCII-літерами (LEFT & 0xFF = 'Q'
        #        тощо).
        if key in _KEY_LEFT:
            paused = True
            if current_idx > 0:
                goto(current_idx - 1, reset_trails=True)
            continue
        if key in _KEY_RIGHT:
            paused = True
            if current_idx < n_frames - 1:
                current_idx += 1
                update_trails_step(trails, frames[current_idx],
                                   K, R, tvec, W, H)
            continue
        if key in _KEY_HOME:
            paused = True
            goto(0, reset_trails=True)
            continue
        if key in _KEY_END:
            paused = True
            goto(n_frames - 1, reset_trails=True)
            continue
        if key in _KEY_UP or key in _KEY_DOWN:
            # зарезервовано; ігноруємо
            continue

        # --- 2) ASCII-клавіші — лише якщо key реально в [0, 255].
        if not (0 <= key < 256):
            # невідомий extended-код, ігноруємо
            continue
        ascii_key = key

        # Quit
        if ascii_key in (ord('q'), 27):  # 27 = ESC
            break

        # Pause / Resume
        if ascii_key == ord(' '):
            paused = not paused
            continue

        # Step forward (n)
        if ascii_key == ord('n'):
            paused = True
            if current_idx < n_frames - 1:
                current_idx += 1
                update_trails_step(trails, frames[current_idx],
                                   K, R, tvec, W, H)
            continue

        # Step backward (p)
        if ascii_key == ord('p'):
            paused = True
            if current_idx > 0:
                goto(current_idx - 1, reset_trails=True)
            continue

        # Skip 30 forward (l)
        if ascii_key == ord('l'):
            paused = True
            goto(current_idx + 30, reset_trails=True)
            continue

        # Skip 30 backward (j)
        if ascii_key == ord('j'):
            paused = True
            goto(current_idx - 30, reset_trails=True)
            continue

        # Speed up / down
        if ascii_key in (ord('+'), ord('=')):
            idx = (speed_levels.index(speed)
                   if speed in speed_levels else 2)
            speed = speed_levels[min(idx + 1, len(speed_levels) - 1)]
            continue
        if ascii_key in (ord('-'), ord('_')):
            idx = (speed_levels.index(speed)
                   if speed in speed_levels else 2)
            speed = speed_levels[max(idx - 1, 0)]
            continue

        # Reset playback
        if ascii_key == ord('r'):
            paused = True
            goto(0, reset_trails=True)
            continue

        # Toggles
        if ascii_key == ord('h'):
            show_hud = not show_hud
            continue
        if ascii_key == ord('t'):
            show_trails = not show_trails
            continue
        if ascii_key == ord('b'):
            show_bbox = not show_bbox
            continue
        if ascii_key == ord('1'):
            draw_mode = "all"
            continue
        if ascii_key == ord('2'):
            draw_mode = "confirmed_only"
            continue
        if ascii_key == ord('3'):
            draw_mode = "dominant"
            continue

        # Snapshot
        if ascii_key == ord('s'):
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snap_path = (snapshot_dir /
                         f"live_view_snap_f{current_idx:05d}.png")
            cv2.imwrite(str(snap_path), display)
            sys.stderr.write(f"[snap] -> {snap_path}\n")
            continue

        # Невідома клавіша — ігноруємо мовчки
        _ = last_step_time

    reader.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
