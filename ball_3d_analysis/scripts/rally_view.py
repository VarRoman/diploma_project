#!/usr/bin/env python3
"""
rally_view.py — інтерактивний переглядач "довіреної єдиної траєкторії".

На відміну від live_view.py (показує усі треки одночасно), цей скрипт:

    1. Постпроцесить JSONL: жадібно склеює підтверджені треки у
       логічні "rallies" (розіграші) — ланцюжки треків, які покривають
       один політ м'яча з оклюзіями та handoff'ами між фрагментами;
    2. Показує лише ОДИН активний трек у кадрі — той, що "несе" м'яч
       у цьому моменті rally;
    3. У HUD виводить повну інформацію: 3D-координати (x, y, z), вектор
       швидкості (|v|, vx, vy, vz), активний IMM-режим (B/H/Bn),
       rally_id, current_track_id, маркер HANDOFF коли модель перемкнулась
       з одного треку на інший у межах того ж rally;
    4. Праворуч від кадру — два sparkline-графіки: висота Y(t) та
       модуль швидкості |v|(t) за останні window_sec секунд.

Сценарій: "якщо ми дійсно довіримо це IMMTracker і всім іншим модулям,
наскільки добре виглядатиме результат?" — побачиш одну плавну траєкторію
з прозорими переходами між фрагментами, або помітиш де модель "губиться".

Алгоритм стичингу (build_rallies):

    Для кожного підтвердженого треку (state=Confirmed принаймні в одному
    кадрі) визначаємо confirmed_first/confirmed_last і last_pos/last_vel.
    Сортуємо треки за confirmed_first. Жадібно проходимо:

        для кожного треку T_new:
            знайди rally R, чий хвіст T_prev задовольняє:
                gap = T_new.confirmed_first - T_prev.confirmed_last
                0 <= gap <= max_gap_frames
                ||extrapolate(T_prev.last_pos, T_prev.last_vel, gap*dt) -
                  T_new.first_pos|| <= max_dist_m
            якщо знайдено — додай T_new в кінець R; інакше — створи новий rally.

    Так склеюються треки де балістична екстраполяція "ловить" початок
    наступного треку, навіть якщо track_id перестрибнув через оклюзію.

Контроли (як у live_view.py + декілька нових):
    SPACE       пауза/play
    n або →     наступний кадр
    p або ←     попередній кадр
    j / l       стрибок −30 / +30 кадрів
    HOME/END    на початок / кінець
    [ / ]       попередній / наступний rally (auto-seek до старту rally)
    +/-         прискорити/уповільнити
    h           toggle HUD
    t           toggle trails
    b           toggle bbox
    g           toggle sparkline-графіки
    s           snapshot PNG поточного кадру (з HUD і графіками)
    q / ESC     вихід

Використання:
    python scripts/rally_view.py \\
        --video ../data/videos/Japan_vs_Poland_ultrashort.mp4 \\
        --jsonl logs/seg_ultrashort_full.jsonl
"""

from __future__ import annotations

import argparse
import colorsys
import math
import sys
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# Реюзимо інфраструктуру з overlay_video.py / live_view.py
from visualize_segment import load_jsonl  # noqa: E402
from overlay_video import (  # noqa: E402
    draw_bbox,
    load_calibration_from_header,
    project_3d_to_pixel,
)
from live_view import FrameReader  # noqa: E402


BGR = Tuple[int, int, int]
_GOLDEN = 0.61803398875


# ======================================================================
# Stitching: треки → rallies
# ======================================================================
@dataclass
class TrackInfo:
    """Зведення по одному треку: де він жив, які мав позиції/швидкості."""
    track_id: int
    first_frame: int = -1
    last_frame: int = -1
    confirmed_first: int = -1
    confirmed_last: int = -1
    # frame_idx -> (x, y, z) у м
    positions: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    # frame_idx -> (vx, vy, vz) у м/с
    velocities: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    # frame_idx -> state ('Tentative' | 'Confirmed' | ...)
    states: Dict[int, str] = field(default_factory=dict)
    # frame_idx -> mu (List[float])
    modes: Dict[int, List[float]] = field(default_factory=dict)
    # frame_idx -> mahalanobis_sq або None
    mahal: Dict[int, Optional[float]] = field(default_factory=dict)
    # frame_idx -> True якщо у цьому кадрі трек був matched з детекцією
    matched_frames: set = field(default_factory=set)


def build_track_infos(frames: List[Dict[str, Any]]) -> Dict[int, TrackInfo]:
    """Збирає TrackInfo для всіх треків, що з'являлись у JSONL."""
    infos: Dict[int, TrackInfo] = {}
    for k, rec in enumerate(frames):
        for tr in rec.get("tracks", []) or []:
            tid = int(tr.get("track_id", -1))
            if tid < 0:
                continue
            x_post = tr.get("x_post") or []
            # 9D state: [x, vx, ax, y, vy, ay, z, vz, az]
            # pos = indices 0, 3, 6; vel = indices 1, 4, 7.
            if len(x_post) < 9:
                continue
            info = infos.get(tid)
            if info is None:
                info = TrackInfo(track_id=tid, first_frame=k)
                infos[tid] = info
            info.last_frame = k
            info.positions[k] = (float(x_post[0]), float(x_post[3]),
                                 float(x_post[6]))
            info.velocities[k] = (float(x_post[1]), float(x_post[4]),
                                  float(x_post[7]))
            state = str(tr.get("state", "?"))
            info.states[k] = state
            if state == "Confirmed":
                if info.confirmed_first < 0:
                    info.confirmed_first = k
                info.confirmed_last = k
            info.modes[k] = list(tr.get("mu") or [])
            mah = tr.get("mahalanobis_sq")
            info.mahal[k] = float(mah) if mah is not None else None
            # Якщо у цьому кадрі є detection, і трек був оновлений — це
            # неточно (run_segment.py не пише per-track matched-flag), але
            # для heuristic'у достатньо: треки, mahalanobis_sq у яких НЕ
            # None у цьому кадрі, отримали measurement.
            if mah is not None:
                info.matched_frames.add(k)
    return infos


def build_rallies(
    track_infos: Dict[int, TrackInfo],
    fps: float,
    *,
    max_gap_frames: int = 100,
    max_dist_m: float = 3.0,
) -> List[List[int]]:
    """
    Жадібно склеює підтверджені треки у rallies.

    :param max_gap_frames: максимальний дозволений розрив між
        confirmed_last попереднього треку і confirmed_first нового.
        100 кадрів @ 50 fps = 2 с — типова оклюзія волейбольного м'яча
        за тілом гравця або поза рамкою кадру.
    :param max_dist_m: максимальна 3D-відстань між
        екстрапольованою позицією попереднього треку (за його останньою
        швидкістю на gap_dt) і першою позицією нового треку. 3 м обрано
        як компроміс: для подачі 25 м/с при gap 0.5 с екстраполяція може
        мати помилку ~1.5 м від нелінійності гравітації; 3 м дає запас.

    :return: список rallies, кожен — chain треків (track_id) у
        порядку часу.
    """
    # Лише підтверджені (мають хоч один Confirmed-кадр).
    confirmed_ids = [
        tid for tid, info in track_infos.items()
        if info.confirmed_first >= 0
    ]
    confirmed_ids.sort(key=lambda t: track_infos[t].confirmed_first)

    rallies: List[List[int]] = []

    for tid in confirmed_ids:
        info = track_infos[tid]
        new_first = info.confirmed_first
        new_pos = info.positions.get(new_first)
        if new_pos is None:
            continue

        best_ri = -1
        best_dist = max_dist_m
        for ri, chain in enumerate(rallies):
            prev_tid = chain[-1]
            prev_info = track_infos[prev_tid]
            gap = new_first - prev_info.confirmed_last
            if gap < 0 or gap > max_gap_frames:
                continue
            last_pos = prev_info.positions.get(prev_info.confirmed_last)
            last_vel = prev_info.velocities.get(prev_info.confirmed_last)
            if last_pos is None or last_vel is None:
                continue
            gap_dt = gap / fps
            extrap = (
                last_pos[0] + last_vel[0] * gap_dt,
                last_pos[1] + last_vel[1] * gap_dt,
                last_pos[2] + last_vel[2] * gap_dt,
            )
            dist = math.sqrt(
                (extrap[0] - new_pos[0]) ** 2
                + (extrap[1] - new_pos[1]) ** 2
                + (extrap[2] - new_pos[2]) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_ri = ri

        if best_ri >= 0:
            rallies[best_ri].append(tid)
        else:
            rallies.append([tid])

    return rallies


def get_active_per_frame(
    n_frames: int,
    track_infos: Dict[int, TrackInfo],
    rallies: List[List[int]],
) -> List[Optional[Tuple[int, int]]]:
    """
    Для кожного кадру визначає (rally_id, active_track_id) — який трек
    "несе" rally у цьому кадрі. Якщо у кадрі активних треків rally нема —
    повертає None. Якщо активних кілька rally — вибирає той rally, що
    має найдовший chain (heuristic: "довіряємо тому, в кого більше
    спостережень").

    :param n_frames: к-сть кадрів у JSONL (= len(frames)).
    :param track_infos: словник з build_track_infos.
    :param rallies: список ланцюгів з build_rallies.
    """
    # Для швидкого lookup'у: track_id → rally_id
    rid_of_tid: Dict[int, int] = {}
    for ri, chain in enumerate(rallies):
        for tid in chain:
            rid_of_tid[tid] = ri

    # Для кожного кадру — list[(rally_id, track_id)] з активних
    active: List[Optional[Tuple[int, int]]] = [None] * n_frames

    for k in range(n_frames):
        candidates: List[Tuple[int, int]] = []
        for tid, info in track_infos.items():
            if info.confirmed_first < 0:
                continue
            if not (info.confirmed_first <= k <= info.confirmed_last):
                continue
            if k not in info.positions:
                continue
            rid = rid_of_tid.get(tid)
            if rid is None:
                continue
            candidates.append((rid, tid))

        if not candidates:
            continue
        if len(candidates) == 1:
            active[k] = candidates[0]
            continue

        # Кілька кандидатів — вибираємо rally з найдовшим chain'ом
        # (за к-стю треків); якщо нічия — той rally, у якого активний
        # трек має більше matched_frames.
        def score(c: Tuple[int, int]) -> Tuple[int, int]:
            rid, tid = c
            return (len(rallies[rid]), len(track_infos[tid].matched_frames))

        active[k] = max(candidates, key=score)

    return active


# ======================================================================
# Кольори по rally_id
# ======================================================================
def rally_color(rid: int) -> BGR:
    """Стабільний HSV-золотий-кут колір по rally_id у BGR."""
    h = (rid * _GOLDEN) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


# ======================================================================
# Sparkline-графіки
# ======================================================================
def _normalize_window(series: List[Tuple[float, float]],
                      now_t: float,
                      window_sec: float,
                      ) -> List[Tuple[float, float]]:
    """Залишає лише точки [(t, y)] із t ∈ [now_t-window_sec, now_t]."""
    if not series:
        return []
    cutoff = now_t - window_sec
    return [(t, y) for (t, y) in series if t >= cutoff]


def draw_sparkline(
    img: np.ndarray,
    *,
    series: List[Tuple[float, float]],
    title: str,
    x0: int,
    y0: int,
    w: int,
    h: int,
    color: BGR = (200, 220, 255),
    y_label_fmt: str = "{:.2f}",
) -> None:
    """
    Малює маленький лінійний графік: послідовність (t, y) із series
    у box'і [(x0,y0), (x0+w, y0+h)]. Авто-нормалізація y у вікно.
    Title — невеликий рядок зверху box'а.
    """
    # Бекдроп
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0.0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (90, 90, 90), 1)

    cv2.putText(img, title, (x0 + 6, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
                cv2.LINE_AA)

    if len(series) < 2:
        cv2.putText(img, "(no data)", (x0 + 6, y0 + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1,
                    cv2.LINE_AA)
        return

    ts = [p[0] for p in series]
    ys = [p[1] for p in series]
    t_min, t_max = ts[0], ts[-1]
    y_min, y_max = min(ys), max(ys)
    if t_max - t_min < 1e-6 or y_max - y_min < 1e-6:
        # Виродженість — намалюємо горизонталь
        yc = y0 + h // 2
        cv2.line(img, (x0 + 5, yc), (x0 + w - 5, yc), color, 1, cv2.LINE_AA)
        return

    # Маржі усередині box'а
    pad_left = 26
    pad_right = 6
    pad_top = 22
    pad_bottom = 14
    inner_x0 = x0 + pad_left
    inner_y0 = y0 + pad_top
    inner_w = w - pad_left - pad_right
    inner_h = h - pad_top - pad_bottom

    pts: List[Tuple[int, int]] = []
    for (t, y) in series:
        nx = (t - t_min) / (t_max - t_min)
        ny = (y - y_min) / (y_max - y_min)
        px = int(inner_x0 + nx * inner_w)
        py = int(inner_y0 + (1.0 - ny) * inner_h)
        pts.append((px, py))

    # Полилин
    if len(pts) >= 2:
        cv2.polylines(img, [np.asarray(pts, dtype=np.int32)],
                      isClosed=False, color=color, thickness=1,
                      lineType=cv2.LINE_AA)
    # Поточна (остання) точка — маркер
    cv2.circle(img, pts[-1], 3, color, -1, cv2.LINE_AA)

    # Підписи: min, max y
    cv2.putText(img, y_label_fmt.format(y_max),
                (x0 + 2, inner_y0 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1,
                cv2.LINE_AA)
    cv2.putText(img, y_label_fmt.format(y_min),
                (x0 + 2, inner_y0 + inner_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1,
                cv2.LINE_AA)


# ======================================================================
# HUD
# ======================================================================
def _mode_label(mu: List[float]) -> str:
    if not mu:
        return "?"
    idx = int(np.argmax(mu))
    return ["B", "H", "Bn"][idx] if idx < 3 else "?"


def draw_rally_hud(
    img: np.ndarray,
    *,
    current_idx: int,
    n_total: int,
    t_sec: float,
    rally_id: Optional[int],
    active_tid: Optional[int],
    prev_tid: Optional[int],
    handoff_active: bool,
    track_rec: Optional[Dict[str, Any]],
    n_rallies: int,
    paused: bool,
    speed: float,
    show_trails: bool,
    show_bbox: bool,
) -> None:
    """HUD ліворуч-зверху, фокус на rally + 3D + швидкість."""
    pad = 8
    w = 380
    h = 220
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0.0, dst=img)

    # Play/Pause + progress bar
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

    y = pad + 50
    line_frame = (f"frame {current_idx + 1}/{n_total}  "
                  f"t={t_sec:6.2f}s  x{speed:g}")
    cv2.putText(img, line_frame, (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    y += 22

    if rally_id is None:
        cv2.putText(img, "rally: --  (no active confirmed track)",
                    (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1, cv2.LINE_AA)
        y += 22
    else:
        rc = rally_color(rally_id)
        rally_str = (f"rally R{rally_id}  ({n_rallies} total)  "
                     f"current T{active_tid}")
        cv2.putText(img, rally_str, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2, cv2.LINE_AA)
        y += 22

        if handoff_active and prev_tid is not None:
            handoff_str = f"HANDOFF: T{prev_tid} -> T{active_tid}"
            cv2.putText(img, handoff_str, (pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 255), 2, cv2.LINE_AA)
            y += 22

    # 3D + швидкість
    if track_rec is not None:
        x_post = track_rec.get("x_post") or []
        # 9D state: pos = 0, 3, 6; vel = 1, 4, 7. Accel (2, 5, 8) ігноруємо в HUD.
        if len(x_post) >= 9:
            x, vx, yy, vy, z, vz = (float(x_post[0]), float(x_post[1]),
                                    float(x_post[3]), float(x_post[4]),
                                    float(x_post[6]), float(x_post[7]))
            v_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
            pos_str = f"pos: x={x:+6.2f}m  y={yy:+5.2f}m  z={z:+6.2f}m"
            vel_str = (f"v:   |v|={v_norm:5.2f} m/s "
                       f"(vx={vx:+5.2f} vy={vy:+5.2f} vz={vz:+5.2f})")
            cv2.putText(img, pos_str, (pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (220, 220, 220), 1, cv2.LINE_AA)
            y += 20
            cv2.putText(img, vel_str, (pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (220, 220, 220), 1, cv2.LINE_AA)
            y += 20

            mu = track_rec.get("mu") or []
            mode = _mode_label(mu)
            mu_str = f"mode mu={mode}  ({','.join(f'{m:.2f}' for m in mu)})"
            cv2.putText(img, mu_str, (pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
            y += 20

            mah = track_rec.get("mahalanobis_sq")
            state = track_rec.get("state", "?")
            extra = f"state={state}"
            if mah is not None:
                extra += f"  mah²={float(mah):.2f}"
            cv2.putText(img, extra, (pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
            y += 20

    y = h - 8
    cv2.putText(img,
                "[ / ] prev/next rally   t/b/g toggles   s snap   q quit",
                (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (160, 160, 160), 1, cv2.LINE_AA)


# ======================================================================
# Малювання активного треку + хвоста
# ======================================================================
def draw_active_track(
    img: np.ndarray,
    rec: Dict[str, Any],
    active_tid: int,
    rally_id: int,
    trails: Dict[int, Deque[Tuple[int, int]]],
    K: np.ndarray, R: np.ndarray, tvec: np.ndarray, W: int, H: int,
    *,
    show_trails: bool,
    show_bbox: bool,
) -> Optional[Tuple[int, int]]:
    """
    Малює один active track цього rally: bbox, trail, точку, лейбл.
    Повертає (u, v) точки треку якщо вдалось спроектувати, інакше None.
    """
    color = rally_color(rally_id)

    if show_bbox:
        rd = rec.get("raw_detection") or {}
        if rd.get("detected"):
            u = rd.get("u")
            v = rd.get("v")
            w_box = rd.get("w_box")
            if u is not None and v is not None and w_box:
                draw_bbox(img, float(u), float(v), float(w_box),
                          color=(0, 255, 0))

    # Шукаємо активний трек у tracks
    target_tr = None
    for tr in rec.get("tracks", []) or []:
        if int(tr.get("track_id", -1)) == active_tid:
            target_tr = tr
            break
    if target_tr is None:
        return None

    x_post = target_tr.get("x_post") or []
    # 9D state: pos = indices 0, 3, 6.
    if len(x_post) < 9:
        return None
    P3d = np.array([x_post[0], x_post[3], x_post[6]], dtype=np.float64)
    uv = project_3d_to_pixel(P3d, R, tvec, K, W, H)
    if uv is None:
        return None

    trails[rally_id].append(uv)

    if show_trails and len(trails[rally_id]) >= 2:
        pts = list(trails[rally_id])
        # Прозорий polyline у градієнті — простий варіант, без per-segment
        overlay = img.copy()
        cv2.polylines(overlay, [np.asarray(pts, dtype=np.int32)],
                      isClosed=False, color=color, thickness=2,
                      lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0.0, dst=img)

    # Точка трека
    cv2.circle(img, uv, 7, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(img, uv, 7, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return uv


# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Інтерактивний переглядач rally-стічених треків.",
    )
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--jsonl", required=True, type=Path)
    p.add_argument("--trail_len", type=int, default=120,
                   help="К-сть кадрів у trail (default 120 ≈ 2.4 с "
                        "@ 50 fps — для rally доцільно бачити довшу "
                        "історію, ніж у live_view.py).")
    p.add_argument("--max_gap_frames", type=int, default=100,
                   help="Максимальний gap (кадри) між треками для "
                        "стічингу в один rally (default 100 ≈ 2 с).")
    p.add_argument("--max_dist_m", type=float, default=3.0,
                   help="Максимальна 3D-відстань (м) між балістичною "
                        "екстраполяцією попереднього треку і початком "
                        "наступного для стічингу (default 3.0).")
    p.add_argument("--sparkline_window_sec", type=float, default=4.0,
                   help="Скільки секунд історії показувати у sparkline-"
                        "графіках Y(t), |v|(t) (default 4.0).")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--window_height", type=int, default=720)
    p.add_argument("--cache_size", type=int, default=240)
    p.add_argument("--snapshot_dir", type=Path, default=None)
    p.add_argument("--export_rallies", type=Path, default=None,
                   help="Якщо вказано — після постпроцесингу записує "
                        "JSON-зведення rally (списки track_id, інтервали "
                        "кадрів, тривалість) у вказаний файл і виходить.")
    return p.parse_args()


# Спецключі (з live_view.py)
_KEY_LEFT = {65361, 2424832}
_KEY_RIGHT = {65363, 2555904}
_KEY_HOME = {65360, 2359296}
_KEY_END = {65367, 2293760}
_KEY_UP = {65362, 2490368}
_KEY_DOWN = {65364, 2621440}


# ======================================================================
# main
# ======================================================================
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

    fps = float(header["fps"])
    frame_range = header["frame_range"]
    f_start = int(frame_range[0])
    frame_size = header["frame_size"]
    W, H = int(frame_size[0]), int(frame_size[1])
    n_frames = len(frames)
    if n_frames == 0:
        sys.stderr.write("[ERR] JSONL не містить кадрів.\n")
        return 2

    # ---- Stitching ----
    sys.stderr.write("[i] post-process: збираю TrackInfo...\n")
    track_infos = build_track_infos(frames)
    sys.stderr.write(
        f"[i] track_infos: {len(track_infos)} треків "
        f"({sum(1 for i in track_infos.values() if i.confirmed_first >= 0)} "
        f"підтверджених)\n"
    )

    rallies = build_rallies(
        track_infos, fps,
        max_gap_frames=args.max_gap_frames,
        max_dist_m=args.max_dist_m,
    )
    sys.stderr.write(
        f"[i] rallies: {len(rallies)} "
        f"(середній chain={sum(len(c) for c in rallies)/max(len(rallies),1):.1f})\n"
    )
    for ri, chain in enumerate(rallies):
        first_k = min(track_infos[t].confirmed_first for t in chain)
        last_k = max(track_infos[t].confirmed_last for t in chain)
        duration_s = (last_k - first_k + 1) / fps
        sys.stderr.write(
            f"    R{ri}: tracks={chain}  frames=[{first_k},{last_k}]  "
            f"≈{duration_s:.2f}s\n"
        )

    # ---- Export only (skip live view) ----
    if args.export_rallies is not None:
        import json
        export = {
            "video": str(args.video),
            "jsonl": str(args.jsonl),
            "fps": fps,
            "n_frames": n_frames,
            "n_tracks_total": len(track_infos),
            "n_confirmed_tracks": sum(
                1 for i in track_infos.values() if i.confirmed_first >= 0
            ),
            "n_rallies": len(rallies),
            "stitch_params": {
                "max_gap_frames": args.max_gap_frames,
                "max_dist_m": args.max_dist_m,
            },
            "rallies": [
                {
                    "rally_id": ri,
                    "track_ids": chain,
                    "first_frame": min(track_infos[t].confirmed_first
                                       for t in chain),
                    "last_frame": max(track_infos[t].confirmed_last
                                      for t in chain),
                    "duration_sec": (
                        max(track_infos[t].confirmed_last for t in chain)
                        - min(track_infos[t].confirmed_first for t in chain)
                        + 1
                    ) / fps,
                }
                for ri, chain in enumerate(rallies)
            ],
        }
        args.export_rallies.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_rallies, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        sys.stderr.write(f"[ok] export -> {args.export_rallies}\n")
        return 0

    active_per_frame = get_active_per_frame(n_frames, track_infos, rallies)

    # ---- Live view ----
    reader = FrameReader(args.video, f_start, n_frames,
                         cache_size=args.cache_size)

    current_idx = max(0, min(n_frames - 1, args.start_frame))
    paused = True
    speed = 1.0
    speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    show_hud = True
    show_trails = True
    show_bbox = True
    show_graphs = True
    trail_len = args.trail_len

    # Trails per rally
    trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=trail_len)
    )

    # Серії для sparkline (rally-локальні; перебудовуємо при стрибках)
    sparkline_y: Deque[Tuple[float, float]] = deque(maxlen=2000)
    sparkline_v: Deque[Tuple[float, float]] = deque(maxlen=2000)

    snapshot_dir = args.snapshot_dir or args.jsonl.parent

    win_name = "rally_view: stitched single trajectory"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    target_h = int(args.window_height)
    target_w = int(round(W * target_h / H))
    cv2.resizeWindow(win_name, target_w, target_h)

    sys.stderr.write(
        f"[i] rally_view: {n_frames} frames @ {fps:.1f} fps  {W}x{H}\n"
        f"[i] контроли: SPACE/n/p/j/l/[/]/+/-/h/t/b/g/s/q\n"
    )

    def rebuild_sparklines(end_idx: int) -> None:
        """Перебудовує sparkline-серії з frames[0..end_idx] для активних
        кадрів того ж rally, що й end_idx (якщо є активний)."""
        sparkline_y.clear()
        sparkline_v.clear()
        current_rid = (active_per_frame[end_idx][0]
                       if active_per_frame[end_idx] is not None else None)
        if current_rid is None:
            return
        window_frames = int(args.sparkline_window_sec * fps)
        start = max(0, end_idx - window_frames + 1)
        for k in range(start, end_idx + 1):
            ap = active_per_frame[k]
            if ap is None or ap[0] != current_rid:
                continue
            tid = ap[1]
            info = track_infos[tid]
            pos = info.positions.get(k)
            vel = info.velocities.get(k)
            if pos is None or vel is None:
                continue
            t = k / fps
            sparkline_y.append((t, pos[1]))   # y = висота
            v_norm = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
            sparkline_v.append((t, v_norm))

    def rebuild_trails(end_idx: int) -> None:
        """Перебудовує trails для активного rally."""
        trails.clear()
        ap_now = active_per_frame[end_idx]
        if ap_now is None:
            return
        current_rid = ap_now[0]
        start = max(0, end_idx - trail_len + 1)
        for k in range(start, end_idx + 1):
            ap = active_per_frame[k]
            if ap is None or ap[0] != current_rid:
                continue
            tid = ap[1]
            info = track_infos[tid]
            pos = info.positions.get(k)
            if pos is None:
                continue
            P3d = np.array([pos[0], pos[1], pos[2]], dtype=np.float64)
            uv = project_3d_to_pixel(P3d, R, tvec, K, W, H)
            if uv is None:
                continue
            trails[current_rid].append(uv)

    rebuild_sparklines(current_idx)
    rebuild_trails(current_idx)

    def goto(new_idx: int) -> None:
        nonlocal current_idx
        new_idx = max(0, min(n_frames - 1, new_idx))
        if new_idx == current_idx:
            return
        current_idx = new_idx
        rebuild_sparklines(current_idx)
        rebuild_trails(current_idx)

    def step_forward() -> None:
        nonlocal current_idx
        if current_idx >= n_frames - 1:
            return
        current_idx += 1
        ap_now = active_per_frame[current_idx]
        ap_prev = active_per_frame[current_idx - 1]
        # Якщо rally змінився — перебудовуємо trails та sparklines.
        if (ap_now is None) != (ap_prev is None) or (
            ap_now is not None and ap_prev is not None
            and ap_now[0] != ap_prev[0]
        ):
            rebuild_trails(current_idx)
            rebuild_sparklines(current_idx)
            return
        # Інакше — інкрементально оновлюємо.
        if ap_now is not None:
            tid = ap_now[1]
            info = track_infos[tid]
            pos = info.positions.get(current_idx)
            vel = info.velocities.get(current_idx)
            if pos is not None:
                P3d = np.array([pos[0], pos[1], pos[2]], dtype=np.float64)
                uv = project_3d_to_pixel(P3d, R, tvec, K, W, H)
                if uv is not None:
                    trails[ap_now[0]].append(uv)
                t = current_idx / fps
                sparkline_y.append((t, pos[1]))
                if vel is not None:
                    v_norm = math.sqrt(vel[0] ** 2 + vel[1] ** 2
                                       + vel[2] ** 2)
                    sparkline_v.append((t, v_norm))
                # Тримаємо вікно sparkline:
                window_t = args.sparkline_window_sec
                cutoff = t - window_t
                while sparkline_y and sparkline_y[0][0] < cutoff:
                    sparkline_y.popleft()
                while sparkline_v and sparkline_v[0][0] < cutoff:
                    sparkline_v.popleft()

    def next_rally() -> None:
        """Стрибок на старт наступного rally (за confirmed_first)."""
        starts = sorted(
            min(track_infos[t].confirmed_first for t in chain)
            for chain in rallies
        )
        for s in starts:
            if s > current_idx:
                goto(s)
                return
        # Немає — нічого не робимо
        sys.stderr.write("[i] немає rally після поточної позиції\n")

    def prev_rally() -> None:
        starts = sorted(
            min(track_infos[t].confirmed_first for t in chain)
            for chain in rallies
        )
        for s in reversed(starts):
            if s < current_idx:
                goto(s)
                return
        sys.stderr.write("[i] немає rally перед поточною позицією\n")

    # ---- Main loop ----
    while True:
        rec = frames[current_idx]
        img = reader.get(current_idx)
        if img is None:
            sys.stderr.write(
                f"[w] cap.read() повернув None для idx={current_idx}\n"
            )
            break

        display = img.copy()

        ap = active_per_frame[current_idx]
        ap_prev = (active_per_frame[current_idx - 1]
                   if current_idx > 0 else None)

        rally_id: Optional[int] = None
        active_tid: Optional[int] = None
        prev_tid: Optional[int] = None
        handoff_active = False
        track_rec_for_hud: Optional[Dict[str, Any]] = None

        if ap is not None:
            rally_id, active_tid = ap
            if ap_prev is not None and ap_prev[0] == rally_id \
               and ap_prev[1] != active_tid:
                handoff_active = True
                prev_tid = ap_prev[1]
            for tr in rec.get("tracks", []) or []:
                if int(tr.get("track_id", -1)) == active_tid:
                    track_rec_for_hud = tr
                    break

            draw_active_track(
                display, rec, active_tid, rally_id, trails,
                K, R, tvec, W, H,
                show_trails=show_trails, show_bbox=show_bbox,
            )

        # Sparkline-графіки праворуч-зверху
        if show_graphs:
            spark_w = 320
            spark_h = 110
            spark_x0 = W - spark_w - 12
            spark_y0 = 12
            draw_sparkline(display,
                           series=list(sparkline_y),
                           title=f"height Y(t) [m]  (window "
                                 f"{args.sparkline_window_sec:.1f}s)",
                           x0=spark_x0, y0=spark_y0,
                           w=spark_w, h=spark_h,
                           color=(120, 220, 120),
                           y_label_fmt="{:+.2f}")
            draw_sparkline(display,
                           series=list(sparkline_v),
                           title="speed |v|(t) [m/s]",
                           x0=spark_x0, y0=spark_y0 + spark_h + 8,
                           w=spark_w, h=spark_h,
                           color=(120, 180, 255),
                           y_label_fmt="{:5.2f}")

        if show_hud:
            draw_rally_hud(
                display,
                current_idx=current_idx,
                n_total=n_frames,
                t_sec=float(rec.get("t", current_idx / fps)),
                rally_id=rally_id,
                active_tid=active_tid,
                prev_tid=prev_tid,
                handoff_active=handoff_active,
                track_rec=track_rec_for_hud,
                n_rallies=len(rallies),
                paused=paused,
                speed=speed,
                show_trails=show_trails,
                show_bbox=show_bbox,
            )

        cv2.imshow(win_name, display)

        if paused:
            delay = 0
        else:
            delay = max(1, int(round(1000.0 / fps / speed)))
        key = cv2.waitKeyEx(delay)

        if key == -1:
            if not paused:
                if current_idx >= n_frames - 1:
                    paused = True
                else:
                    step_forward()
            continue

        # Спецключі
        if key in _KEY_LEFT:
            paused = True
            if current_idx > 0:
                goto(current_idx - 1)
            continue
        if key in _KEY_RIGHT:
            paused = True
            if current_idx < n_frames - 1:
                step_forward()
            continue
        if key in _KEY_HOME:
            paused = True
            goto(0)
            continue
        if key in _KEY_END:
            paused = True
            goto(n_frames - 1)
            continue
        if key in _KEY_UP or key in _KEY_DOWN:
            continue

        if not (0 <= key < 256):
            continue
        ascii_key = key

        if ascii_key in (ord('q'), 27):
            break
        if ascii_key == ord(' '):
            paused = not paused
            continue
        if ascii_key == ord('n'):
            paused = True
            step_forward()
            continue
        if ascii_key == ord('p'):
            paused = True
            if current_idx > 0:
                goto(current_idx - 1)
            continue
        if ascii_key == ord('l'):
            paused = True
            goto(current_idx + 30)
            continue
        if ascii_key == ord('j'):
            paused = True
            goto(current_idx - 30)
            continue
        if ascii_key == ord('['):
            paused = True
            prev_rally()
            continue
        if ascii_key == ord(']'):
            paused = True
            next_rally()
            continue
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
        if ascii_key == ord('r'):
            paused = True
            goto(0)
            continue
        if ascii_key == ord('h'):
            show_hud = not show_hud
            continue
        if ascii_key == ord('t'):
            show_trails = not show_trails
            continue
        if ascii_key == ord('b'):
            show_bbox = not show_bbox
            continue
        if ascii_key == ord('g'):
            show_graphs = not show_graphs
            continue
        if ascii_key == ord('s'):
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snap_path = (snapshot_dir /
                         f"rally_view_snap_f{current_idx:05d}.png")
            cv2.imwrite(str(snap_path), display)
            sys.stderr.write(f"[snap] -> {snap_path}\n")
            continue

    reader.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
