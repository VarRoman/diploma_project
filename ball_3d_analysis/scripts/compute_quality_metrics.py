#!/usr/bin/env python3
"""
compute_quality_metrics.py — кількісні метрики якості роботи
IMMTracker'а на одному сегменті відео (за виходом run_segment.py).

Призначення:
    Дати числові показники, які можна напряму вкласти у дипломну
    («Розділ 3 — Експериментальна перевірка»). Усі метрики
    обчислюються над одним JSONL-логом і пишуться у:
        — JSON-файл (опційно --out_json),
        — людино-читабельну таблицю у stdout.

Метрики:
    1. Tracking coverage:
       coverage = #frames-with-≥1-track / #frames-with-detection.
       Чим вище — тим менше пропусків (без урахування ID-стабільності).

    2. Track fragmentation:
       fragmentation = n_tracks_total / n_detection_runs, де
       n_detection_runs — це к-сть «безперервних суцільних» проміжків,
       у яких на кожному кадрі є детекція з вікном <= max_gap.
       Ідеал = 1.0 (один трек на одну «послідовність м'яча»).

    3. ID switches (IDSW):
       Евристика: трек A зник за кадр t_A, у межах [t_A+1, t_A+max_gap]
       з'являється новий трек B, перша 3D-позиція якого знаходиться
       у межах max_dist (м) від останньої 3D-позиції A. Тоді
       вважаємо, що сталося перемикання ID (можливо false positive,
       але оцінка корисна для grid-search).

    4. Smoothness (per track):
       jerk = середнє ||a_t - a_{t-1}|| / dt, де a_t = (v_t - v_{t-1})/dt.
       Чим менше — тим плавніша оцінка. Окремо для x_post[1,3,5]
       (vx, vy, vz). Агрегується як median по треках із hits >= min_hits.

    5. Mode-switching rate:
       Скільки разів argmax(mu) змінюється на секунду життя треку.
       Високі значення = трекер «нервує» і часто перемикає модель.

    6. RMSE 3D self-consistency:
       Для кадрів з детекцією: ||raw_3d - x_post[0,2,4]|| у метрах.
       Не є ground-truth-помилкою, а саме «розбіжність із власним
       виміром». Великі значення можуть вказувати на надмірну
       екстраполяцію або поганий gating.

    7. Mahalanobis@gate stats (matched only):
       p50, p90, p99, % above gate (для tracks з time_since_update=0).

Приклад:
    python scripts/compute_quality_metrics.py \\
        --jsonl logs/seg_ultrashort_full.jsonl \\
        --out_json logs/seg_ultrashort_full_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Re-use parser from visualize_segment.py — щоб формат JSONL був
# у одному місці.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from visualize_segment import load_jsonl  # noqa: E402


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Метрики якості IMM-трекера за виходом run_segment.py."
    )
    p.add_argument("--jsonl", required=True, type=Path,
                   help="JSONL з діагностикою (вихід run_segment.py).")
    p.add_argument("--out_json", type=Path, default=None,
                   help="Куди писати JSON із усіма метриками.")
    p.add_argument("--min_hits", type=int, default=3,
                   help="Для метрик 'на трек' беремо лише треки, "
                        "що отримали хоч би стільки hits (за замовч. 3).")
    p.add_argument("--idsw_max_gap", type=int, default=3,
                   help="Макс. розрив у кадрах між зникненням треку A "
                        "та появою треку B для зарахування IDSW (3).")
    p.add_argument("--idsw_max_dist", type=float, default=2.0,
                   help="Макс. 3D-відстань (м) між кінцем A та "
                        "початком B для зарахування IDSW (2.0).")
    p.add_argument("--detection_run_max_gap", type=int, default=2,
                   help="Для підрахунку n_detection_runs: проміжок "
                        "<= цього к-ва кадрів вважаємо продовженням "
                        "тієї ж послідовності (2).")
    p.add_argument("--gating_sq", type=float, default=11.34,
                   help="Поріг Mahalanobis^2 для статистики "
                        "matched-above-gate (default 11.34 = χ² df=3 p=0.99).")
    p.add_argument("--quiet", action="store_true",
                   help="Не друкувати таблицю у stdout.")
    return p.parse_args()


# ----------------------------------------------------------------------
# Допоміжне
# ----------------------------------------------------------------------
def _safe_median(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.mean(values))


def _safe_percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values), q))


def _norm3(v: List[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


# ----------------------------------------------------------------------
# Збір сирих даних по треках
# ----------------------------------------------------------------------
class TrackData:
    """Усі точки одного треку для подальшого обчислення метрик."""

    def __init__(self) -> None:
        self.frames: List[int] = []
        # x_post у точку часу: 9D-вектори [x, vx, ax, y, vy, ay, z, vz, az]
        self.x_post: List[List[float]] = []
        # raw_3d на тому ж кадрі (None, якщо не було детекції)
        self.raw_3d: List[Optional[List[float]]] = []
        # argmax(mu) у точку часу
        self.mode: List[int] = []
        # mahalanobis_sq у точку часу (None, якщо tsu>0 чи нема)
        self.mahal: List[Optional[float]] = []
        self.time_since_update: List[int] = []
        self.states: List[str] = []


def collect_tracks(frames: List[Dict[str, Any]]) -> Dict[int, TrackData]:
    """Просканувати JSONL-frames та зібрати TrackData по track_id."""
    data: Dict[int, TrackData] = defaultdict(TrackData)
    for fr in frames:
        f_idx = int(fr.get("frame_0based", fr.get("frame", 0)))
        raw3d = fr.get("raw_3d")
        for tr in fr.get("tracks", []) or []:
            tid = int(tr["track_id"])
            d = data[tid]
            d.frames.append(f_idx)
            d.x_post.append(list(tr["x_post"]))
            d.raw_3d.append(list(raw3d) if raw3d is not None else None)
            mu = tr.get("mu") or [1.0, 0.0, 0.0]
            d.mode.append(int(np.argmax(mu)))
            d.mahal.append(tr.get("mahalanobis_sq"))
            d.time_since_update.append(int(tr.get("time_since_update", 0)))
            d.states.append(str(tr.get("state", "?")))
    return data


# ----------------------------------------------------------------------
# Метрики
# ----------------------------------------------------------------------
def compute_coverage(frames: List[Dict[str, Any]]) -> Dict[str, float]:
    """Скільки кадрів-із-детекцією покриті хоча б одним треком."""
    n_det = 0
    n_det_with_track = 0
    n_total = len(frames)
    n_track_frames = 0
    for fr in frames:
        rd = fr.get("raw_detection") or {}
        if rd.get("detected"):
            n_det += 1
            if (fr.get("n_tracks") or 0) > 0:
                n_det_with_track += 1
        if (fr.get("n_tracks") or 0) > 0:
            n_track_frames += 1
    return {
        "n_total_frames": n_total,
        "n_frames_with_detection": n_det,
        "n_frames_with_track": n_track_frames,
        "coverage_detected_to_tracked":
            n_det_with_track / n_det if n_det else float("nan"),
        "coverage_total_frames":
            n_track_frames / n_total if n_total else float("nan"),
    }


def compute_fragmentation(
    frames: List[Dict[str, Any]],
    tracks: Dict[int, TrackData],
    max_gap: int,
) -> Dict[str, float]:
    """
    fragmentation = n_tracks_confirmed / n_detection_runs.
    n_detection_runs — група детекцій, де між сусідніми <= max_gap кадрів.

    Рахуємо лише треки, що БУЛИ Confirmed — це реальні шматки траєкторії.
    Tentative-спавни (hits 1-2, відкинуті як шум/фантоми) НЕ є фрагментами
    траєкторії; їх включення штучно роздувало б метрику (особливо при
    gate=16 + single-ball NMS, де маневр породжує короткі Tentative-
    конкуренти). n_tracks_total лишаємо для довідки.
    """
    det_frames = [
        int(fr.get("frame_0based", fr.get("frame", 0)))
        for fr in frames if (fr.get("raw_detection") or {}).get("detected")
    ]
    det_frames.sort()
    n_confirmed = sum(
        1 for d in tracks.values() if "Confirmed" in set(d.states)
    )
    if not det_frames:
        return {
            "n_tracks_total": len(tracks),
            "n_tracks_confirmed": n_confirmed,
            "n_detection_runs": 0,
            "fragmentation": float("nan"),
        }
    n_runs = 1
    for i in range(1, len(det_frames)):
        if det_frames[i] - det_frames[i - 1] > max_gap:
            n_runs += 1
    return {
        "n_tracks_total": len(tracks),
        "n_tracks_confirmed": n_confirmed,
        "n_detection_runs": n_runs,
        "fragmentation": n_confirmed / n_runs,
    }


def compute_idsw(
    tracks: Dict[int, TrackData],
    max_gap: int,
    max_dist: float,
) -> Dict[str, Any]:
    """
    Евристично оцінюємо к-сть ID-перемикань: для кожного треку A
    знаходимо «трек-кандидат» B, що з'явився протягом max_gap кадрів
    після кінця A і чий перший x_post знаходиться у max_dist від
    останнього x_post треку A.
    """
    # Список (track_id, first_frame, last_frame, first_xyz, last_xyz)
    summaries: List[Tuple[int, int, int, np.ndarray, np.ndarray]] = []
    for tid, d in tracks.items():
        if not d.frames:
            continue
        first = d.frames[0]
        last = d.frames[-1]
        # 9D state: pos = indices 0, 3, 6.
        first_xyz = np.array(
            [d.x_post[0][0], d.x_post[0][3], d.x_post[0][6]]
        )
        last_xyz = np.array(
            [d.x_post[-1][0], d.x_post[-1][3], d.x_post[-1][6]]
        )
        summaries.append((tid, first, last, first_xyz, last_xyz))

    n_idsw = 0
    pairs: List[Tuple[int, int, int, float]] = []  # (A, B, gap, dist)
    for (tid_a, _f0a, last_a, _, last_xyz) in summaries:
        for (tid_b, first_b, _last_b, first_xyz, _) in summaries:
            if tid_b == tid_a:
                continue
            gap = first_b - last_a
            if 1 <= gap <= max_gap:
                dist = float(np.linalg.norm(first_xyz - last_xyz))
                if dist <= max_dist:
                    n_idsw += 1
                    pairs.append((tid_a, tid_b, int(gap), dist))
                    break  # лише один B-кандидат на A
    return {
        "n_idsw_estimated": n_idsw,
        "idsw_pairs_sample": pairs[:10],
    }


def compute_smoothness(
    tracks: Dict[int, TrackData],
    dt: float,
    min_hits: int,
) -> Dict[str, float]:
    """
    jerk = mean ||a_t - a_{t-1}|| / dt, де a_t = (v_t - v_{t-1}) / dt.
    Беремо швидкості з x_post (9D state, компоненти 1, 4, 7). Агрегуємо
    як median по треках.

    Зауваження: з переходом на 9D у x_post є вже власна оцінка accel
    (індекси 2, 5, 8) — її можна використовувати як прямий jerk-proxy,
    але для backward-сумісності метрик лишаємо finite-difference з v.
    """
    jerk_per_track: List[float] = []
    accel_per_track: List[float] = []
    for tid, d in tracks.items():
        if len(d.x_post) < 4 or len(d.frames) < 4:
            continue
        if len(d.x_post) < min_hits:
            continue
        # Витягуємо v_t — 9D state: vel indices 1, 4, 7.
        v = np.array(
            [[p[1], p[4], p[7]] for p in d.x_post], dtype=float
        )
        # a_t = (v_t - v_{t-1}) / dt
        a = np.diff(v, axis=0) / max(dt, 1e-9)
        # j_t = (a_t - a_{t-1}) / dt
        j = np.diff(a, axis=0) / max(dt, 1e-9)
        if len(a) > 0:
            accel_per_track.append(
                float(np.mean(np.linalg.norm(a, axis=1)))
            )
        if len(j) > 0:
            jerk_per_track.append(
                float(np.mean(np.linalg.norm(j, axis=1)))
            )
    return {
        "smoothness_jerk_median":   _safe_median(jerk_per_track),
        "smoothness_jerk_p90":      _safe_percentile(jerk_per_track, 90),
        "smoothness_accel_median":  _safe_median(accel_per_track),
        "smoothness_accel_p90":     _safe_percentile(accel_per_track, 90),
        "n_tracks_for_smoothness":  len(jerk_per_track),
    }


def compute_mode_switching(
    tracks: Dict[int, TrackData],
    dt: float,
    min_hits: int,
) -> Dict[str, float]:
    """
    rate = (к-сть змін argmax(mu)) / (тривалість треку у секундах).
    Агрегуємо як median по треках з hits >= min_hits.
    """
    rates: List[float] = []
    total_switches = 0
    total_seconds = 0.0
    for tid, d in tracks.items():
        if len(d.mode) < min_hits:
            continue
        n_switches = sum(
            1 for i in range(1, len(d.mode)) if d.mode[i] != d.mode[i - 1]
        )
        duration_sec = (d.frames[-1] - d.frames[0] + 1) * dt
        total_switches += n_switches
        total_seconds += duration_sec
        if duration_sec > 0:
            rates.append(n_switches / duration_sec)
    return {
        "mode_switch_rate_median":  _safe_median(rates),
        "mode_switch_rate_p90":     _safe_percentile(rates, 90),
        "mode_switch_overall_total":      int(total_switches),
        "mode_switch_overall_seconds":    float(total_seconds),
        "mode_switch_overall_rate_hz":
            (total_switches / total_seconds) if total_seconds > 0
            else float("nan"),
        "n_tracks_for_mode_switch":  len(rates),
    }


def compute_self_consistency_rmse(
    tracks: Dict[int, TrackData],
) -> Dict[str, float]:
    """RMSE між raw_3d та x_post[(0,3,6)] на кадрах з детекцією.

    9D state: позиційні індекси 0, 3, 6 (було 0, 2, 4 у 6D).
    """
    sq_errs: List[float] = []
    for tid, d in tracks.items():
        for i, raw in enumerate(d.raw_3d):
            if raw is None:
                continue
            # Тільки на кадрах, де трек був асоційований з детекцією
            # (time_since_update=0).
            if d.time_since_update[i] != 0:
                continue
            xp = d.x_post[i]
            err = (raw[0] - xp[0]) ** 2 \
                + (raw[1] - xp[3]) ** 2 \
                + (raw[2] - xp[6]) ** 2
            sq_errs.append(err)
    if not sq_errs:
        return {
            "self_consistency_rmse_3d":  float("nan"),
            "self_consistency_n_pairs":  0,
        }
    return {
        "self_consistency_rmse_3d":   float(math.sqrt(_safe_mean(sq_errs))),
        "self_consistency_n_pairs":   len(sq_errs),
    }


def compute_mahal_stats(
    tracks: Dict[int, TrackData],
    gate_sq: float,
) -> Dict[str, float]:
    """Статистика Mahalanobis^2 ТІЛЬКИ для асоційованих треків (tsu=0)."""
    matched: List[float] = []
    coast: List[float] = []
    for tid, d in tracks.items():
        for m, tsu in zip(d.mahal, d.time_since_update):
            if m is None:
                continue
            if tsu == 0:
                matched.append(float(m))
            else:
                coast.append(float(m))
    above = sum(1 for m in matched if m > gate_sq)
    return {
        "matched_n":           len(matched),
        "matched_mahal_p50":   _safe_percentile(matched, 50),
        "matched_mahal_p90":   _safe_percentile(matched, 90),
        "matched_mahal_p99":   _safe_percentile(matched, 99),
        "matched_above_gate_rate":
            above / len(matched) if matched else float("nan"),
        "coast_mahal_p50":     _safe_percentile(coast, 50),
        "coast_n":             len(coast),
    }


def compute_lifetime_stats(
    tracks: Dict[int, TrackData],
    min_hits: int,
) -> Dict[str, float]:
    lifetimes_frames: List[int] = []
    lifetimes_hits: List[int] = []
    confirmed_lifetimes: List[int] = []
    for tid, d in tracks.items():
        if not d.frames:
            continue
        lf = d.frames[-1] - d.frames[0] + 1
        lifetimes_frames.append(lf)
        lifetimes_hits.append(len(d.frames))
        if "Confirmed" in set(d.states):
            confirmed_lifetimes.append(lf)
    return {
        "n_tracks":                       len(tracks),
        "lifetime_frames_median":         _safe_median(lifetimes_frames),
        "lifetime_frames_p90":            _safe_percentile(lifetimes_frames, 90),
        "lifetime_frames_max":            (max(lifetimes_frames)
                                            if lifetimes_frames else 0),
        "confirmed_lifetime_frames_median":
            _safe_median(confirmed_lifetimes),
        "confirmed_lifetime_frames_max":
            (max(confirmed_lifetimes) if confirmed_lifetimes else 0),
        "n_confirmed_tracks":             len(confirmed_lifetimes),
    }


# ----------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    if not args.jsonl.exists():
        sys.stderr.write(f"[ERR] JSONL не знайдено: {args.jsonl}\n")
        return 2

    header, frames, summary = load_jsonl(args.jsonl)
    dt = float(header.get("dt", 0.02))
    fps = float(header.get("fps", 50.0))

    tracks = collect_tracks(frames)

    metrics: Dict[str, Any] = {
        "source": str(args.jsonl),
        "fps": fps,
        "dt": dt,
        "frame_range": header.get("frame_range"),
        "n_frames": len(frames),
        "tracker_config": header.get("tracker"),
        "yolo_config": header.get("yolo"),
        "lifetime": compute_lifetime_stats(tracks, args.min_hits),
        "coverage": compute_coverage(frames),
        "fragmentation": compute_fragmentation(
            frames, tracks, args.detection_run_max_gap
        ),
        "idsw": compute_idsw(tracks, args.idsw_max_gap,
                              args.idsw_max_dist),
        "smoothness": compute_smoothness(tracks, dt, args.min_hits),
        "mode_switching": compute_mode_switching(tracks, dt,
                                                   args.min_hits),
        "self_consistency": compute_self_consistency_rmse(tracks),
        "mahalanobis": compute_mahal_stats(tracks, args.gating_sq),
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        # JSON: не серіалізуємо numpy float NaN — заміна на null
        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_clean(v) for v in o]
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            if isinstance(o, (np.floating,)):
                return None if math.isnan(float(o)) else float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            return o
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(_clean(metrics), f, ensure_ascii=False, indent=2)
        sys.stderr.write(f"[ok] metrics -> {args.out_json}\n")

    if not args.quiet:
        _pretty_print(metrics)
    return 0


# ----------------------------------------------------------------------
def _pretty_print(m: Dict[str, Any]) -> None:
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            if math.isnan(v):
                return "—"
            if abs(v) >= 1000 or (0 < abs(v) < 0.001):
                return f"{v:.3e}"
            return f"{v:.3f}"
        return str(v)

    print()
    print(f"=== Quality metrics: {m['source']} ===")
    print(f"frames={m['n_frames']} fps={fmt(m['fps'])} dt={fmt(m['dt'])}")
    if m.get("tracker_config"):
        tc = m["tracker_config"]
        print(f"tracker: max_age={tc.get('max_age')} "
              f"min_hits={tc.get('min_hits')} "
              f"gating={tc.get('gating_threshold')}")
    print()
    sections = [
        ("Lifetime", m["lifetime"]),
        ("Coverage", m["coverage"]),
        ("Fragmentation", m["fragmentation"]),
        ("ID switches", {k: v for k, v in m["idsw"].items()
                          if k != "idsw_pairs_sample"}),
        ("Smoothness", m["smoothness"]),
        ("Mode switching", m["mode_switching"]),
        ("Self-consistency RMSE", m["self_consistency"]),
        ("Mahalanobis (matched only)", m["mahalanobis"]),
    ]
    for name, sub in sections:
        print(f"-- {name} --")
        for k, v in sub.items():
            print(f"  {k:45} {fmt(v)}")
        print()


if __name__ == "__main__":
    sys.exit(main())
