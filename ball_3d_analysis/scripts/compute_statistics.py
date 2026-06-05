#!/usr/bin/env python3
"""
compute_statistics.py — spatio-temporal game statistics from the ball trajectory.

Input:  JSONL diagnostics of a `run_segment.py` run (header + one line per frame).
Output: a readable report on stdout + (optionally) a JSON summary (--out_json).

Geometry (consistent with the run_segment / overlay_tracks calibration):
    World coordinates: X = court width  [0, 9] m,
                       Y = height above the floor (m),
                       Z = court length [0, 18] m.
    The net splits the court lengthwise -> plane Z = 9 m, height 2.43 m.
    "Near" side = Z < 9, "far" side = Z > 9.

What we compute:
    • coverage (fraction of frames with a confirmed ball);
    • time the ball spends on each side (via the sign of Z - 9);
    • net crossings (sign changes of Z - 9 along a single track);
    • ball speed (from the filter state [vx,vy,vz]): mean / median / max;
    • height (mean / max, frames above the net);
    • total ball path length;
    • rally segments (continuous stretches of a single track).

Garbage filter (important for broadcast with camera cuts): calibration is only
valid for the main game camera. On replays / close-ups the homography is wrong
-> 3D "flies" off the court. By default we count only physically plausible
frames (inside the court + a sanity speed cap).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Court geometry (m) — same as in overlay_tracks / run_segment.
COURT_X = 9.0          # width
COURT_Z = 18.0         # length
NET_Z = COURT_Z / 2.0  # net plane along the length
NET_H = 2.43           # net height (men's volleyball)


def load_frames(path: Path) -> tuple[dict, List[dict]]:
    """Read the header + all per-frame JSONL lines."""
    header: Dict[str, Any] = {}
    frames: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if i == 0 and d.get("type") == "header":
                header = d
            elif d.get("type") == "frame":
                frames.append(d)
    return header, frames


def pick_primary(frame: dict) -> Optional[dict]:
    """Dominant ball of the frame: the confirmed track with the most hits.
    If none is confirmed — None (Tentative tracks are not counted)."""
    tracks = frame.get("tracks") or []
    confirmed = [t for t in tracks if t.get("state") == "Confirmed"]
    if not confirmed:
        return None
    return max(confirmed, key=lambda t: t.get("hits", 0))


def in_court(x: float, y: float, z: float, ceiling: float,
             margin: float) -> bool:
    """Whether the 3D point lies in a physically plausible volume (with a small
    margin around the perimeter)."""
    return (-margin <= x <= COURT_X + margin
            and -0.5 <= y <= ceiling
            and -margin <= z <= COURT_Z + margin)


def compute(path: Path, speed_cap: float, court_margin: float,
            ceiling: float, min_rally_frames: int) -> Dict[str, Any]:
    header, frames = load_frames(path)
    dt = float(header.get("dt") or 0.02)
    fps = float(header.get("fps") or (1.0 / dt))
    n_total = len(frames)

    # Per-frame samples of the dominant track (only valid 3D + tsu==0, i.e. a
    # real association with a detection — cleanest for statistics).
    samples: List[dict] = []   # {fr, tid, X,Y,Z, speed, side}
    track_ids = set()
    confirmed_frames = 0

    for fr in frames:
        t = pick_primary(fr)
        if t is None:
            continue
        confirmed_frames += 1
        xp = t.get("x_post") or []
        if len(xp) < 9:
            continue
        X, Y, Z = xp[0], xp[3], xp[6]
        vx, vy, vz = xp[1], xp[4], xp[7]
        if not in_court(X, Y, Z, ceiling, court_margin):
            continue
        if t.get("time_since_update", 0) != 0:
            continue  # only genuinely updated frames (no coast extrapolation)
        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))
        if speed > speed_cap:
            continue  # sanity rejection of garbage speed spikes
        track_ids.add(t.get("track_id"))
        samples.append(dict(fr=fr.get("frame"), tid=t.get("track_id"),
                            X=X, Y=Y, Z=Z, speed=speed))

    n_valid = len(samples)
    res: Dict[str, Any] = {
        "source": str(path),
        "video": header.get("video"),
        "fps": fps,
        "dt": dt,
        "frames_total": n_total,
        "frames_confirmed": confirmed_frames,
        "frames_valid_inplay": n_valid,
        "coverage_confirmed": (confirmed_frames / n_total) if n_total else 0.0,
        "coverage_valid": (n_valid / n_total) if n_total else 0.0,
        "unique_tracks": len(track_ids),
    }
    if n_valid == 0:
        res["note"] = "Немає валідних ігрових кадрів — нема що рахувати."
        return res

    Z = np.array([s["Z"] for s in samples])
    Y = np.array([s["Y"] for s in samples])
    spd = np.array([s["speed"] for s in samples])

    # --- Time per side (net @ Z=9) ---
    near = int(np.sum(Z < NET_Z))
    far = int(np.sum(Z >= NET_Z))
    res["time_near_s"] = near * dt
    res["time_far_s"] = far * dt
    res["time_near_pct"] = near / n_valid
    res["time_far_pct"] = far / n_valid

    # --- Speed (m/s) ---
    res["speed_mean_ms"] = float(np.mean(spd))
    res["speed_median_ms"] = float(np.median(spd))
    res["speed_max_ms"] = float(np.max(spd))
    res["speed_p95_ms"] = float(np.percentile(spd, 95))
    res["speed_mean_kmh"] = res["speed_mean_ms"] * 3.6
    res["speed_max_kmh"] = res["speed_max_ms"] * 3.6
    res["speed_max_frame"] = int(samples[int(np.argmax(spd))]["fr"])

    # --- Height ---
    res["height_mean_m"] = float(np.mean(Y))
    res["height_max_m"] = float(np.max(Y))
    res["height_max_frame"] = int(samples[int(np.argmax(Y))]["fr"])
    res["frames_above_net"] = int(np.sum(Y > NET_H))

    # --- Net crossings + rally segments ---
    # Walk the samples, grouping them into continuous "rallies": a track break
    # (track_id change or frame jump) ends a segment. A net crossing is counted
    # only within a single segment (physical continuity).
    net_crossings = 0
    rallies: List[dict] = []
    seg_start = 0
    for i in range(1, n_valid + 1):
        broken = (i == n_valid
                  or samples[i]["tid"] != samples[i - 1]["tid"]
                  or (samples[i]["fr"] - samples[i - 1]["fr"]) > fps)  # >1 s gap
        if not broken:
            # net crossing inside the segment
            if (Z[i] - NET_Z) * (Z[i - 1] - NET_Z) < 0:
                net_crossings += 1
            continue
        seg = samples[seg_start:i]
        if len(seg) >= min_rally_frames:
            zz = np.array([s["Z"] for s in seg])
            rallies.append(dict(
                tid=seg[0]["tid"],
                frame_start=int(seg[0]["fr"]),
                frame_end=int(seg[-1]["fr"]),
                duration_s=len(seg) * dt,
                z_min=float(zz.min()), z_max=float(zz.max()),
                max_speed_ms=float(max(s["speed"] for s in seg)),
            ))
        seg_start = i
    res["net_crossings"] = net_crossings
    res["rallies"] = rallies
    res["rally_count"] = len(rallies)

    # --- Total ball path (within a single track, consecutive frames only) ---
    dist = 0.0
    for i in range(1, n_valid):
        if (samples[i]["tid"] == samples[i - 1]["tid"]
                and (samples[i]["fr"] - samples[i - 1]["fr"]) <= 2):
            dx = samples[i]["X"] - samples[i - 1]["X"]
            dy = samples[i]["Y"] - samples[i - 1]["Y"]
            dz = samples[i]["Z"] - samples[i - 1]["Z"]
            dist += float(np.sqrt(dx * dx + dy * dy + dz * dz))
    res["ball_path_length_m"] = dist
    return res


def fmt_report(r: Dict[str, Any]) -> str:
    L: List[str] = []
    a = L.append
    a("=" * 64)
    a("  СТАТИСТИКА ГРИ З 3D-ТРАЄКТОРІЇ М'ЯЧА")
    a("=" * 64)
    a(f"Джерело : {r['source']}")
    a(f"Відео   : {r.get('video')}")
    a(f"Кадрів  : {r['frames_total']}  "
      f"({r['frames_total'] * r['dt']:.1f} с @ {r['fps']:.0f} FPS)")
    a(f"Сітка   : площина Z = {NET_Z:.1f} м, висота {NET_H} м")
    a("")
    a("-- Покриття --")
    a(f"  Кадрів із підтвердженим м'ячем : {r['frames_confirmed']} "
      f"({r['coverage_confirmed']*100:.1f}%)")
    a(f"  Валідних ігрових кадрів        : {r['frames_valid_inplay']} "
      f"({r['coverage_valid']*100:.1f}%)  [у межах майданчика, tsu=0]")
    a(f"  Унікальних треків              : {r['unique_tracks']}")
    if r.get("note"):
        a("")
        a(r["note"])
        return "\n".join(L)
    a("")
    a("-- Час за сторонами (сітка @ Z=9 м) --")
    a(f"  Ближня (Z<9) : {r['time_near_s']:.1f} с  ({r['time_near_pct']*100:.1f}%)")
    a(f"  Дальня (Z>9) : {r['time_far_s']:.1f} с  ({r['time_far_pct']*100:.1f}%)")
    a("")
    a("-- Швидкість м'яча --")
    a(f"  Середня : {r['speed_mean_ms']:.2f} м/с  ({r['speed_mean_kmh']:.1f} км/год)")
    a(f"  Медіана : {r['speed_median_ms']:.2f} м/с")
    a(f"  95-й перц.: {r['speed_p95_ms']:.2f} м/с")
    a(f"  Максимум: {r['speed_max_ms']:.2f} м/с  ({r['speed_max_kmh']:.1f} км/год)"
      f"  [кадр {r['speed_max_frame']}]")
    a("")
    a("-- Висота --")
    a(f"  Середня : {r['height_mean_m']:.2f} м")
    a(f"  Максимум: {r['height_max_m']:.2f} м  [кадр {r['height_max_frame']}]")
    a(f"  Кадрів над сіткою (Y>{NET_H}) : {r['frames_above_net']}")
    a("")
    a("-- Переходи через сітку та розіграші --")
    a(f"  Переходів через сітку : {r['net_crossings']}")
    a(f"  Сегментів-розіграшів  : {r['rally_count']}")
    a(f"  Сумарний шлях м'яча   : {r['ball_path_length_m']:.1f} м")
    if r["rallies"]:
        a("")
        a("  Топ-розіграші за тривалістю:")
        top = sorted(r["rallies"], key=lambda x: -x["duration_s"])[:8]
        for s in top:
            a(f"    #{s['tid']:<3} кадри {s['frame_start']}-{s['frame_end']}  "
              f"{s['duration_s']:.1f}с  Z[{s['z_min']:.1f}..{s['z_max']:.1f}]  "
              f"vmax {s['max_speed_ms']:.1f} м/с")
    a("=" * 64)
    return "\n".join(L)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", required=True, type=Path,
                   help="JSONL-діагностика з run_segment.py.")
    p.add_argument("--out_json", type=Path, default=None,
                   help="Куди зберегти JSON-зведення (опційно).")
    p.add_argument("--speed_cap", type=float, default=40.0,
                   help="Санітарний кап швидкості (м/с): кадри понад нього "
                        "вважаємо сміттям. Default 40 (верх елітної подачі).")
    p.add_argument("--court_margin", type=float, default=2.0,
                   help="Запас (м) за периметром майданчика для фільтра "
                        "правдоподібності 3D. Default 2.0.")
    p.add_argument("--ceiling", type=float, default=15.0,
                   help="Стеля висоти (м) для фільтра правдоподібності. "
                        "Default 15.0.")
    p.add_argument("--min_rally_frames", type=int, default=10,
                   help="Мін. к-сть валідних кадрів, щоб відрізок вважався "
                        "розіграшем. Default 10 (0.2 с @ 50 FPS).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.jsonl.exists():
        sys.stderr.write(f"[ERR] Немає файлу: {args.jsonl}\n")
        sys.exit(1)
    r = compute(args.jsonl, args.speed_cap, args.court_margin,
                args.ceiling, args.min_rally_frames)
    print(fmt_report(r))
    if args.out_json:
        args.out_json.write_text(json.dumps(r, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        sys.stderr.write(f"[ok] JSON-зведення -> {args.out_json}\n")


if __name__ == "__main__":
    main()
