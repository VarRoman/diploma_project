#!/usr/bin/env python3
"""
compute_statistics.py — просторово-часова статистика гри з траєкторії м'яча.

Вхід: JSONL-діагностика прогону `run_segment.py` (header + по рядку на кадр).
Вихід: читабельний звіт у stdout + (опційно) JSON-зведення (--out_json).

ГЕОМЕТРІЯ (узгоджено з калібруванням run_segment / overlay_tracks):
    Світові координати: X = ширина майданчика [0, 9] м,
                        Y = висота над підлогою (м),
                        Z = довжина майданчика [0, 18] м.
    Сітка ділить майданчик по довжині навпіл → площина Z = 9 м, висота 2.43 м.
    «Ближня» сторона = Z < 9, «дальня» = Z > 9.

ЩО РАХУЄМО:
    • покриття (частка кадрів із підтвердженим м'ячем);
    • час перебування м'яча на кожній стороні (через знак Z − 9);
    • переходи через сітку (зміни знаку Z − 9 уздовж одного треку);
    • швидкість м'яча (зі стану фільтра [vx,vy,vz]): середня / медіана / макс.;
    • висоту (середня / макс., кадри над сіткою);
    • сумарний шлях м'яча;
    • сегменти-розіграші (неперервні відрізки одного треку).

ФІЛЬТР СМІТТЯ (важливо для бродкасту з перемиканням камер): калібрування
валідне лише для основної ігрової камери. На повторах/крупних планах гомографія
хибна → 3D «вилітає» за майданчик. За замовчуванням рахуємо лише фізично
правдоподібні кадри (всередині майданчика + санітарний кап швидкості).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Геометрія майданчика (м) — та сама, що в overlay_tracks / run_segment.
COURT_X = 9.0          # ширина
COURT_Z = 18.0         # довжина
NET_Z = COURT_Z / 2.0  # площина сітки по довжині
NET_H = 2.43           # висота сітки (чоловічий волейбол)


def load_frames(path: Path) -> tuple[dict, List[dict]]:
    """Зчитати header + усі кадрові рядки JSONL."""
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
    """Домінантний м'яч кадру: підтверджений трек із найбільшою к-стю влучень.
    Якщо підтверджених нема — None (Tentative не рахуємо у статистику)."""
    tracks = frame.get("tracks") or []
    confirmed = [t for t in tracks if t.get("state") == "Confirmed"]
    if not confirmed:
        return None
    return max(confirmed, key=lambda t: t.get("hits", 0))


def in_court(x: float, y: float, z: float, ceiling: float,
             margin: float) -> bool:
    """Чи лежить 3D-точка у фізично правдоподібному обсязі (з невеликим
    запасом margin по периметру)."""
    return (-margin <= x <= COURT_X + margin
            and -0.5 <= y <= ceiling
            and -margin <= z <= COURT_Z + margin)


def compute(path: Path, speed_cap: float, court_margin: float,
            ceiling: float, min_rally_frames: int) -> Dict[str, Any]:
    header, frames = load_frames(path)
    dt = float(header.get("dt") or 0.02)
    fps = float(header.get("fps") or (1.0 / dt))
    n_total = len(frames)

    # Per-кадрові вибірки домінантного треку (лише валідні 3D + tsu==0,
    # тобто реальна асоціація з детекцією — найчистіше для статистики).
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
            continue  # лише реально оновлені кадри (без коаст-екстраполяції)
        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))
        if speed > speed_cap:
            continue  # санітарний відсів сміттєвих стрибків швидкості
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

    # --- Час за сторонами (сітка @ Z=9) ---
    near = int(np.sum(Z < NET_Z))
    far = int(np.sum(Z >= NET_Z))
    res["time_near_s"] = near * dt
    res["time_far_s"] = far * dt
    res["time_near_pct"] = near / n_valid
    res["time_far_pct"] = far / n_valid

    # --- Швидкість (м/с) ---
    res["speed_mean_ms"] = float(np.mean(spd))
    res["speed_median_ms"] = float(np.median(spd))
    res["speed_max_ms"] = float(np.max(spd))
    res["speed_p95_ms"] = float(np.percentile(spd, 95))
    res["speed_mean_kmh"] = res["speed_mean_ms"] * 3.6
    res["speed_max_kmh"] = res["speed_max_ms"] * 3.6
    res["speed_max_frame"] = int(samples[int(np.argmax(spd))]["fr"])

    # --- Висота ---
    res["height_mean_m"] = float(np.mean(Y))
    res["height_max_m"] = float(np.max(Y))
    res["height_max_frame"] = int(samples[int(np.argmax(Y))]["fr"])
    res["frames_above_net"] = int(np.sum(Y > NET_H))

    # --- Переходи через сітку + сегменти-розіграші ---
    # Йдемо по вибірках, групуючи у неперервні «розіграші»: розрив треку
    # (зміна track_id або стрибок кадру) завершує сегмент. Перехід сітки
    # рахуємо лише ВСЕРЕДИНІ одного сегмента (фізична безперервність).
    net_crossings = 0
    rallies: List[dict] = []
    seg_start = 0
    for i in range(1, n_valid + 1):
        broken = (i == n_valid
                  or samples[i]["tid"] != samples[i - 1]["tid"]
                  or (samples[i]["fr"] - samples[i - 1]["fr"]) > fps)  # >1 c gap
        if not broken:
            # перехід сітки всередині сегмента
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

    # --- Сумарний шлях м'яча (лише в межах одного треку, послідовні кадри) ---
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
