#!/usr/bin/env python3
"""
depth_padding_sweep.py — Phase F, A/B/C тест глибини за «порожнім відступом»
рамки YOLO (padding).

ІДЕЯ. Рамка YOLO охоплює м'яч (фізичний діаметр 0.21 м) ПЛЮС невеликий
порожній відступ ~0.5–3 см з кожного боку. Глибина рахується як
    Z_c = f · D_eff / w_box,
де D_eff = 0.21 + 2·padding (padding — відступ на одну сторону, м). Оскільки
Z_c ∝ D_eff, цей відступ ПРОПОРЦІЙНО зсуває ВСЮ глибину — систематична (не
випадкова) корекція. Фізична калібровка (вимога g=9.81 на чистих балістичних
дугах) дала D_eff≈0.23 м (padding≈1 см) для рівно 9.81 м/с².

ЩО РОБИТЬ СКРИПТ. Для кожного значення padding зі списку:
  1) запускає run_segment.py з --ball_diameter D_eff (повний трекер,
     решта параметрів — поточні дефолти run_segment);
  2) друкує A/B/C-рядок статистики глибини домінантного Confirmed-треку
     (світові X/Y/Z, дистанція до камери Z_cam, jerk);
  3) (опційно, --video_out) рендерить overlay-відео з траєкторією + табло
     координат через overlay_tracks.py — по одному файлу на padding.

ВИКОРИСТАННЯ.
  # повна розгортка 0.5..3.0 см (крок 0.5), зі статистикою + відео:
  python scripts/depth_padding_sweep.py \
      --video ../data/videos/Japan_vs_Poland_ultrashort.mp4 \
      --from_json detect_infos/detection_data.json \
      --out_dir overlays/padding --video_out

  # одне значення на свій розсуд (швидко, лише статистика):
  python scripts/depth_padding_sweep.py ... --padding_list 1.5

«Регулювати параметр на свій розсуд» = передати будь-який --padding_list
(одне число чи кілька через кому). Фільтрована траєкторія залежить від
D_eff нелінійно (гравітація фіксована, drag квадратичний), тож кожне
значення = окремий прогін трекера (живого слайдера бути не може — це чесно).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_BALL_DIR = _SCRIPT_DIR.parent
PY = sys.executable
BALL_PHYS = 0.21  # фізичний діаметр м'яча (м)


def run_segment(video, from_json, d_eff, out_jsonl, extra_args):
    cmd = [PY, str(_SCRIPT_DIR / "run_segment.py"),
           "--video", str(video),
           "--from_json", str(from_json),
           "--ball_diameter", f"{d_eff:.5f}",
           "--save_calibration",
           "--out_jsonl", str(out_jsonl)] + list(extra_args)
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_overlay(video, jsonl, out_mp4, max_frames):
    cmd = [PY, str(_SCRIPT_DIR / "overlay_tracks.py"),
           "--video", str(video), "--jsonl", str(jsonl),
           "--out", str(out_mp4)]
    if max_frames:
        cmd += ["--max_frames", str(max_frames)]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def depth_stats(jsonl):
    """Статистика глибини домінантного Confirmed-треку з JSONL-логу."""
    with open(jsonl, "r", encoding="utf-8") as f:
        recs = [json.loads(x) for x in f if x.strip()]
    hdr = recs[0]
    frames = [r for r in recs if r.get("type") == "frame"]
    cal = hdr.get("calibration", {})
    R = np.array(cal["R"], float) if "R" in cal else None
    cam = (np.array(cal["camera_pos"], float).reshape(3)
           if "camera_pos" in cal else None)

    by_id = {}
    n_used = 0
    for r in frames:
        if r.get("reject_reason") is None and r.get("raw_3d") is not None:
            n_used += 1
        for t in r["tracks"]:
            if t["state"] != "Confirmed":
                continue
            xp = t["x_post"]
            by_id.setdefault(t["track_id"], []).append(
                (r["frame_0based"], xp[0], xp[3], xp[6]))
    if not by_id:
        return None
    dom = max(by_id, key=lambda k: len(by_id[k]))
    seq = sorted(by_id[dom])
    P = np.array([[s[1], s[2], s[3]] for s in seq])  # world X,Y,Z
    # дистанція камера→м'яч уздовж осі камери (z-компонента q=R·(P−cam))
    if R is not None and cam is not None:
        q = (R @ (P - cam.reshape(1, 3)).T).T
        zcam = q[:, 2]
    else:
        zcam = np.full(len(P), np.nan)
    d3 = np.diff(P, n=3, axis=0)
    jerk = float(np.mean(np.linalg.norm(d3, axis=1))) if len(d3) else float("nan")
    return {
        "n_used": n_used, "dom_id": dom, "dom_len": len(seq),
        "X": (P[:, 0].min(), P[:, 0].max()),
        "Y": (P[:, 1].min(), P[:, 1].max()),
        "Z": (P[:, 2].min(), P[:, 2].max()),
        "Zcam": (np.nanmin(zcam), np.nanmax(zcam), np.nanmean(zcam)),
        "jerk": jerk,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Phase F: A/B/C тест глибини за padding рамки YOLO.")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--from_json", required=True, type=Path)
    ap.add_argument("--out_dir", type=Path, default=Path("overlays/padding"))
    ap.add_argument("--padding_list", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0",
                    help="Список padding (см/сторону) через кому. "
                         "Default '0.5,1.0,1.5,2.0,2.5,3.0'. Можна одне число.")
    ap.add_argument("--video_out", action="store_true",
                    help="Рендерити overlay-відео на кожне padding-значення "
                         "(інакше лише статистика).")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Обмежити к-сть кадрів overlay (0=всі).")
    ap.add_argument("--keep_jsonl", action="store_true",
                    help="Лишати JSONL-логи в out_dir (інакше у /tmp).")
    # усе після '--' прокидається у run_segment (напр. IMM-тригери).
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="Додаткові аргументи run_segment після '--'.")
    args = ap.parse_args()

    try:
        pads = [float(x) for x in args.padding_list.split(",") if x.strip()]
    except ValueError:
        sys.exit(f"[err] некоректний --padding_list: {args.padding_list}")
    extra = [a for a in args.rest if a != "--"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir if args.keep_jsonl else Path("/tmp")

    print(f"\n{'pad/side':>8} {'D_eff':>6} | {'used':>4} {'domLen':>6} "
          f"| {'Y[min,max]':>16} | {'Z(len)[min,max]':>18} "
          f"| {'Zcam[min,max,mean]':>22} | {'jerk':>7}")
    print("-" * 110)

    results = []
    for p in pads:
        d_eff = BALL_PHYS + 2.0 * (p / 100.0)
        tag = f"pad{p:g}".replace(".", "_")
        jsonl = log_dir / f"sweep_{tag}.jsonl"
        run_segment(args.video, args.from_json, d_eff, jsonl, extra)
        st = depth_stats(jsonl)
        if st is None:
            print(f"{p:>7.1f}c {d_eff:>6.3f} | NO confirmed tracks")
            continue
        results.append((p, d_eff, st))
        print(f"{p:>7.1f}c {d_eff:>6.3f} | {st['n_used']:>4} {st['dom_len']:>6} "
              f"| [{st['Y'][0]:5.2f},{st['Y'][1]:5.2f}] "
              f"| [{st['Z'][0]:6.2f},{st['Z'][1]:6.2f}] "
              f"| [{st['Zcam'][0]:5.1f},{st['Zcam'][1]:5.1f},{st['Zcam'][2]:5.1f}] "
              f"| {st['jerk']:7.4f}")
        if args.video_out:
            out_mp4 = args.out_dir / f"overlay_{tag}.mp4"
            run_overlay(args.video, jsonl, out_mp4, args.max_frames)
            sys.stderr.write(f"  [video] -> {out_mp4}\n")

    print("-" * 110)
    print("Z_cam ∝ D_eff: padding пропорційно відсуває ВСЮ глибину. "
          "Фіз. калібровка (g=9.81) → D_eff≈0.23 (padding≈1.0 см).")
    if args.video_out:
        print(f"Відео: {args.out_dir}/overlay_pad*.mp4 (HUD підписує D_eff/padding).")


if __name__ == "__main__":
    main()
