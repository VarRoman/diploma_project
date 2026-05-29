#!/usr/bin/env python3
"""
sweep_imm_params.py — сітковий пошук параметрів IMMTracker'а на одному
сегменті відео. Для кожної комбінації запускає `run_segment.py` із
`--from_json` (тобто без повторного YOLO), обчислює метрики через
`compute_quality_metrics.py`, і сортує результати за обраною
«цільовою функцією».

Сітка задається через YAML-сумісний JSON-файл або через CLI-прапори.
За замовчуванням обходить розумні діапазони `max_age`, `min_hits`,
`gating`. Для повноцінного дослідження Q-матриць у дипломі цей скрипт
можна розширити, додавши перемикач у run_segment.py для Q_var.

Використання:
    python scripts/sweep_imm_params.py \\
        --video ../data/videos/Japan_vs_Poland_ultrashort.mp4 \\
        --detections_json detect_infos/detection_data.json \\
        --out_dir logs/sweep_v1 \\
        --max_age 50,75,100,125 \\
        --min_hits 2,3,4 \\
        --gating 9.21,11.34,14.16 \\
        --score_metric lifetime.lifetime_frames_p90

Сценарій:
    1. Кожна комбінація → новий JSONL у --out_dir.
    2. Усі metrics-json'и зливаються у --out_dir/sweep_results.json
       (масив об'єктів {params, metrics, run_id}).
    3. Друкуємо top-5 та bottom-5 за score_metric.

Безпека: для кожної комбінації існують перевірки, що run_segment.py
завершився з ненульовим detect-лічильником; інакше пропускаємо.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_SCRIPT_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Сітковий пошук параметрів IMMTracker'а."
    )
    p.add_argument("--video", required=True, type=Path,
                   help="Оригінальне відео.")
    p.add_argument("--detections_json", required=True, type=Path,
                   help="Кеш YOLO-детекцій (detection_data.json).")
    p.add_argument("--out_dir", required=True, type=Path,
                   help="Куди писати per-run JSONL та підсумок.")
    p.add_argument("--t_start", type=float, default=0.0)
    p.add_argument("--t_end", type=float, default=None)
    p.add_argument("--max_age", type=str, default="50,75,100,125",
                   help="Кома-розділений список значень max_age.")
    p.add_argument("--min_hits", type=str, default="2,3,4",
                   help="Кома-розділений список значень min_hits.")
    p.add_argument("--gating", type=str, default="9.21,11.34,14.16",
                   help="Кома-розділений список значень "
                        "(χ²-поріг Mahalanobis^2).")
    p.add_argument("--score_metric", type=str,
                   default="lifetime.lifetime_frames_p90",
                   help="Шлях dotted-keys до метрики, за якою сортуємо "
                        "(default: lifetime.lifetime_frames_p90).")
    p.add_argument("--score_descending", action="store_true",
                   default=True,
                   help="Сортувати від найкращих (більше — краще). "
                        "Default True. Для метрик типу jerk/RMSE "
                        "вкажи --no_score_descending.")
    p.add_argument("--no_score_descending", dest="score_descending",
                   action="store_false")
    p.add_argument("--top_k", type=int, default=5,
                   help="Скільки top-/bottom- результатів друкувати "
                        "(default 5).")
    p.add_argument("--keep_jsonls", action="store_true",
                   help="Якщо вказано — НЕ видаляти per-run JSONL "
                        "після завершення (інакше залишаємо тільки "
                        "metrics).")
    p.add_argument("--dry_run", action="store_true",
                   help="Показати, які виклики run_segment.py буде "
                        "зроблено, але нічого не запускати.")
    return p.parse_args()


# ----------------------------------------------------------------------
def parse_csv_numbers(s: str, ty=float) -> List[Any]:
    return [ty(x.strip()) for x in s.split(",") if x.strip()]


def dotted_get(d: Dict[str, Any], path: str) -> Optional[Any]:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# ----------------------------------------------------------------------
def run_one(
    args: argparse.Namespace,
    run_id: int,
    max_age: int,
    min_hits: int,
    gating: float,
    out_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Виконує один запуск run_segment.py + compute_quality_metrics.py.
    Повертає словник {params, metrics, run_id, jsonl_path}, або None
    якщо щось зламалось.
    """
    jsonl_path = out_dir / f"run_{run_id:03d}.jsonl"
    metrics_path = out_dir / f"run_{run_id:03d}_metrics.json"

    run_segment = _SCRIPT_DIR / "run_segment.py"
    cmd_seg = [
        sys.executable, str(run_segment),
        "--video", str(args.video),
        "--from_json", str(args.detections_json),
        "--out_jsonl", str(jsonl_path),
        "--max_age", str(max_age),
        "--min_hits", str(min_hits),
        "--gating", str(gating),
        "--save_calibration",
    ]
    if args.t_start is not None:
        cmd_seg += ["--t_start", str(args.t_start)]
    if args.t_end is not None:
        cmd_seg += ["--t_end", str(args.t_end)]

    if args.dry_run:
        sys.stderr.write(f"[dry] {' '.join(cmd_seg)}\n")
        return None

    t0 = time.time()
    r = subprocess.run(cmd_seg, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(
            f"[!] run_segment.py failed for run {run_id} "
            f"(max_age={max_age} min_hits={min_hits} gating={gating}):\n"
            f"    stdout: {r.stdout[-500:]}\n"
            f"    stderr: {r.stderr[-500:]}\n"
        )
        return None
    elapsed_seg = time.time() - t0

    compute_metrics = _SCRIPT_DIR / "compute_quality_metrics.py"
    cmd_met = [
        sys.executable, str(compute_metrics),
        "--jsonl", str(jsonl_path),
        "--out_json", str(metrics_path),
        "--quiet",
    ]
    r = subprocess.run(cmd_met, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(
            f"[!] compute_quality_metrics failed for run {run_id}:\n"
            f"    {r.stderr[-500:]}\n"
        )
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Прибираємо JSONL після того як зчитали метрики (за бажанням).
    if not args.keep_jsonls and jsonl_path.exists():
        jsonl_path.unlink()

    return {
        "run_id": run_id,
        "params": {
            "max_age": max_age,
            "min_hits": min_hits,
            "gating": gating,
        },
        "metrics": metrics,
        "elapsed_seg_sec": elapsed_seg,
        "metrics_path": str(metrics_path),
    }


# ----------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    max_age_vals = parse_csv_numbers(args.max_age, int)
    min_hits_vals = parse_csv_numbers(args.min_hits, int)
    gating_vals = parse_csv_numbers(args.gating, float)

    combos = list(product(max_age_vals, min_hits_vals, gating_vals))
    sys.stderr.write(
        f"[i] sweep: {len(combos)} комбінацій "
        f"(max_age × min_hits × gating = "
        f"{len(max_age_vals)}×{len(min_hits_vals)}×{len(gating_vals)})\n"
    )

    results: List[Dict[str, Any]] = []
    for i, (ma, mh, ga) in enumerate(combos, start=1):
        sys.stderr.write(
            f"[i] [{i}/{len(combos)}] max_age={ma} "
            f"min_hits={mh} gating={ga:.2f} ... "
        )
        sys.stderr.flush()
        res = run_one(args, run_id=i, max_age=ma,
                      min_hits=mh, gating=ga, out_dir=args.out_dir)
        if res is not None:
            score = dotted_get(res["metrics"], args.score_metric)
            sys.stderr.write(
                f"done {res['elapsed_seg_sec']:.1f}s "
                f"{args.score_metric}={score}\n"
            )
            results.append(res)
        else:
            sys.stderr.write("FAILED або dry_run\n")

    if args.dry_run:
        return 0

    # Підсумок
    summary = {
        "video": str(args.video),
        "detections_json": str(args.detections_json),
        "t_start": args.t_start,
        "t_end": args.t_end,
        "grid": {
            "max_age": max_age_vals,
            "min_hits": min_hits_vals,
            "gating": gating_vals,
        },
        "score_metric": args.score_metric,
        "score_descending": args.score_descending,
        "n_runs_completed": len(results),
        "runs": results,
    }
    out_summary = args.out_dir / "sweep_results.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    sys.stderr.write(f"[ok] summary -> {out_summary}\n")

    # Ранжування
    def score_of(r):
        v = dotted_get(r["metrics"], args.score_metric)
        if v is None or (isinstance(v, float)
                          and (math.isnan(v) or math.isinf(v))):
            return float("-inf") if args.score_descending else float("inf")
        return v

    ranked = sorted(results, key=score_of, reverse=args.score_descending)
    top = ranked[: args.top_k]
    bottom = ranked[-args.top_k:] if len(ranked) > args.top_k else []

    print()
    print(f"=== TOP {len(top)} за {args.score_metric} "
          f"({'desc' if args.score_descending else 'asc'}) ===")
    _print_rank_table(top, args.score_metric)
    if bottom:
        print()
        print(f"=== BOTTOM {len(bottom)} ===")
        _print_rank_table(bottom, args.score_metric)

    return 0


def _print_rank_table(rows: List[Dict[str, Any]],
                       score_metric: str) -> None:
    # Шапка
    cols_metric_short = [
        "lifetime.lifetime_frames_p90",
        "fragmentation.fragmentation",
        "smoothness.smoothness_jerk_median",
        "mode_switching.mode_switch_overall_rate_hz",
        "self_consistency.self_consistency_rmse_3d",
        "mahalanobis.matched_above_gate_rate",
    ]
    print(f"  {'max_age':>7} {'min_hits':>8} {'gating':>7}"
          f"  {'score':>10}"
          f"  {'lifeP90':>8} {'frag':>6} {'jerk':>8}"
          f" {'modeHz':>7} {'rmse':>6} {'mGate%':>7}")
    for r in rows:
        p = r["params"]
        m = r["metrics"]
        score_val = dotted_get(m, score_metric)
        vals = [dotted_get(m, c) for c in cols_metric_short]
        def f(x, prec=2):
            if x is None or (isinstance(x, float)
                              and (math.isnan(x) or math.isinf(x))):
                return "—"
            return f"{x:.{prec}f}"
        print(
            f"  {p['max_age']:>7} {p['min_hits']:>8} {p['gating']:>7.2f}"
            f"  {f(score_val, 3):>10}"
            f"  {f(vals[0], 1):>8} {f(vals[1], 2):>6} "
            f"{f(vals[2], 0):>8} {f(vals[3], 2):>7} "
            f"{f(vals[4], 2):>6} {f(100*vals[5] if vals[5] is not None else None, 1):>7}"
        )


if __name__ == "__main__":
    sys.exit(main())
