#!/usr/bin/env python3
"""
run_segment.py — CLI-скрипт для обробки сегмента відеозапису та запису
повної діагностики роботи IMMTracker у форматі JSONL (по одному рядку на
кадр).

Призначення:
    - Прокрутити відрізок відео [t_start, t_end] (за замовчуванням — все
      відео) через YOLO26 → measurement_transform (3D-промінь) →
      IMMTracker.update.
    - Зберегти у JSONL для кожного кадру: сиру детекцію (u, v, w_box),
      3D-точку (x, y, z), стани усіх активних треків (x_prior, P_prior_diag,
      x_post, P_post_diag, mu вектор 3 режимів, відстань Махаланобіса,
      залишок residual, правдоподібності фільтрів).
    - НЕ використовувати локальні визначення fx_* — лише імпорт із модулів
      IMM_UKF / IMMTracker. dt = 1.0 / source_fps.

Конвенція світових осей: X — поперек майданчика, Y — вертикальна вісь
(вгору, перпендикулярно підлозі), Z — вздовж довгої сторони. Підлога: Y=0.

Приклад використання:
    python scripts/run_segment.py \\
        --video ../data/videos/Japan_vs_Poland_ultrashort.mp4 \\
        --model ../training_models/models/main_model_april.pt \\
        --t_start 0 --t_end 5 \\
        --out_jsonl logs/seg_ultrashort_0_5.jsonl

Можна також підвантажити вже згенеровані детекції з JSON (без перезапуску
YOLO), якщо `--from_json detect_infos/detection_data.json` (тоді --model
не обов'язковий, проте --video все одно потрібен для отримання fps та
координат калібрування за замовчуванням).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Імпорти з модулів проєкту. Будь-які локальні версії fx_*, hx,
# measurement_transform, get_dynamic_transition_matrix заборонені — щоб
# виключити розбіжності між скриптом і ядром фільтра.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BALL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_BALL_DIR))

from get_court_position_methods import (  # noqa: E402
    calibrate_camera,
    get_3d_position,
)
from IMM_UKF import (  # noqa: E402
    measurement_transform,
    hx,
)
from IMMTracker import IMMTracker  # noqa: E402


# ----------------------------------------------------------------------
# Калібрувальні константи за замовчуванням (узгоджені з notebook'ом).
# Конвенція: Y — вертикаль, площа майданчика лежить у площині Y=0.
# ----------------------------------------------------------------------
DEFAULT_PTS_REAL_3D = np.array([
    [0.0, 0.0,  0.0],
    [9.0, 0.0,  0.0],
    [9.0, 0.0, 18.0],
    [0.0, 0.0, 18.0],
], dtype=np.float32)

DEFAULT_PTS_VIDEO_2D = np.array([
    [  69, 1033],
    [1840, 1031],
    [1507,  609],
    [ 405,  609],
], dtype=np.float32)

DEFAULT_K = np.array([
    [1300.0,    0.0,  960.0],
    [   0.0, 1300.0,  540.0],
    [   0.0,    0.0,    1.0],
], dtype=np.float32)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Обробка сегмента відео IMMTracker'ом та збереження "
                    "повної діагностики у JSONL.",
    )
    p.add_argument("--video", required=True, type=Path,
                   help="Шлях до відеофайлу (обов'язково).")
    p.add_argument("--model", type=Path, default=None,
                   help="Шлях до YOLO26 .pt-файлу. Не потрібен, якщо "
                        "передано --from_json.")
    p.add_argument("--from_json", type=Path, default=None,
                   help="Якщо вказано — пропустити YOLO та підвантажити "
                        "детекції з готового JSON (як detect_infos/"
                        "detection_data.json: список об'єктів з полями "
                        "frame, ball_detected, x_pos, y_pos, w_box).")
    p.add_argument("--t_start", type=float, default=0.0,
                   help="Початок сегмента в секундах. За замовчуванням 0.")
    p.add_argument("--t_end", type=float, default=None,
                   help="Кінець сегмента в секундах. За замовчуванням — "
                        "до кінця відео.")
    p.add_argument("--out_jsonl", required=True, type=Path,
                   help="Куди писати JSONL з діагностикою (буде створено "
                        "теку, якщо потрібно).")
    p.add_argument("--conf", type=float, default=0.4,
                   help="YOLO confidence threshold (default 0.4).")
    p.add_argument("--iou", type=float, default=0.45,
                   help="YOLO IoU NMS threshold (default 0.45).")
    p.add_argument("--imgsz", type=int, default=1920,
                   help="YOLO inference image size (default 1920).")
    p.add_argument("--max_age", type=int, default=150,
                   help="IMMTracker.max_age (default 150 — пiдiбраний "
                        "sweep'ом logs/sweep_v1: lifeP90=325 кадрiв "
                        "vs 117 для legacy 50, ~ те саме mGate%).")
    p.add_argument("--min_hits", type=int, default=3,
                   help="IMMTracker.min_hits (default 3).")
    p.add_argument("--gating", type=float, default=11.34,
                   help="χ²-поріг гейтингу Махаланобіса (default 11.34, "
                        "df=3, p=0.99).")
    p.add_argument("--floor_eps", type=float, default=-0.5,
                   help="Якщо raw_3d[1] (Y, висота) < floor_eps — детекція "
                        "відкидається як фізично неможлива. Default -0.5 m.")
    p.add_argument("--ceiling_max", type=float, default=15.0,
                   help="Якщо raw_3d[1] > ceiling_max — детекція "
                        "відкидається. Default 15 m.")
    p.add_argument("--save_calibration", action="store_true",
                   help="Записати в header (перший рядок JSONL) усі "
                        "калібрувальні матриці для відтворюваності.")
    p.add_argument("--verbose", action="store_true",
                   help="Друкувати у stderr прогрес кожні 50 кадрів.")
    return p.parse_args()


# ----------------------------------------------------------------------
# Допоміжне
# ----------------------------------------------------------------------
def to_jsonable(obj: Any) -> Any:
    """Конвертує numpy-структури у звичайні Python-типи для json.dumps."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def load_detection_cache(path: Path) -> Dict[int, Dict[str, Any]]:
    """Завантажити готовий JSON з детекціями у словник {frame_idx: record}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cache: Dict[int, Dict[str, Any]] = {}
    for rec in raw:
        cache[int(rec["frame"])] = rec
    return cache


def yolo_detect_frame(model, frame: np.ndarray,
                      conf: float, iou: float,
                      imgsz: int) -> Optional[Tuple[float, float, float]]:
    """
    Запустити YOLO на одному кадрі. Повертає (x_pos, y_pos, w_box) для
    першої (з найвищою впевненістю) детекції класу 'ball', або None.

    Передбачається single-ball сценарій: якщо знайдено декілька боксів,
    беремо той, що має найвищий conf — це поведінка узгоджена з
    оригінальним notebook'ом.
    """
    res = model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
        stream=False,
    )
    if not res:
        return None
    r = res[0]
    if len(r.boxes) == 0:
        return None
    # Беремо найбільш впевнений бокс (Ultralytics уже сортує за conf desc,
    # але робимо це явно для надійності).
    confs = r.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    xywh = r.boxes.xywh[best_idx].cpu().numpy()
    return float(xywh[0]), float(xywh[1]), float(xywh[2])


def snapshot_track(track) -> Dict[str, Any]:
    """Зробити повний знімок стану одного треку після tracker.update()."""
    imm = track.imm
    x_prior = imm.x_prior
    P_prior = imm.P_prior
    x_post = imm.x_post
    P_post = imm.P_post

    # Знімок усіх трьох фільтрів (правдоподібності — для діагностики
    # перемикання режимів).
    filter_snapshots = []
    for f in imm.filters:
        filter_snapshots.append({
            "name": getattr(f, "name", "?"),
            "x": f.x.tolist(),
            "P_diag": np.diag(f.P).tolist(),
            "likelihood": float(f.likelihood) if f.likelihood is not None
                          else None,
        })

    return {
        "track_id": int(track.track_id),
        "state": str(track.state),
        "hits": int(track.hits),
        "hit_streak": int(track.hit_streak),
        "time_since_update": int(track.time_since_update),
        "x_prior": x_prior.tolist(),
        "P_prior_diag": np.diag(P_prior).tolist(),
        "x_post": x_post.tolist(),
        "P_post_diag": np.diag(P_post).tolist(),
        "mu": imm.mu.tolist(),  # [mu_ballistic, mu_hit, mu_bounce]
        "filters": filter_snapshots,
    }


def compute_innovation(track, z: Optional[np.ndarray],
                       R_matrix: np.ndarray) -> Tuple[Optional[List[float]],
                                                       Optional[float]]:
    """
    Обчислити innovation y = z - h(x_prior) та відстань Махаланобіса
    стосовно поточної детекції (повертаємо None, None якщо детекції немає).
    Викликається ПЕРЕД tracker.update — інакше x_prior ще не оновлено
    черговим predict'ом. Тому ми викликаємо це у місці, де x_prior
    оновлений predict'ом, але update'у ще не було.

    Альтернативно: після tracker.update() x_prior лишається тим самим
    (predict не перезаписує його повторно), тому виклик після оновлення
    також коректний.
    """
    if z is None:
        return None, None
    z_pred = hx(track.imm.x_prior)
    y = (z - z_pred).astype(float)
    try:
        mahal_sq = float(track.get_mahalanobis_distance(z, R_matrix))
    except Exception:
        mahal_sq = None
    return y.tolist(), mahal_sq


# ----------------------------------------------------------------------
# Основний пайплайн
# ----------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    if not args.video.exists():
        sys.stderr.write(f"[ERR] Відеофайл не знайдено: {args.video}\n")
        return 2

    if args.from_json is None and args.model is None:
        sys.stderr.write(
            "[ERR] Потрібно вказати або --model (запуск YOLO), або "
            "--from_json (підвантажити готові детекції).\n"
        )
        return 2

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # 1. Відкриваємо відео, знаходимо fps та межі кадрів.
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.stderr.write(f"[ERR] Не вдалося відкрити відео: {args.video}\n")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        sys.stderr.write(f"[ERR] FPS у відео некоректний: {fps}\n")
        return 2
    dt = 1.0 / fps

    f_start = max(0, int(round(args.t_start * fps)))
    f_end = (total_frames if args.t_end is None
             else min(total_frames, int(round(args.t_end * fps))))
    if f_end <= f_start:
        sys.stderr.write(
            f"[ERR] Порожній діапазон кадрів: [{f_start}, {f_end}).\n"
        )
        return 2
    n_frames_in_segment = f_end - f_start

    sys.stderr.write(
        f"[i] video={args.video.name} fps={fps:.3f} dt={dt:.5f}s "
        f"frames=[{f_start},{f_end}) total={n_frames_in_segment} "
        f"{width}x{height}\n"
    )

    # 2. Калібрування камери (одноразово на сегмент).
    R, tvec, camera_pos = calibrate_camera(
        DEFAULT_PTS_REAL_3D,
        DEFAULT_PTS_VIDEO_2D,
        DEFAULT_K,
        dist=np.zeros((4, 1)),
    )

    # 3. Підготовка джерела детекцій.
    detection_cache: Optional[Dict[int, Dict[str, Any]]] = None
    yolo_model = None
    if args.from_json is not None:
        detection_cache = load_detection_cache(args.from_json)
        sys.stderr.write(
            f"[i] from_json: завантажено {len(detection_cache)} записів із "
            f"{args.from_json}\n"
        )
    else:
        # Ультра-ліниве підключення YOLO (щоб не тягнути ваги, якщо
        # передали --from_json).
        from ultralytics import YOLO
        sys.stderr.write(f"[i] Завантажую YOLO model: {args.model}\n")
        yolo_model = YOLO(str(args.model))

    # 4. IMMTracker.
    tracker = IMMTracker(
        dt=dt,
        max_age=args.max_age,
        min_hits=args.min_hits,
        gating_threshold=args.gating,
    )

    # 5. Перемотування на початковий кадр.
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    out_f = open(args.out_jsonl, "w", encoding="utf-8")
    started_ts = time.time()

    # Header (перший рядок) — конфіг прогону. Так уся діагностика лишається
    # самодостатньою для подальшого аналізу.
    header = {
        "type": "header",
        "video": str(args.video),
        "model": str(args.model) if args.model else None,
        "from_json": str(args.from_json) if args.from_json else None,
        "fps": fps,
        "dt": dt,
        "frame_range": [f_start, f_end],
        "total_frames_in_segment": n_frames_in_segment,
        "yolo": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz},
        "tracker": {
            "max_age": args.max_age,
            "min_hits": args.min_hits,
            "gating_threshold": args.gating,
            "floor_eps": args.floor_eps,
            "ceiling_max": args.ceiling_max,
        },
        "frame_size": [width, height],
    }
    if args.save_calibration:
        header["calibration"] = {
            "pts_real_3d": DEFAULT_PTS_REAL_3D.tolist(),
            "pts_video_2d": DEFAULT_PTS_VIDEO_2D.tolist(),
            "K": DEFAULT_K.tolist(),
            "R": R.tolist(),
            "tvec": tvec.tolist(),
            "camera_pos": camera_pos.ravel().tolist(),
        }
    out_f.write(json.dumps(header, ensure_ascii=False) + "\n")

    # Лічильники для фінальної статистики.
    n_detected = 0
    n_rejected_floor = 0
    n_rejected_ceiling = 0
    n_used = 0

    # 6. Основний цикл по кадрах сегмента.
    # Зауваження: 1-індексація кадрів узгоджується з detect_infos/
    # detection_data.json (frame=1 — перший кадр).
    for k_local in range(n_frames_in_segment):
        f_idx_0based = f_start + k_local         # для cap.read()
        f_idx_1based = f_idx_0based + 1          # для логування / JSON-cache
        t_sec = f_idx_0based * dt

        ok, frame = cap.read()
        if not ok:
            sys.stderr.write(
                f"[w] Не вдалося прочитати кадр {f_idx_0based}; "
                f"завершую цикл.\n"
            )
            break

        # 6.1 Сира детекція.
        raw_det: Dict[str, Any] = {
            "detected": False,
            "u": None, "v": None, "w_box": None,
        }
        if detection_cache is not None:
            rec = detection_cache.get(f_idx_1based)
            if rec is not None and rec.get("ball_detected", False):
                raw_det = {
                    "detected": True,
                    "u": float(rec["x_pos"]),
                    "v": float(rec["y_pos"]),
                    "w_box": float(rec["w_box"]),
                }
        else:
            det = yolo_detect_frame(yolo_model, frame,
                                     args.conf, args.iou, args.imgsz)
            if det is not None:
                u, v, w = det
                raw_det = {"detected": True, "u": u, "v": v, "w_box": w}

        # 6.2 Зворотне проєктування у 3D.
        raw_3d: Optional[List[float]] = None
        z_filtered: Optional[np.ndarray] = None
        reject_reason: Optional[str] = None
        if raw_det["detected"]:
            n_detected += 1
            try:
                pos = get_3d_position(
                    raw_det["u"], raw_det["v"], raw_det["w_box"],
                    DEFAULT_K, R, camera_pos,
                )
                raw_3d = pos.tolist()
                # Фільтр санітарних меж по Y (висоті).
                if pos[1] < args.floor_eps:
                    reject_reason = "below_floor"
                    n_rejected_floor += 1
                elif pos[1] > args.ceiling_max:
                    reject_reason = "above_ceiling"
                    n_rejected_ceiling += 1
                else:
                    z_filtered = pos.astype(np.float32)
                    n_used += 1
            except Exception as e:
                reject_reason = f"get_3d_position_error: {e}"

        # 6.3 Збираємо innovation/Mahalanobis для існуючих треків ДО update
        #     (потрібно, щоб x_prior відповідав прогнозу від попередньої
        #     ітерації). Робимо це лише якщо є валідна детекція. Виклик
        #     get_mahalanobis_distance у track використовує x_prior до того,
        #     як predict цього кадру оновить його — тому виносимо
        #     обчислення до того, як викликати tracker.update.
        #
        #     Але tracker.update внутрішньо викликає track.predict, тож
        #     x_prior буде ПЕРЕЗАПИСАНО. Тому щоб логувати innovation із
        #     properly aligned predict-update parou, нам зручніше викликати
        #     get_mahalanobis_distance ПІСЛЯ tracker.update — там
        #     x_prior — це нове передбачення поточного кадру, P_prior —
        #     нова коваріація передбачення, що саме і використовується
        #     гейтингом Hungarian-асоціації.
        #
        # 6.4 Запускаємо трекер.
        detections_3d: List[np.ndarray] = (
            [z_filtered] if z_filtered is not None else []
        )
        tracker.update(detections_3d)

        # 6.5 Знімаємо стани треків + innovation відносно поточної z.
        track_records: List[Dict[str, Any]] = []
        for track in tracker.tracks:
            snap = snapshot_track(track)
            if z_filtered is not None:
                # Тут x_prior — це predict у цьому кадрі (виконаний
                # всередині tracker.update до асоціації). Розраховуємо
                # innovation / mahalanobis відносно реальної детекції.
                resid, mahal_sq = compute_innovation(
                    track, z_filtered, tracker.R_matrix
                )
                snap["residual"] = resid
                snap["mahalanobis_sq"] = mahal_sq
                snap["mahalanobis"] = (
                    float(np.sqrt(mahal_sq))
                    if mahal_sq is not None and mahal_sq >= 0
                    else None
                )
            else:
                snap["residual"] = None
                snap["mahalanobis_sq"] = None
                snap["mahalanobis"] = None
            track_records.append(snap)

        # 6.6 Запис у JSONL.
        line = {
            "type": "frame",
            "frame": f_idx_1based,
            "frame_0based": f_idx_0based,
            "t": t_sec,
            "raw_detection": raw_det,
            "raw_3d": raw_3d,
            "reject_reason": reject_reason,
            "n_tracks": len(tracker.tracks),
            "n_confirmed": sum(1 for t in tracker.tracks
                                if t.state == "Confirmed"),
            "tracks": track_records,
        }
        out_f.write(json.dumps(to_jsonable(line), ensure_ascii=False) + "\n")

        if args.verbose and (k_local % 50 == 0):
            sys.stderr.write(
                f"  [progress] frame={f_idx_1based:6d} t={t_sec:7.2f}s "
                f"det={raw_det['detected']} tracks={len(tracker.tracks)}\n"
            )

    elapsed = time.time() - started_ts
    summary = {
        "type": "summary",
        "elapsed_sec": elapsed,
        "frames_processed": k_local + 1 if n_frames_in_segment > 0 else 0,
        "n_detected": n_detected,
        "n_rejected_floor": n_rejected_floor,
        "n_rejected_ceiling": n_rejected_ceiling,
        "n_used_for_update": n_used,
        "final_n_tracks": len(tracker.tracks),
        "final_track_ids": [int(t.track_id) for t in tracker.tracks],
    }
    out_f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    out_f.close()
    cap.release()

    sys.stderr.write(
        f"[ok] Готово за {elapsed:.1f}s. detect={n_detected} "
        f"used={n_used} reject_floor={n_rejected_floor} "
        f"reject_ceiling={n_rejected_ceiling}\n"
        f"     final_tracks={summary['final_track_ids']}\n"
        f"     out -> {args.out_jsonl}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
