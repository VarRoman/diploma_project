"""
Калібрувальний помічник для розмітки 4 кутів волейбольного майданчика на
кадрі відео. Користувач натискає 4 точки у такому ж порядку, як визначено
у `pts_real_3d` (за замовчуванням: лівий-передній → правий-передній →
правий-задній → лівий-задній). Скрипт друкує отримані координати у
форматі numpy-масиву для копіювання у `get_court_position_methods.py` /
ноутбук.

Кнопки управління (фокус — у вікні Calibrator):
    лівий клік     — додати точку
    r              — скинути всі точки (Reset)
    u              — undo останньої точки
    s              — зберегти PNG із розміткою (calibration_overlay.png)
    n / Space      — наступний кадр (читаємо ще один з відео)
    q / Esc        — вихід

Виправлено від попередньої версії:
    * cap.release() винесений ЗА межі циклу (раніше викликався після
      першого read → другий read падав).
    * Прибрано закоментований cv2.imshow — тепер кадр реально
      відмальовується.
    * Додано можливість прогортати кадри ('n'), undo ('u'), reset ('r').
    * Інформативніший друк масиву (готовий для копіювання).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Інтерактивна розмітка 4 кутів майданчика у кадрі.",
    )
    p.add_argument(
        "--video", type=Path,
        default=Path("/home/var-roman/Desktop/my_projects/diploma_project/"
                     "data/videos/Japan_vs_Poland_ultrashort.mp4"),
        help="Шлях до відео.",
    )
    p.add_argument("--frame", type=int, default=0,
                   help="Номер кадру (0-based) з якого почати розмітку. "
                        "Default 0.")
    p.add_argument("--max_points", type=int, default=4,
                   help="К-сть точок для розмітки (default 4).")
    p.add_argument("--save_overlay", type=Path,
                   default=Path("calibration_overlay.png"),
                   help="Куди зберігати PNG з розміткою при натисканні 's'.")
    return p.parse_args()


# ----------------------------------------------------------------------
# Інтерфейс
# ----------------------------------------------------------------------
class Calibrator:
    POINT_COLOR = (0, 0, 255)         # BGR — червоний
    LINE_COLOR = (0, 255, 0)          # зелений
    TEXT_COLOR = (255, 255, 255)
    TEXT_BG = (0, 0, 0)
    POINT_RADIUS = 7

    def __init__(self, max_points: int = 4):
        self.max_points = max_points
        self.points: List[List[int]] = []
        self.original_frame: np.ndarray | None = None
        self.window = "Calibrator"

    def _redraw(self) -> None:
        if self.original_frame is None:
            return
        frame = self.original_frame.copy()

        # Точки + нумерація
        for i, (x, y) in enumerate(self.points):
            cv2.circle(frame, (x, y), self.POINT_RADIUS,
                       self.POINT_COLOR, -1)
            label = f"P{i}"
            (tw, th), _ = cv2.getTextSize(label,
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.6, 2)
            cv2.rectangle(frame, (x + 10, y - th - 6),
                          (x + 10 + tw + 4, y + 4),
                          self.TEXT_BG, -1)
            cv2.putText(frame, label, (x + 12, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.TEXT_COLOR, 2)

        # З'єднати точки лініями + замкнути полігон, якщо набрано
        # max_points точок.
        if len(self.points) >= 2:
            for i in range(1, len(self.points)):
                cv2.line(frame, tuple(self.points[i - 1]),
                         tuple(self.points[i]),
                         self.LINE_COLOR, 2)
            if len(self.points) == self.max_points:
                cv2.line(frame, tuple(self.points[-1]),
                         tuple(self.points[0]),
                         self.LINE_COLOR, 2)

        # HUD-підказка
        hud = (f"Click {self.max_points} corners | r=reset u=undo "
               f"s=save n=next q=quit | {len(self.points)}/{self.max_points}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32),
                      self.TEXT_BG, -1)
        cv2.putText(frame, hud, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    self.TEXT_COLOR, 1)

        cv2.imshow(self.window, frame)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.max_points:
            self.points.append([int(x), int(y)])
            print(f"  P{len(self.points) - 1}: [{x}, {y}]")
            if len(self.points) == self.max_points:
                self._print_array()
            self._redraw()

    def _print_array(self) -> None:
        print()
        print("# Скопіюй цей блок у ноутбук / "
              "get_court_position_methods.py:")
        print("pts_video_2d = np.array([")
        for i, (x, y) in enumerate(self.points):
            print(f"    [{x:>5}, {y:>5}],  # P{i}")
        print("], dtype=np.float32)")
        print()

    def reset(self) -> None:
        self.points.clear()
        self._redraw()
        print("[reset]")

    def undo(self) -> None:
        if self.points:
            popped = self.points.pop()
            print(f"[undo] зняв точку {popped}")
            self._redraw()

    def save_overlay(self, path: Path) -> None:
        if self.original_frame is None:
            return
        frame = self.original_frame.copy()
        for i, (x, y) in enumerate(self.points):
            cv2.circle(frame, (x, y), self.POINT_RADIUS,
                       self.POINT_COLOR, -1)
            cv2.putText(frame, f"P{i}", (x + 12, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.TEXT_COLOR, 2)
        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts],
                          isClosed=(len(self.points) == self.max_points),
                          color=self.LINE_COLOR, thickness=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        print(f"[save] overlay -> {path}")

    def set_frame(self, frame: np.ndarray) -> None:
        self.original_frame = frame
        self.points.clear()
        self._redraw()


# ----------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    if not args.video.exists():
        sys.stderr.write(f"[ERR] Відео не знайдено: {args.video}\n")
        return 2

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.stderr.write(f"[ERR] Не вдалося відкрити відео: {args.video}\n")
        return 2

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[i] video={args.video.name} fps={fps:.2f} frames={total}")

    # Прокручуємось на стартовий кадр.
    cur_frame_idx = max(0, min(args.frame, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_idx+500)
    ok, frame = cap.read()
    if not ok or frame is None:
        sys.stderr.write(
            f"[ERR] Не вдалося прочитати кадр {cur_frame_idx}.\n"
        )
        cap.release()
        return 2

    cal = Calibrator(max_points=args.max_points)
    cv2.namedWindow(cal.window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(cal.window, cal._on_mouse)
    cal.set_frame(frame)

    print(f"[i] кадр {cur_frame_idx}: натисни {args.max_points} кутів "
          f"майданчика. q=вийти, n=наступний, r=reset, u=undo, s=save.")

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):  # q / Esc
            break
        elif k == ord("r"):
            cal.reset()
        elif k == ord("u"):
            cal.undo()
        elif k == ord("s"):
            cal.save_overlay(args.save_overlay)
        elif k in (ord("n"), 32):  # n / space
            ok, next_frame = cap.read()
            if not ok or next_frame is None:
                print("[w] Відео скінчилось. Лишаюсь на останньому кадрі.")
                continue
            cur_frame_idx += 1
            cal.set_frame(next_frame)
            print(f"[i] кадр {cur_frame_idx}/{total}")

    cv2.destroyAllWindows()
    cap.release()

    if cal.points:
        print()
        print("=== Підсумок ===")
        cal._print_array()
    return 0


if __name__ == "__main__":
    sys.exit(main())
