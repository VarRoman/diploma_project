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
    get_3d_position_with_cov,
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

# Фізичні розміри майданчика (м): X — ширина 9, Z — довжина 18.
# Використовуються court-bounds гейтом (відсіювання truncated-детекцій
# на межах кадру, де bbox_w обвалюється → Z_c роздувається → 3D-точка
# вистрілює за майданчик). Див. --court_margin.
COURT_X_SIZE = 9.0
COURT_Z_SIZE = 18.0

DEFAULT_PTS_VIDEO_2D = np.array([
    [  69, 1033],   # P0  лівий-передній
    [1840, 1031],   # P1  правий-передній
    [1507,  609],   # P2  правий-задній
    [ 405,  609],   # P3  лівий-задній
], dtype=np.float32)

# Фокусна f=5805 px — РЕАЛЬНА фокусна бродкаст-камери офіційного матчу
# (калібрування 12 точок: calibrateCamera RMS≈1.6 px, репроєкція 1.33 px,
# гострий мінімум при f≈5805). Попереднє значення 1300 px було фокусною
# телефона Google Pixel 9 — стискало глибину у ~4.5× (Z_c ∝ f), через що
# м'яч «літав» на 2-8 м замість фізичних 0-4 м. Принципова точка лишається
# у центрі кадру (960, 540).
DEFAULT_K = np.array([
    [5805.0,    0.0,  960.0],
    [   0.0, 5805.0,  540.0],
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
    p.add_argument("--max_age", type=int, default=80,
                   help="IMMTracker.max_age. Історія: 150 (sweep) → 15 "
                        "(Plan A, проти фантомного коастингу на сміттєвих "
                        "детекціях) → 80 (Phase B + edge-gate). Court-bounds "
                        "гейт (--court_margin) тепер прибирає викиди-фантоми "
                        "в джерелі, тож причина для 15 зникла. Чиста "
                        "розгортка з гейтом: 80 дає найкращу fragmentation "
                        "(0.235) + coverage (0.907) при mode-switch 2.54 Гц.")
    p.add_argument("--min_hits", type=int, default=3,
                   help="IMMTracker.min_hits (default 3).")
    p.add_argument("--gating", type=float, default=16.0,
                   help="χ²-поріг гейтингу Махаланобіса (default 16.0). "
                        "χ²₃ p=0.99 = 11.34, але 16.0 пускає прикордонні "
                        "детекції маневру (mah 12-15) асоціюватись — оживляє "
                        "hit-режим без зростання проліферації треків.")
    p.add_argument("--floor_eps", type=float, default=-0.5,
                   help="Якщо raw_3d[1] (Y, висота) < floor_eps — детекція "
                        "відкидається як фізично неможлива. Default -0.5 m.")
    p.add_argument("--ceiling_max", type=float, default=15.0,
                   help="Якщо raw_3d[1] > ceiling_max — детекція "
                        "відкидається. Default 15 m.")
    p.add_argument("--court_margin", type=float, default=10.0,
                   help="Court-bounds гейт: відкидати детекцію, якщо її "
                        "3D-проєкція лежить далі ніж court_margin метрів за "
                        "межами майданчика по X (ширина 9 м) чи Z (довжина "
                        "18 м). Ловить truncated-детекції на межах кадру, де "
                        "обвал bbox_w роздуває Z_c і кидає точку за зал. "
                        "Default 3.0 m (запас на ігрову зону за лініями). "
                        "Встановіть велике значення (напр. 99) щоб вимкнути.")
    p.add_argument("--enable_cov_reset", action="store_true",
                   help="Увімкнути Step 2.A covariance reset. За замовч. "
                        "ВИМКНЕНО (Path B): cov-reset снапить маневр за 1 "
                        "кадр і короутить IMM hit-режим. Вимкнений — лишає "
                        "residual'у 1-2 кадри → hit-тригер встигає вистрілити "
                        "→ real μ_hit росте 0.12→0.41 через IMM-likelihood. "
                        "Увімкнення повертає старий 1-кадровий cov-reset.")
    p.add_argument("--enable_hysteresis", action="store_true",
                   help="Увімкнути Phase B delayed-accept hysteresis "
                        "(відкладання підозрілих детекцій). Default OFF: "
                        "defer ділить поріг mah_sq=8 з cov-reset і hit-"
                        "тригером і канібалізує їх (обнуляє residual для "
                        "наступного predict). Тримаємо OFF, доки не "
                        "доналаштуємо cov-reset+hit. Coast-skip працює "
                        "незалежно від прапорця.")
    p.add_argument("--disable_single_ball_nms", action="store_true",
                   help="Вимкнути Phase C single-ball NMS. За замовч. "
                        "увімкнено: домен — один м'яч, тож лишаємо лише "
                        "ОДИН Confirmed-трек (найкращий за tsu/hits/mah), "
                        "решту понижуємо до Tentative. Прибирає фантом-"
                        "дублі від сміттєвих спавнів і дивергентного "
                        "коастингу (~20%% кадрів мали 2 confirmed на 1 м'яч).")
    p.add_argument("--disable_physical_gate", action="store_true",
                   help="Вимкнути Gate A — коваріаційно-незалежний стелаж на "
                        "евклідів residual ||z − прогноз||. За замовч. "
                        "увімкнено: Mahalanobis-гейт залежить від P, а P "
                        "роздувається під час коастингу (tsu=47 → P_pos ~80-"
                        "130) → телепорт на FP за 8 м дає mah≈0.7 < гейт і "
                        "ПРИЙМАЄТЬСЯ (катастрофа fr763/807/86). Фіксований "
                        "стелаж незалежний від коастингу й ловить це.")
    p.add_argument("--max_assoc_residual", type=float, default=3.0,
                   help="Поріг Gate A (м). Розподіл прийнятих апдейтів: "
                        "реальний-маневр максимум ≈2.2 м, чиста прірва до "
                        "найближчого телепорта 5.95 м. Default 3.0.")
    p.add_argument("--enable_coast_cov_cap", action="store_true",
                   help="Gate C (ЕКСПЕРИМЕНТ, default OFF) — кап росту "
                        "P-діагоналі під час коастингу. Робить хвіст коастингу "
                        "падучим замість замерзлого, але на довгих сліпих "
                        "коастах (60-80 кадрів без детекцій) малює переконливу "
                        "фікцію падіння з заниженого апекса → відкочено в "
                        "opt-in. Корінь заниженого апекса — драг, не P.")
    p.add_argument("--enable_adaptive_depth_R", action="store_true",
                   help="Phase F (ЕКСПЕРИМЕНТ, default OFF) — адаптивний σ_Z. "
                        "Замість фіксованого R_z=0.5 рахуємо per-detection "
                        "3×3 коваріацію вимірювання, витягнуту вздовж променя: "
                        "σ_Zc = f·D/w²·σ_w. Дрібніший (далекий) bbox → більша "
                        "невизначеність глибини → менша довіра фільтра. Чесне "
                        "відображення монокулярної degeneracy глибини.")
    p.add_argument("--sigma_w", type=float, default=1.0,
                   help="σ ширини bbox (px) для адаптивного σ_Z. Емпірика на "
                        "Japan_vs_Poland: MAD(w_box)≈0.38 px → ~0.56 px після "
                        "1.4826·MAD; default 1.0 px — консервативний запас.")
    p.add_argument("--sigma_w_speed_gain", type=float, default=0.0,
                   help="Phase F — ШВИДКІСНИЙ ҐЕЙТ довіри до глибини. Ефективний "
                        "σ_w на кадрі = σ_w·(1 + gain·img_speed), де img_speed = "
                        "|швидкість центра bbox у кадрі| (px/кадр, hypot(du,dv)). "
                        "МОТИВАЦІЯ: при швидкому русі в кадрі глибина-з-розміру "
                        "найненадійніша (motion-blur роздуває w + висхідний рух "
                        "плутається з відходом углиб → фільтр будує ХИБНУ "
                        "depth-швидкість vz, що жене коаст-овершут, напр. високий "
                        "прийом fr826-882: vz 0→15 м/с, Z 3→14 м). В апексі "
                        "(img_speed→0, чітка рамка) σ_w=номінал → глибині "
                        "довіряємо повністю (точність у полі збережена). Потребує "
                        "from_json (швидкість з сусідніх детекцій). Емпірично "
                        "gain≈0.15-0.25 (швидкий підйом ~15 px/кадр → ×3-4 σ_w). "
                        "Default 0 (вимкнено, A/B-сумісно).")
    p.add_argument("--ball_diameter", type=float, default=0.21,
                   help="Phase F (варіант B) — ЕФЕКТИВНИЙ діаметр м'яча (м) у "
                        "Z_c=f·D/w. Фізичний м'яч = 0.21 м, АЛЕ bbox YOLO "
                        "охоплює м'яч + ~1 см порожнього відступу з кожного "
                        "боку → ефективний D трохи більший. Калібрування за "
                        "фізикою (вимога g=9.81 на чистих балістичних дугах "
                        "fr526-545, fr970-989) дає g≈9.1 при D=0.21 і D_eff≈"
                        "0.23 для рівно 9.81. D ∝ глибина: збільшення D "
                        "ПРОПОРЦІЙНО відсуває весь діапазон глибини (систематична "
                        "корекція padding-зсуву). Default 0.21 (фізичний).")
    p.add_argument("--focal_override", type=float, default=None,
                   help="Перевизначити фокусну f_x=f_y у матриці K (px). "
                        "Калібрування 4 копланарних кутів мінімізує репроєкцію "
                        "при f≈5900 (проти 1300 у коді: 166→2.2 px), що "
                        "розтягує стиснутий діапазон глибини. ЕКСПЕРИМЕНТ для "
                        "A/B; default None (лишити K як є).")
    p.add_argument("--z_min", type=float, default=15.0,
                   help="Нижня межа дистанції камера→м'яч Z_c (м) у "
                        "get_3d_position_with_cov. Z_c = f·D/w. За замовч. 2.0 "
                        "(прийнятно для f=1300/Pixel). ДЛЯ БРОДКАСТ-f≈5805 "
                        "Z_c роздувається у ~4.5× (камера за лінією на ~29 м), "
                        "діапазон Z_c≈[20,52] м → ставте --z_min 15.")
    p.add_argument("--z_max", type=float, default=60.0,
                   help="Верхня межа Z_c (м). За замовч. 25.0 (для f=1300). "
                        "ДЛЯ f≈5805 ставте --z_max 60, інакше z_max=25 "
                        "відрізає УСІ детекції (Z_c=20-52 м) → 0 треків.")
    p.add_argument("--smooth_wbox", type=int, default=3,
                   help="Phase F (варіант B) — часове згладжування ширини "
                        "bbox перед оберненням у глибину Z_c=f·D/w. Вікно "
                        "(непарне число кадрів) ковзної МЕДІАНИ по сусідніх "
                        "ДЕТЕКТОВАНИХ кадрах (gap-aware). Статична камера → "
                        "справжній проєкційний розмір м'яча змінюється плавно, "
                        "тож кадр-до-кадру шум w (робастно ≈1.15 px) — це "
                        "вимірювальний шум, що на далекому м'ячі (w≈24) "
                        "роздувається у ~2.5 м джиттера глибини. Медіана-3/5 "
                        "зрізає одно-кадрові спайки, НЕ зміщуючи траєкторію. "
                        "Default 0 (вимкнено). Типово 3 або 5.")
    p.add_argument("--reject_truncated", type=float, default=3.0,
                   help="Phase F (варіант B) — відкидати детекції, чий bbox "
                        "торкається межі кадру ближче ніж N px (truncation). "
                        "Обрізаний bbox → обвал w → Z_c вистрілює (спайк → "
                        "телепорт → фрагментація треку). ~3%% детекцій труться "
                        "об край. Default 0 (вимкнено). Типово 2-4 px.")
    p.add_argument("--wbox_debias", type=float, default=0.0,
                   help="Phase F (motion-blur) — лінійна корекція напрямкового "
                        "роздування ширини bbox горизонтальним motion-blur: "
                        "w_eff = w - a·du, де du = горизонтальна швидкість "
                        "м'яча у кадрі (px/кадр, центральна різниця по сусідніх "
                        "детекціях), a = цей коеф. Дані Japan_vs_Poland: рух "
                        "праворуч роздуває w (corr(res_w,du)=+0.214), що дає "
                        "систематичний дрейф глибини на пасах (~0.76 м/px при "
                        "w≈40). Емпіричний a≈0.099 px на одиницю du повністю "
                        "знуляє цей зсув (corr→0.000). Default 0 (вимкнено).")
    p.add_argument("--wbox_fuse_h", type=float, default=0.0,
                   help="Phase F (motion-blur) — вага h_box у злитті ефективної "
                        "ширини: w_eff = (1-wt)·w_eff + wt·(h·mean_w/mean_h). "
                        "Висота РОЗВ'ЯЗАНА з горизонтальним рухом (corr 0.012), "
                        "але шумніша (std 3.56 vs w 2.52); зважене злиття "
                        "(wt≈0.33) дає ще ~3%% по джитеру. Потребує h_box у "
                        "from_json (detection_data_wh.json). Default 0 "
                        "(вимкнено; w-only).")
    p.add_argument("--wbox_perp", type=float, default=0.0,
                   help="Phase F (motion-blur, orthogonal) — мотив-ґейтоване "
                        "злиття ПЕРПЕНДИКУЛЯРНОЇ до руху осі bbox. Blur роздуває "
                        "w уздовж |du| та h уздовж |dv| (ортогонально). Тому під "
                        "час низького ГОРИЗОНТАЛЬНОГО пасу зв'язуючого (|du|≫|dv|) "
                        "чиста саме h, а під час вертикального руху — w. Вага: "
                        "blend = (|dv|·w + |du|·h')/(|du|+|dv|), h'=h·mean_w/mean_h; "
                        "в апексі (|du|+|dv|<1) blend=½(w+h') — обидві осі чисті. "
                        "Підсумок: w_eff=(1-perp)·w_eff+perp·blend. Потребує h_box "
                        "(detection_data_wh.json). Default 0 (вимкнено).")
    # --- IMM-тригери (Phase F експеримент із режимами hit/bounce) ----------
    # Дозволяють тюнити активацію режимів просто з CLI, без правок ядра.
    # Дефолти збігаються з get_dynamic_transition_matrix (A/B-сумісність).
    p.add_argument("--hit_residual_min_sq", type=float, default=8.0,
                   help="IMM hit-тригер: нижній поріг Mahalanobis² для "
                        "активації mid-air hit-режиму (default 8.0 ≈ 95-й "
                        "перцентиль χ²₃). Нижче → hit вмикається частіше "
                        "(ризик осциляції); вище → лише на сильних маневрах.")
    p.add_argument("--hit_residual_max_sq", type=float, default=11.34,
                   help="IMM hit-тригер: верхня межа інтерполяції до "
                        "M_hit_target (default 11.34 = χ²₃ p=0.99). При "
                        "mah²≥цього alpha=1 (повний зсув до M_hit_target).")
    p.add_argument("--bounce_height_max", type=float, default=0.55,
                   help="IMM bounce-тригер: висота Y (м), нижче якої при "
                        "vy<0 вмикається bounce-режим (default 0.55). "
                        "Підняти → bounce ловиться вище над підлогою "
                        "(корисно, якщо детекції біля підлоги розріджені).")
    p.add_argument("--bounce_max_coast", type=int, default=3,
                   help="IMM: макс. к-сть кадрів коастингу (tsu), при яких "
                        "ще дозволено bounce-тригер; понад це M=eye(3) "
                        "(чиста балістика на сліпому коасті). Default 3.")
    p.add_argument("--m_hit_target", type=str, default=None,
                   help="IMM hit-режим: цільовий рядок M[0] (3 числа через "
                        "кому, напр. '0.45,0.50,0.05'). До нього "
                        "інтерполюється M[0] при спрацюванні hit-тригера. "
                        "Default None → захардкоджений [0.45,0.50,0.05]. "
                        "Агресивніший (більший hit-компонент) → сильніший "
                        "відгук на удар, але ризик Ballistic↔Hit осциляції.")
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


def smooth_wbox_cache(cache: Dict[int, Dict[str, Any]],
                      window: int) -> Dict[int, float]:
    """
    Gap-aware ковзна медіана ширини bbox по ДЕТЕКТОВАНИХ кадрах.

    Повертає {frame_idx: smoothed_w}. Для кожного детектованого кадру i
    бере медіану w усіх детектованих кадрів у вікні [i-r, i+r] (r=window//2),
    що реально присутні в кеші. Пропуски (недетектовані кадри) природно
    звужують вибірку — зсуву не вносять. Статична камера: справжній
    проєкційний розмір м'яча змінюється плавно, тож медіана прибирає
    одно-кадровий вимірювальний шум w, не торкаючись повільного тренду.
    """
    if window < 2:
        return {}
    r = window // 2
    det_frames = sorted(
        fr for fr, rec in cache.items() if rec.get("ball_detected", False)
    )
    w_by_frame = {
        fr: float(cache[fr]["w_box"]) for fr in det_frames
    }
    out: Dict[int, float] = {}
    for fr in det_frames:
        neigh = [w_by_frame[g] for g in range(fr - r, fr + r + 1)
                 if g in w_by_frame]
        out[fr] = float(np.median(neigh))
    return out


def _img_velocity(cache: Dict[int, Dict[str, Any]], fr: int,
                  key: str) -> float:
    """Швидкість центра bbox у кадрі (px/кадр) уздовж осі key ('x_pos'/'y_pos'),
    центральною різницею по СУСІДНІХ детектованих кадрах (gap-aware)."""
    c0 = cache[fr][key]
    prev = cache.get(fr - 1)
    nxt = cache.get(fr + 1)
    p = prev[key] if prev and prev.get("ball_detected") else None
    n = nxt[key] if nxt and nxt.get("ball_detected") else None
    if p is not None and n is not None:
        return (float(n) - float(p)) / 2.0
    if n is not None:
        return float(n) - float(c0)
    if p is not None:
        return float(c0) - float(p)
    return 0.0


def effective_wbox_map(cache: Dict[int, Dict[str, Any]],
                       wbox_smooth_map: Dict[int, float],
                       debias_a: float,
                       fuse_h_wt: float,
                       perp: float = 0.0) -> Dict[int, float]:
    """
    Motion-blur-стійка ефективна ширина bbox для глибини Z_c = f·D/w_eff.

    Базою береться згладжена w (якщо є wbox_smooth_map), інакше сира w_box.
    Опційні корекції (можна комбінувати):
      (1) debias_a > 0: w_eff = w - a·du. Прибирає НАПРЯМКОВЕ роздування w
          горизонтальним motion-blur (corr(res_w,du)=+0.214 → 0.000 при a≈0.099).
      (2) perp > 0: МОТИВ-ҐЕЙТОВАНА перпендикулярна вісь (h), СУВОРО лише під
          час ГОРИЗОНТАЛЬНО-домінантного руху (низькі паси зв'язуючого на краї).
          Дані: розмиття ОРТОГОНАЛЬНЕ — w росте з |du| (corr +0.233), h росте з
          |dv| (corr +0.214); тобто під час горизонт. пасу чиста саме h. Ґейт:
              g = clamp((|du|-|dv|)/(|du|+|dv|), 0, 1)
          (g=1 чистий горизонт → довіра h; g=0 вертикаль/апекс/діагональ →
          лишаємо w незмінною). h' = h·mean_w/mean_h (масштаб до одиниць w).
          САНІТАРНИЙ ҐЕЙТ: h використовується лише якщо 0.6 ≤ h'/w ≤ 1.5 —
          інакше h обрізана/сміттєва (h_box нестабільна на Y/Z!) і ми падаємо
          назад на w. Підсумок: w_eff = (1 - perp·g)·w + perp·g·h'.
          perp∈(0,1] = макс. частка довіри h. Потребує h_box у from_json.
          NB: апекс НЕ чіпаємо (g=0) — там обидві осі чисті й рівні, суміш
          лише ризикує сміттєвою h; апекс-«стрибок» глибини лікує adaptive R.
      (3) fuse_h_wt > 0: (ЗАСТАРІЛЕ) фіксоване глобальне злиття з h — фрагментує
          трек; лишено для сумісності. Перевага — perp.

    Повертає {frame_idx: w_eff}. Якщо всі корекції вимкнені — порожній dict
    (виклик-сайт падає назад на згладжену/сиру w, поведінка A/B-сумісна).
    """
    if debias_a <= 0.0 and fuse_h_wt <= 0.0 and perp <= 0.0:
        return {}
    det_frames = sorted(
        fr for fr, rec in cache.items() if rec.get("ball_detected", False)
    )
    if not det_frames:
        return {}
    # масштаб h→w (середні по детекціях), якщо потрібні perp або fuse і h_box є
    h_scale = 1.0
    need_h = (perp > 0.0) or (fuse_h_wt > 0.0)
    if need_h:
        ws = [float(cache[fr]["w_box"]) for fr in det_frames
              if cache[fr].get("h_box") is not None]
        hs = [float(cache[fr]["h_box"]) for fr in det_frames
              if cache[fr].get("h_box") is not None]
        if hs and sum(hs) > 0:
            h_scale = float(np.mean(ws) / np.mean(hs))
        else:
            sys.stderr.write(
                "[w] --wbox_perp/--wbox_fuse_h задано, але h_box відсутній у "
                "from_json (потрібен detection_data_wh.json) — вимкнено.\n")
            perp = 0.0
            fuse_h_wt = 0.0
    out: Dict[int, float] = {}
    for fr in det_frames:
        w_base = wbox_smooth_map.get(fr, float(cache[fr]["w_box"]))
        if debias_a > 0.0:
            w_base = w_base - debias_a * _img_velocity(cache, fr, "x_pos")
        if perp > 0.0 and cache[fr].get("h_box") is not None:
            adu = abs(_img_velocity(cache, fr, "x_pos"))
            adv = abs(_img_velocity(cache, fr, "y_pos"))
            h_sc = float(cache[fr]["h_box"]) * h_scale
            denom = adu + adv
            # ґейт горизонтальності: g=1 чистий горизонт, g=0 вертик./апекс/діаг.
            g = 0.0 if denom < 1.0 else max(0.0, min(1.0, (adu - adv) / denom))
            # санітарний ґейт h: відкидаємо обрізану/сміттєву h (нестабільна вісь)
            if g > 0.0 and w_base > 0.0 and 0.6 <= (h_sc / w_base) <= 1.5:
                w_base = (1.0 - perp * g) * w_base + perp * g * h_sc
        elif fuse_h_wt > 0.0 and cache[fr].get("h_box") is not None:
            h_sc = float(cache[fr]["h_box"]) * h_scale
            w_base = (1.0 - fuse_h_wt) * w_base + fuse_h_wt * h_sc
        out[fr] = float(w_base)
    return out


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
    # Phase F: дозволяємо перевизначити фокусну (A/B f=1300 vs ≈5900).
    K_eff = DEFAULT_K.copy()
    if args.focal_override is not None:
        K_eff[0, 0] = float(args.focal_override)
        K_eff[1, 1] = float(args.focal_override)
        sys.stderr.write(
            f"[i] focal_override: f_x=f_y={args.focal_override:.1f} px "
            f"(K-дефолт {DEFAULT_K[0, 0]:.0f})\n"
        )
    R, tvec, camera_pos = calibrate_camera(
        DEFAULT_PTS_REAL_3D,
        DEFAULT_PTS_VIDEO_2D,
        K_eff,
        dist=np.zeros((4, 1)),
    )

    # 3. Підготовка джерела детекцій.
    detection_cache: Optional[Dict[int, Dict[str, Any]]] = None
    wbox_smooth_map: Dict[int, float] = {}
    wbox_eff_map: Dict[int, float] = {}
    yolo_model = None
    if args.from_json is not None:
        detection_cache = load_detection_cache(args.from_json)
        sys.stderr.write(
            f"[i] from_json: завантажено {len(detection_cache)} записів із "
            f"{args.from_json}\n"
        )
        if args.smooth_wbox >= 2:
            wbox_smooth_map = smooth_wbox_cache(detection_cache,
                                                args.smooth_wbox)
            sys.stderr.write(
                f"[i] smooth_wbox: ковзна медіана вікно={args.smooth_wbox} "
                f"застосована до {len(wbox_smooth_map)} детекцій\n"
            )
        if args.wbox_debias > 0.0 or args.wbox_fuse_h > 0.0 or args.wbox_perp > 0.0:
            wbox_eff_map = effective_wbox_map(
                detection_cache, wbox_smooth_map,
                args.wbox_debias, args.wbox_fuse_h, args.wbox_perp)
            sys.stderr.write(
                f"[i] wbox motion-blur корекція: debias_a={args.wbox_debias:.3f} "
                f"fuse_h={args.wbox_fuse_h:.2f} perp={args.wbox_perp:.2f} "
                f"→ {len(wbox_eff_map)} детекцій\n"
            )
    else:
        # Ультра-ліниве підключення YOLO (щоб не тягнути ваги, якщо
        # передали --from_json).
        from ultralytics import YOLO
        sys.stderr.write(f"[i] Завантажую YOLO model: {args.model}\n")
        yolo_model = YOLO(str(args.model))

    # 4. IMMTracker.
    # Парсимо --m_hit_target ("a,b,c" → [a,b,c]); None лишає захардкоджений.
    m_hit_target = None
    if args.m_hit_target is not None:
        try:
            m_hit_target = [float(x) for x in args.m_hit_target.split(",")]
            if len(m_hit_target) != 3:
                raise ValueError("очікую рівно 3 числа")
        except Exception as e:
            sys.stderr.write(
                f"[ERR] --m_hit_target '{args.m_hit_target}' некоректний "
                f"(потрібно 'a,b,c', напр. '0.45,0.50,0.05'): {e}\n"
            )
            return 2
        sys.stderr.write(f"[i] m_hit_target = {m_hit_target}\n")

    tracker = IMMTracker(
        dt=dt,
        max_age=args.max_age,
        min_hits=args.min_hits,
        gating_threshold=args.gating,
        enable_hysteresis=args.enable_hysteresis,
        enable_cov_reset=args.enable_cov_reset,
        enable_single_ball_nms=not args.disable_single_ball_nms,
        enable_physical_gate=not args.disable_physical_gate,
        max_assoc_residual=args.max_assoc_residual,
        enable_coast_cov_cap=args.enable_coast_cov_cap,
        enable_adaptive_depth_R=args.enable_adaptive_depth_R,
        hit_residual_min_sq=args.hit_residual_min_sq,
        hit_residual_max_sq=args.hit_residual_max_sq,
        bounce_height_max=args.bounce_height_max,
        bounce_max_coast=args.bounce_max_coast,
        m_hit_target=m_hit_target,
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
            "court_margin": args.court_margin,
            "enable_hysteresis": args.enable_hysteresis,
            "enable_cov_reset": args.enable_cov_reset,
            "enable_single_ball_nms": not args.disable_single_ball_nms,
            "enable_physical_gate": not args.disable_physical_gate,
            "max_assoc_residual": args.max_assoc_residual,
            "enable_coast_cov_cap": args.enable_coast_cov_cap,
            "enable_adaptive_depth_R": args.enable_adaptive_depth_R,
            "sigma_w": args.sigma_w,
            "sigma_w_speed_gain": args.sigma_w_speed_gain,
            "focal_override": args.focal_override,
            "z_min": args.z_min,
            "z_max": args.z_max,
            "smooth_wbox": args.smooth_wbox,
            "reject_truncated": args.reject_truncated,
            "wbox_debias": args.wbox_debias,
            "wbox_fuse_h": args.wbox_fuse_h,
            "wbox_perp": args.wbox_perp,
            "ball_diameter": args.ball_diameter,
            "hit_residual_min_sq": args.hit_residual_min_sq,
            "hit_residual_max_sq": args.hit_residual_max_sq,
            "bounce_height_max": args.bounce_height_max,
            "bounce_max_coast": args.bounce_max_coast,
            "m_hit_target": m_hit_target,
        },
        "frame_size": [width, height],
    }
    if args.save_calibration:
        header["calibration"] = {
            "pts_real_3d": DEFAULT_PTS_REAL_3D.tolist(),
            "pts_video_2d": DEFAULT_PTS_VIDEO_2D.tolist(),
            "K": K_eff.tolist(),
            "R": R.tolist(),
            "tvec": tvec.tolist(),
            "camera_pos": camera_pos.ravel().tolist(),
        }
    out_f.write(json.dumps(header, ensure_ascii=False) + "\n")

    # Лічильники для фінальної статистики.
    n_detected = 0
    n_rejected_floor = 0
    n_rejected_ceiling = 0
    n_rejected_oob = 0
    n_rejected_trunc = 0
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
                # ефективна (motion-blur-стійка) ширина має пріоритет;
                # далі згладжена; далі сира — A/B-сумісний фолбек.
                w_use = wbox_eff_map.get(
                    f_idx_1based,
                    wbox_smooth_map.get(f_idx_1based, float(rec["w_box"])))
                raw_det = {
                    "detected": True,
                    "u": float(rec["x_pos"]),
                    "v": float(rec["y_pos"]),
                    "w_box": w_use,
                    "w_box_raw": float(rec["w_box"]),
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
        z_cov: Optional[np.ndarray] = None  # per-detection R (адаптивний σ_Z)
        reject_reason: Optional[str] = None
        if raw_det["detected"]:
            n_detected += 1
            # 6.2.0 Гейт truncation: bbox, що торкається межі кадру, має
            # обрізану (замалу) ширину → Z_c вистрілює уздовж променя.
            # Використовуємо СИРУ ширину (геометрія краю), не згладжену.
            if args.reject_truncated > 0:
                w_raw = raw_det.get("w_box_raw", raw_det["w_box"])
                hw = w_raw / 2.0
                eps = args.reject_truncated
                uu, vv = raw_det["u"], raw_det["v"]
                if (uu - hw < eps or uu + hw > width - eps
                        or vv - hw < eps or vv + hw > height - eps):
                    reject_reason = "truncated_bbox"
                    n_rejected_trunc += 1
            if reject_reason is None:
                # Швидкісний ґейт довіри до глибини: при швидкому русі в кадрі
                # глибина-з-розміру ненадійна (blur + up-motion↔depth) → роздуваємо
                # σ_w, щоб фільтр не будував хибну depth-швидкість. В апексі
                # (повільно) σ_w=номінал → глибині довіряємо повністю.
                sigma_w_eff = args.sigma_w
                if (args.sigma_w_speed_gain > 0.0
                        and detection_cache is not None):
                    du = _img_velocity(detection_cache, f_idx_1based, "x_pos")
                    dv = _img_velocity(detection_cache, f_idx_1based, "y_pos")
                    img_speed = float(np.hypot(du, dv))
                    sigma_w_eff = args.sigma_w * (
                        1.0 + args.sigma_w_speed_gain * img_speed)
                try:
                    pos, pos_cov = get_3d_position_with_cov(
                        raw_det["u"], raw_det["v"], raw_det["w_box"],
                        K_eff, R, camera_pos,
                        ball_diameter=args.ball_diameter,
                        sigma_w=sigma_w_eff,
                        z_min=args.z_min, z_max=args.z_max,
                        return_none_on_clip=True,
                    )
                    if pos is None:
                        # PLAN A: Z_c вийшов за межі [z_min, z_max] м — це
                        # майже завжди false positive YOLO з аномально малою
                        # (або великою) шириною bbox. Не передаємо у трекер,
                        # щоб не плодити "треки з кута екрана".
                        reject_reason = "depth_out_of_range"
                    else:
                        raw_3d = pos.tolist()
                        # Фільтр санітарних меж по Y (висоті).
                        m = args.court_margin
                        if pos[1] < args.floor_eps:
                            reject_reason = "below_floor"
                            n_rejected_floor += 1
                        elif pos[1] > args.ceiling_max:
                            reject_reason = "above_ceiling"
                            n_rejected_ceiling += 1
                        elif (pos[0] < -m or pos[0] > COURT_X_SIZE + m
                              or pos[2] < -m or pos[2] > COURT_Z_SIZE + m):
                            # Court-bounds гейт. Truncated-детекція на межі
                            # кадру: bbox_w обвалюється → Z_c роздувається →
                            # точка вистрілює уздовж променя за майданчик.
                            # Глибина зіпсована, довіряти позиції не можна.
                            reject_reason = "out_of_court"
                            n_rejected_oob += 1
                        else:
                            z_filtered = pos.astype(np.float32)
                            z_cov = pos_cov
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
        detection_covs: Optional[List[Optional[np.ndarray]]] = (
            [z_cov] if z_filtered is not None else None
        )
        tracker.update(detections_3d, detection_covs=detection_covs)

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
        "n_rejected_oob": n_rejected_oob,
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
        f"reject_ceiling={n_rejected_ceiling} reject_oob={n_rejected_oob} "
        f"reject_trunc={n_rejected_trunc}\n"
        f"     final_tracks={summary['final_track_ids']}\n"
        f"     out -> {args.out_jsonl}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
