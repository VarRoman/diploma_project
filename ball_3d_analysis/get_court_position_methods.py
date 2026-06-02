import cv2
import numpy as np
import torch as t
import torchvision
import torchaudio
import albumentations as A
from ultralytics import YOLO

pts_real_3d = np.array([
    [0.0, 0.0, 0.0],
    [9.0, 0.0, 0.0],
    [9.0, 0.0, 18.0],
    [0.0, 0.0, 18.0]],dtype=np.float32)

pts_video_2d = np.array([
    [  69, 1033],   # P0  лівий-передній
    [1840, 1031],   # P1  правий-передній
    [1507,  609],   # P2  правий-задній
    [ 405,  609]],  # P3  лівий-задній
    dtype=np.float32)

K = np.array([
    [1300.0, 0.0,    960.0],
    [0.0,    1300.0, 540.0],
    [0.0,    0.0,    1.0]],dtype=np.float32)


def calibrate_camera(pts_3d, pts_2d, camera_matrix, dist, up_axis=1):
    """
    Калібрування камери методом PnP. Повертає (R, tvec, camera_position),
    де R — матриця обертання world→camera (3x3), tvec — вектор переносу
    у систему камери (3,1), camera_position — позиція камери у світовій
    СК (3,1).

    Параметр up_axis (за замовчуванням 1, тобто Y) — індекс «вертикальної»
    осі у вашій світовій СК. Використовується для дезамбіюації знаку при
    калібруванні за КОПЛАНАРНИМИ точками: solvePnP для копланарної задачі
    має дві гілки розв'язку, віддзеркалені відносно площини точок, і
    OpenCV може повернути ту, де камера опиняється «під підлогою»
    (camera_position[up_axis] < 0). Така гілка дає коректну репроєкцію
    калібрувальних точок, але робить наступні виклики get_3d_position
    «дзеркальними» по вертикалі (м'яч у повітрі отримує від'ємну висоту).
    Тут ми це детектимо і явно віддзеркалюємо вертикальну вісь світу так,
    щоб камера стояла НАД підлогою. Якщо калібрування виконане за
    некопланарними точками, передайте up_axis=None для вимкнення
    автоматичного фліпу.
    """
    success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, camera_matrix, dist,
    flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        raise ValueError("solvePnP не зміг знайти рішення,"
                         " треба перевірити порядок точок")

    R, _ = cv2.Rodrigues(rvec)

    # Знаходимо фізичну позицію камери в залі (C = -R^T * tvec)
    R_inv = np.linalg.inv(R)
    camera_position = -np.dot(R_inv, tvec)

    # Дезамбіюація знаку вертикальної осі для калібрування за
    # копланарними точками підлоги. Якщо камера «опинилась під
    # підлогою» — застосовуємо віддзеркалення світу по up_axis:
    #     R'  = R  @ diag(±1, ±1, ±1)   (мінус саме у компоненті up_axis)
    #     C'  = diag(...) @ C
    # tvec не змінюється. Репроєкція точок з координатою up_axis = 0
    # (тобто всіх калібрувальних точок підлоги) лишається ідентичною.
    if (up_axis is not None
            and 0 <= up_axis < 3
            and camera_position[up_axis, 0] < 0.0):
        flip = np.eye(3)
        flip[up_axis, up_axis] = -1.0
        R = np.dot(R, flip)
        camera_position = np.dot(flip, camera_position)

    return R, tvec, camera_position

def get_3d_position(u, v, bbox_w, K, R_matrix, camera_pos,
    ball_diameter=0.21, z_min=2.0, z_max=25.0,
    return_none_on_clip=False):
    """
    Параметри:
    u, v: центр BBox м'яча в пікселях
    bbox_w: ширина BBox м'яча в пікселях (беремо ширину, бо вона менше
        страждає від Motion Blur, ніж висота)
    K: матриця камери (Intrinsic)
    R_matrix: матриця обертання з solvePnP
    camera_pos: 3D позиція камери (C) з solvePnP
    ball_diameter: діаметр волейбольного м'яча в метрах (стандарт 21 см)
    z_min, z_max: санітарні межі для Z_c (відстані від камери вздовж
        променя). Z_c обернено пропорційний до bbox_w (Z_c = f·D / bbox_w),
        тому крихітний bbox дає Z_c у десятки/сотні метрів, а гігантський
        bbox — < метра. PLAN A: 2.0 м (м'яч біля самої камери) — 25.0 м
        (діагональ майданчика + запас).
    return_none_on_clip: якщо True й Z_c поза [z_min, z_max] → повертає
        None (детекцію треба відкинути як ймовірний false positive). Якщо
        False (default, backward-compat) — повертає clamp'нуту позицію.

    Returns:
        np.ndarray shape (3,) — 3D-позиція [X, Y, Z] у світовій СК. Або
        None, якщо return_none_on_clip=True і Z_c вийшов за межі.
    """
    f_x = K[0, 0]
    c_x = K[0, 2]
    f_y = K[1, 1]
    c_y = K[1, 2]

    Z_c = (f_x * ball_diameter) / bbox_w

    # PLAN A: санітарний фільтр на дальність уздовж променя камери.
    # Маленький false-positive bbox (тінь гравця, плями розмиття) →
    # Z_c у 50–100 м → 3D-точка летить далеко за межі майданчика, та
    # її проєкція в наступних кадрах виглядає як "трек з кута екрана".
    if Z_c < z_min or Z_c > z_max:
        if return_none_on_clip:
            return None
        Z_c = max(z_min, min(z_max, Z_c))

    x_norm = (u - c_x) / f_x
    y_norm = (v - c_y) / f_y
    d_cam = np.array([[x_norm], [y_norm], [1.0]])

    d_world = np.dot(R_matrix.T, d_cam)
    P_world = camera_pos.reshape(3, 1) + Z_c * d_world

    return P_world.ravel()


def get_3d_position_with_cov(u, v, bbox_w, K, R_matrix, camera_pos,
    ball_diameter=0.21, sigma_w=1.0, sigma_uv=1.0, r_floor=0.02,
    z_min=2.0, z_max=25.0, return_none_on_clip=False):
    """
    Як get_3d_position, але додатково повертає 3×3 коваріацію вимірювання
    R у СВІТОВІЙ СК — анізотропну, фізично узгоджену з монокулярною
    геометрією глибини.

    Корінь невизначеності: глибина вздовж променя Z_c = f·D / w_box.
    Її похибка визначається похибкою ширини bbox σ_w через лінеаризацію:

        σ_Zc = |dZ_c/dw| · σ_w = (f·D / w_box²) · σ_w.

    σ_Zc росте ∝ f і ∝ 1/w² — тобто при коректнішій (більшій) фокусній
    та для дрібніших (далеких) м'ячів глибині треба довіряти МЕНШЕ.
    Це і є чесна заміна фіксованому R_z = 0.5.

    Невизначеність ЛЕЖИТЬ УЗДОВЖ ПРОМЕНЯ (напрямок d_world), а не вздовж
    світової осі Z. Тому world-коваріація — це переважно rank-1 еліпсоїд,
    витягнутий уздовж променя:

        Cov ≈ σ_Zc² · d_world·d_worldᵀ            (домінантний, глибинний)
            + Z_c² · σ_u² · J_u·J_uᵀ              (поперечний, від пікс. u)
            + Z_c² · σ_v² · J_v·J_vᵀ              (поперечний, від пікс. v)

    де J_u = R_matrixᵀ·[1/f_x, 0, 0]ᵀ, J_v = R_matrixᵀ·[0, 1/f_y, 0]ᵀ —
    напрямки зсуву точки при русі центра bbox на 1 px.

    Параметри:
        sigma_w:  σ ширини bbox у пікселях (шум детектора). Default 1.0.
        sigma_uv: σ центра bbox по u,v у пікселях. Default 1.0.
        r_floor:  ізотропний floor дисперсії (м²), що додається до всіх осей.
            Default 0.02 — узгоджено з латеральним членом старого фіксованого
            R = diag([0.02, 0.02, 0.5]). Гарантує, що латеральна довіра не
            колапсує нижче перевіреного значення (монокуляр локалізує вбік
            дуже точно → rank-1 терм лишає поперечні власні числа ~мікро), і
            забезпечує строгу позитивну визначеність. Адаптивним лишається
            ЛИШЕ глибина вздовж променя (σ_Zc² поверх floor).

    Returns:
        (P_world (3,), R_cov (3,3)) або (None, None) при кліпі з
        return_none_on_clip=True.
    """
    f_x = K[0, 0]
    c_x = K[0, 2]
    f_y = K[1, 1]
    c_y = K[1, 2]

    Z_c = (f_x * ball_diameter) / bbox_w

    if Z_c < z_min or Z_c > z_max:
        if return_none_on_clip:
            return None, None
        Z_c = max(z_min, min(z_max, Z_c))

    x_norm = (u - c_x) / f_x
    y_norm = (v - c_y) / f_y
    d_cam = np.array([[x_norm], [y_norm], [1.0]])

    d_world = np.dot(R_matrix.T, d_cam)              # (3,1) напрямок променя
    P_world = camera_pos.reshape(3, 1) + Z_c * d_world

    # σ глибини вздовж променя від похибки ширини bbox.
    sigma_Zc = (f_x * ball_diameter) / (bbox_w ** 2) * sigma_w

    # Поперечні якобіани (зсув точки при зсуві центра bbox на 1 px).
    j_u = np.dot(R_matrix.T, np.array([[1.0 / f_x], [0.0], [0.0]]))
    j_v = np.dot(R_matrix.T, np.array([[0.0], [1.0 / f_y], [0.0]]))

    R_cov = (sigma_Zc ** 2) * np.dot(d_world, d_world.T)
    R_cov += (Z_c ** 2) * (sigma_uv ** 2) * np.dot(j_u, j_u.T)
    R_cov += (Z_c ** 2) * (sigma_uv ** 2) * np.dot(j_v, j_v.T)

    # Ізотропний floor: тримає латеральну довіру на перевіреному рівні
    # старого R (0.02) і гарантує строгу позитивну визначеність. Адаптивним
    # лишається ЛИШЕ глибина вздовж променя (σ_Zc² поверх floor).
    R_cov += np.eye(3) * r_floor

    return P_world.ravel(), R_cov.astype(np.float64)


