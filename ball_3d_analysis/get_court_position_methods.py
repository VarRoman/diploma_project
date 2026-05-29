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
    [  69, 1033],
    [1840, 1031],
    [1507,  609],
    [ 405,  609]],dtype=np.float32)

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
    ball_diameter=0.21):
    """
    Параметри:
    u, v: центр BBox м'яча в пікселях
    bbox_w: ширина BBox м'яча в пікселях (беремо ширину, бо вона менше
        страждає від Motion Blur, ніж висота)
    K: матриця камери (Intrinsic)
    R_matrix: матриця обертання з solvePnP
    camera_pos: 3D позиція камери (C) з solvePnP
    ball_diameter: діаметр волейбольного м'яча в метрах (стандарт 21 см)
    """
    f_x = K[0, 0]
    c_x = K[0, 2]
    f_y = K[1, 1]
    c_y = K[1, 2]

    Z_c = (f_x * ball_diameter) / bbox_w

    x_norm = (u - c_x) / f_x
    y_norm = (v - c_y) / f_y
    d_cam = np.array([[x_norm], [y_norm], [1.0]])

    d_world = np.dot(R_matrix.T, d_cam)
    P_world = camera_pos.reshape(3, 1) + Z_c * d_world

    return P_world.ravel()


