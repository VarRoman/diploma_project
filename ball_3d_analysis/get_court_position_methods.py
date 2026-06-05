import cv2
import numpy as np
import torch as t
import torchvision
import torchaudio
import albumentations as A
from ultralytics import YOLO

# Reference calibration correspondences: the four court corners.
# Court is 9 m wide x 18 m long; world coordinates are in metres.
pts_real_3d = np.array([
    [0.0, 0.0, 0.0],
    [9.0, 0.0, 0.0],
    [9.0, 0.0, 18.0],
    [0.0, 0.0, 18.0]],dtype=np.float32)

# Pixel locations of the same four corners in the broadcast frame.
pts_video_2d = np.array([
    [  69, 1033],   # P0  front-left
    [1840, 1031],   # P1  front-right
    [1507,  609],   # P2  back-right
    [ 405,  609]],  # P3  back-left
    dtype=np.float32)

# Initial pinhole intrinsics (principal point at the image centre). The focal
# length is only a starting guess here — it is refined by calibration.
K = np.array([
    [1300.0, 0.0,    960.0],
    [0.0,    1300.0, 540.0],
    [0.0,    0.0,    1.0]],dtype=np.float32)


def calibrate_camera(pts_3d, pts_2d, camera_matrix, dist, up_axis=1):
    """
    Calibrate the camera with PnP. Returns (R, tvec, camera_position):
        R               — world->camera rotation matrix (3x3),
        tvec            — translation into the camera frame (3,1),
        camera_position — camera location in world coordinates (3,1).

    up_axis flips R so the camera ends up above the floor: the four coplanar
    court corners leave the vertical direction sign-ambiguous otherwise.
    """
    success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, camera_matrix, dist,
    flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        raise ValueError("solvePnP failed to converge; "
                         "check the order of the point correspondences")

    R, _ = cv2.Rodrigues(rvec)

    # Camera position in world coordinates (C = -R^T * tvec).
    R_inv = np.linalg.inv(R)
    camera_position = -np.dot(R_inv, tvec)

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
    Back-project a ball detection into a 3D world position.

    Args:
        u, v:          ball bbox centre, in pixels.
        bbox_w:        ball bbox width, in pixels. Width is used (not height)
                       because it suffers less from motion blur.
        K:             camera intrinsics matrix.
        R_matrix:      rotation matrix from solvePnP.
        camera_pos:    3D camera position (C) from solvePnP.
        ball_diameter: real volleyball diameter, in metres (21 cm nominal).
        z_min, z_max:  sanity bounds on Z_c (distance from camera along the ray).

    Returns:
        np.ndarray (3,) — world position [X, Y, Z]; or None when
        return_none_on_clip=True and Z_c fell outside [z_min, z_max].
    """
    f_x = K[0, 0]
    c_x = K[0, 2]
    f_y = K[1, 1]
    c_y = K[1, 2]

    Z_c = (f_x * ball_diameter) / bbox_w
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
    Like get_3d_position, but also returns a 3x3 measurement covariance R in
    world coordinates — anisotropic and physically consistent with the
    monocular depth geometry.

    Depth along the ray is Z_c = f*D / w_box, so its error is driven by the
    bbox-width error sigma_w through linearisation:

        sigma_Zc = |dZ_c/dw| * sigma_w = (f*D / w_box^2) * sigma_w.

    sigma_Zc grows with f and with 1/w^2 — i.e. depth deserves less trust for
    a larger (correct) focal length and for smaller (distant) balls. This is
    the honest replacement for a fixed R_z = 0.5.

    The uncertainty lies along the ray (direction d_world), not along the world
    Z axis, so the world covariance is mostly a rank-1 ellipsoid stretched
    along the ray:

        Cov ~ sigma_Zc^2 * d_world*d_world^T     (dominant, depth)
            + Z_c^2 * sigma_u^2 * J_u*J_u^T       (lateral, from pixel u)
            + Z_c^2 * sigma_v^2 * J_v*J_v^T       (lateral, from pixel v)

    with J_u = R^T*[1/f_x, 0, 0]^T, J_v = R^T*[0, 1/f_y, 0]^T — the shift of
    the point per 1 px move of the bbox centre.

    Args:
        sigma_w:  bbox-width sigma, in pixels (detector noise). Default 1.0.
        sigma_uv: bbox-centre sigma in u,v, in pixels. Default 1.0.
        r_floor:  isotropic variance floor (m^2) added to every axis.
            Default 0.02, matching the lateral term of the old fixed
            R = diag([0.02, 0.02, 0.5]). Keeps lateral trust from collapsing
            below the proven value (monocular localises sideways very well, so
            the rank-1 term leaves the lateral eigenvalues tiny) and guarantees
            positive-definiteness. Only the along-ray depth stays adaptive
            (sigma_Zc^2 on top of the floor).

    Returns:
        (P_world (3,), R_cov (3,3)); or (None, None) on clip when
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

    d_world = np.dot(R_matrix.T, d_cam)              # (3,1) ray direction
    P_world = camera_pos.reshape(3, 1) + Z_c * d_world

    # Depth sigma along the ray, from the bbox-width error.
    sigma_Zc = (f_x * ball_diameter) / (bbox_w ** 2) * sigma_w

    # Lateral Jacobians (point shift per 1 px move of the bbox centre).
    j_u = np.dot(R_matrix.T, np.array([[1.0 / f_x], [0.0], [0.0]]))
    j_v = np.dot(R_matrix.T, np.array([[0.0], [1.0 / f_y], [0.0]]))

    R_cov = (sigma_Zc ** 2) * np.dot(d_world, d_world.T)
    R_cov += (Z_c ** 2) * (sigma_uv ** 2) * np.dot(j_u, j_u.T)
    R_cov += (Z_c ** 2) * (sigma_uv ** 2) * np.dot(j_v, j_v.T)

    # Isotropic floor: keeps lateral trust at the proven level of the old R
    # (0.02) and guarantees positive-definiteness. Only the along-ray depth
    # stays adaptive (sigma_Zc^2 on top of the floor).
    R_cov += np.eye(3) * r_floor

    return P_world.ravel(), R_cov.astype(np.float64)


