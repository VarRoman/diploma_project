#!/usr/bin/env python3
"""
Phase F — F.2: тест СПОСТЕРЕЖНОСТІ монокулярної глибини.

Питання go/no-go: чи дає піксельна вимірювальна модель + відома гравітація
(захардкоджена driving-term у fx_ballistic) спостережну АБСОЛЮТНУ глибину Z
для далекого м'яча, де cue розміру (Z_c=f·D/w) ненадійний?

Метод (чесний — фізика моделі = фізика «істини», тестуємо спостережність,
не model-mismatch):
  1. Генеруємо ground-truth балістичну дугу ТОЮ Ж fx_ballistic (g + drag).
  2. Проєктуємо кожну 3D-точку у піксель (u,v) + camera-frame глибину Z_c,
     рахуємо «істинний» bbox_w = f·D/Z_c.
  3. Додаємо шум: σ_uv (пікселі), σ_w (ширина боксу).
  4. Будуємо ГІБРИДНИЙ вимір z=[u, v, Z_c_size], де Z_c_size=f·D/w_noisy має
     АДАПТИВНИЙ шум σ_Z=(f·D/w²)·σ_w (на дистанції сам згасає).
  5. UKF (вендорений) із hx=проєкція, fx=fx_ballistic, анізотропна P0.
  6. Порівнюємо: filter-Z vs GT-Z vs size-only-Z (baseline).

НЕ RTS. Це форвард-фільтр-тест спостережності.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Імпорт вендореного UKF + фізики. get_court_position_methods тягне cv2/torch —
# важко, але це наш реальний код, тестуємо саме його.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from IMM_UKF import (  # noqa: E402
    UnscentedKalmanFilter,
    MerweScaledSigmaPoints,
    fx_ballistic,
    Q_discrete_white_noise,
)
from scipy.linalg import block_diag  # noqa: E402

BALL_D = 0.21  # діаметр волейбольного м'яча, м

# Сценарії [x,vx,ax, y,vy,ay, z,vz,az]: ДАЛЕКІ м'ячі (Z≈20, cue розміру слабкий).
SCENARIOS = {
    'FAR_serve (Z~16-20, vy=9 high curve)':
        [6.0, 0.0, 0.0,  2.0, 9.0, 0.0,  20.0, -3.0, 0.0],
    'FAR_lowcurve (Z~16-20, vy=0.5 flat)':
        [6.0, 0.0, 0.0,  3.0, 0.5, 0.0,  20.0, -3.0, 0.0],
    'NEAR_serve (Z~4-7, vy=8)':
        [4.0, 0.0, 0.0,  1.5, 8.0, 0.0,  4.0, 2.0, 0.0],
}


# ----------------------------------------------------------------------
# Калібрування — беремо з реального header лога (консистентність)
# ----------------------------------------------------------------------
def load_calib(jsonl_path):
    with open(jsonl_path) as f:
        header = json.loads(f.readline())
    cal = header["calibration"]
    K = np.asarray(cal["K"], float)
    R = np.asarray(cal["R"], float)
    tvec = np.asarray(cal["tvec"], float).reshape(3)
    cam = np.asarray(cal["camera_pos"], float).reshape(3)
    return K, R, tvec, cam


# ----------------------------------------------------------------------
# Проєкція 3D(світ) → camera-frame та піксель (ручна, як в overlay_video)
# ----------------------------------------------------------------------
def world_to_cam(P, R, tvec):
    return R.dot(P) + tvec


def cam_to_pixel(p_cam, K):
    Zc = p_cam[2]
    u = K[0, 0] * p_cam[0] / Zc + K[0, 2]
    v = K[1, 1] * p_cam[1] / Zc + K[1, 2]
    return u, v, Zc


# ----------------------------------------------------------------------
# GT-траєкторія тим самим fx_ballistic
# ----------------------------------------------------------------------
def gen_gt(x0, dt, n):
    xs = np.zeros((n, 9))
    x = np.asarray(x0, float).copy()
    for k in range(n):
        xs[k] = x
        x = fx_ballistic(x, dt)
    return xs


# ----------------------------------------------------------------------
# Анізотропна початкова коваріація позиції: велика вздовж променя, мала поперек
# ----------------------------------------------------------------------
def anisotropic_P0(u, v, K, R, sigma_perp, sigma_ray):
    xn = (u - K[0, 2]) / K[0, 0]
    yn = (v - K[1, 2]) / K[1, 1]
    d_cam = np.array([xn, yn, 1.0])
    d_world = R.T.dot(d_cam)
    d_hat = d_world / np.linalg.norm(d_world)
    ddt = np.outer(d_hat, d_hat)
    P_pos = sigma_perp**2 * (np.eye(3) - ddt) + sigma_ray**2 * ddt
    return P_pos, d_hat


def backproject(u, v, Zc, K, R, cam):
    """Світова точка з пікселя + camera-frame глибини (як get_3d_position)."""
    xn = (u - K[0, 2]) / K[0, 0]
    yn = (v - K[1, 2]) / K[1, 1]
    d_cam = np.array([xn, yn, 1.0])
    d_world = R.T.dot(d_cam)
    return cam + Zc * d_world


def run_filter(meas, K, R, tvec, cam, dt, sigma_uv, sigma_w,
               sigma_perp, sigma_ray, q_var, no_size=False):
    """
    meas: list of dict {u,v,w,Zc_size,sigma_Z} (зашумлені виміри).
    no_size=True → ЧИСТА фізика: вимір лише [u,v] (dim_z=2), розмірний
        канал вимкнено. Тест: чи спостережна глибина від пікселів+гравітації
        САМОСТІЙНО, без cue розміру.
    Повертає масив оцінених станів (n,9).
    """
    n = len(meas)
    dim_x = 9
    dim_z = 2 if no_size else 3

    # hx: гібрид [u, v, Zc(camera-frame)] або [u, v] (no_size)
    def hx(x):
        P = np.array([x[0], x[3], x[6]])
        p_cam = world_to_cam(P, R, tvec)
        Zc = p_cam[2]
        if Zc <= 1e-6:
            Zc = 1e-6
        u = K[0, 0] * p_cam[0] / Zc + K[0, 2]
        v = K[1, 1] * p_cam[1] / Zc + K[1, 2]
        if no_size:
            return np.array([u, v])
        return np.array([u, v, Zc])

    points = MerweScaledSigmaPoints(n=dim_x, alpha=.1, beta=2., kappa=1.)
    ukf = UnscentedKalmanFilter(name='pix', dim_x=dim_x, dim_z=dim_z, dt=dt,
                                fx=fx_ballistic, hx=hx, points=points)
    q = Q_discrete_white_noise(dim=3, dt=dt, var=q_var)
    ukf.Q = block_diag(q, q, q)

    # --- ініціалізація з перших двох вимірів ---
    m0, m1 = meas[0], meas[1]
    p0 = backproject(m0['u'], m0['v'], m0['Zc_size'], K, R, cam)
    p1 = backproject(m1['u'], m1['v'], m1['Zc_size'], K, R, cam)
    P_pos, d_hat = anisotropic_P0(m0['u'], m0['v'], K, R, sigma_perp, sigma_ray)
    # Швидкість уздовж променя (по глибині) з двох зашумлених глибин-з-розміру
    # НЕСПОСТЕРЕЖНА (поділ ~5м похибки на dt=0.02 → ~250 м/с). Лишаємо лише
    # ПОПЕРЕЧНУ складову (її задає піксельний рух, добре визначена), а вздовж
    # променя зануляємо — гравітація+кривизна знайдуть її, велика P_vel покриває.
    v0_raw = (p1 - p0) / dt
    v0 = v0_raw - np.dot(v0_raw, d_hat) * d_hat
    ukf.x = np.array([p0[0], v0[0], 0.0,
                      p0[1], v0[1], 0.0,
                      p0[2], v0[2], 0.0])

    P0 = np.diag([0.1, 50.0, 25.0, 0.1, 50.0, 25.0, 0.1, 50.0, 25.0])
    idx = [0, 3, 6]
    P0[np.ix_(idx, idx)] = P_pos
    ukf.P = P0

    est = np.zeros((n, 9))
    for k in range(n):
        if k > 0:
            ukf.predict(dt=dt)
        mk = meas[k]
        if no_size:
            z = np.array([mk['u'], mk['v']])
            Rk = np.diag([sigma_uv**2, sigma_uv**2])
        else:
            z = np.array([mk['u'], mk['v'], mk['Zc_size']])
            Rk = np.diag([sigma_uv**2, sigma_uv**2, mk['sigma_Z']**2])
        ukf.update(z, R=Rk)
        est[k] = ukf.x.copy()
    return est


def eval_once(x0, args, K, R, tvec, cam, rng):
    """Одна реалізація шуму: повертає (gt_Z, filt_Z, size_Z, conv, rmse_f, rmse_s)."""
    gt = gen_gt(x0, args.dt, args.n)
    meas = []
    size_only_Z = np.zeros(args.n)
    for k in range(args.n):
        P = gt[k, [0, 3, 6]]
        p_cam = world_to_cam(P, R, tvec)
        u, v, Zc = cam_to_pixel(p_cam, K)
        w_true = K[0, 0] * BALL_D / Zc
        u_n = u + rng.normal(0, args.sigma_uv)
        v_n = v + rng.normal(0, args.sigma_uv)
        w_n = max(1.0, w_true + rng.normal(0, args.sigma_w))
        Zc_size = K[0, 0] * BALL_D / w_n
        sigma_Z = (K[0, 0] * BALL_D / w_n**2) * args.sigma_w
        meas.append(dict(u=u_n, v=v_n, w=w_n, Zc_size=Zc_size, sigma_Z=sigma_Z))
        size_only_Z[k] = backproject(u_n, v_n, Zc_size, K, R, cam)[2]

    est = run_filter(meas, K, R, tvec, cam, args.dt, args.sigma_uv,
                     args.sigma_w, args.sigma_perp, args.sigma_ray, args.q_var,
                     no_size=getattr(args, 'no_size', False))
    gt_Z, filt_Z = gt[:, 6], est[:, 6]
    err_f, err_s = filt_Z - gt_Z, size_only_Z - gt_Z
    conv = None
    for k in range(args.n):
        if np.all(np.abs(err_f[k:]) < 1.0):
            conv = k
            break
    tail = slice(args.n - 20, args.n)
    rmse_f = float(np.sqrt(np.mean(err_f[tail]**2)))
    rmse_s = float(np.sqrt(np.mean(err_s[tail]**2)))
    return gt_Z, filt_Z, size_only_Z, conv, rmse_f, rmse_s


def aggregate(args, K, R, tvec, cam):
    """Агрегує по n_seeds: mean/median RMSE + % збіжності per сценарій."""
    print(f"[aggregate] n_seeds={args.n_seeds} n={args.n} dt={args.dt} "
          f"q_var={args.q_var} sigma_uv={args.sigma_uv} sigma_w={args.sigma_w} "
          f"sigma_ray={args.sigma_ray}")
    for title, x0 in SCENARIOS.items():
        rmses_f, rmses_s, convs = [], [], []
        for s in range(args.n_seeds):
            rng = np.random.default_rng(1000 + s)
            _, _, _, conv, rf, rs = eval_once(x0, args, K, R, tvec, cam, rng)
            rmses_f.append(rf); rmses_s.append(rs)
            convs.append(conv if conv is not None else -1)
        rf = np.array(rmses_f); rs = np.array(rmses_s)
        conv_arr = np.array(convs)
        frac_conv = float(np.mean(conv_arr >= 0))
        conv_t = conv_arr[conv_arr >= 0] * args.dt
        print(f"\n=== {title} ===")
        print(f"  RMSE_Z filter:   mean={rf.mean():.2f}  median={np.median(rf):.2f}  "
              f"max={rf.max():.2f} м")
        print(f"  RMSE_Z size-only: mean={rs.mean():.2f}  median={np.median(rs):.2f} м")
        print(f"  покращення (mean): x{rs.mean()/max(rf.mean(),1e-6):.1f}")
        print(f"  збіжність |err|<1м: {frac_conv*100:.0f}% сідів"
              + (f", median час {np.median(conv_t):.2f}с" if len(conv_t) else ""))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', type=Path,
                    default=Path('logs/seg_defects12.jsonl'),
                    help='лог для зчитування калібрування')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sigma_uv', type=float, default=2.0, help='px шум центру')
    ap.add_argument('--sigma_w', type=float, default=2.0, help='px шум ширини')
    ap.add_argument('--q_var', type=float, default=0.1, help='ballistic jerk var')
    ap.add_argument('--sigma_perp', type=float, default=0.1)
    ap.add_argument('--sigma_ray', type=float, default=10.0)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--dt', type=float, default=0.02)
    ap.add_argument('--n_seeds', type=int, default=1,
                    help='якщо >1 — агрегує по сідах (mean RMSE + % збіжності)')
    ap.add_argument('--no_size', action='store_true',
                    help='вимкнути розмірний канал: чиста фізика піксель+гравітація')
    args = ap.parse_args()

    K, R, tvec, cam = load_calib(args.jsonl)
    print(f"[calib] cam_pos={cam.round(2)}  K_f={K[0,0]:.0f}")
    if args.n_seeds > 1:
        return aggregate(args, K, R, tvec, cam)

    rng = np.random.default_rng(args.seed)
    for title, x0 in SCENARIOS.items():
        gt_Z, filt_Z, size_only_Z, conv, rmse_filt, rmse_size = eval_once(
            x0, args, K, R, tvec, cam, rng)
        gt = gen_gt(x0, args.dt, args.n)
        err_filt = filt_Z - gt_Z
        err_size = size_only_Z - gt_Z
        print(f"\n=== {title} ===")
        print(f"  GT Z: {gt_Z[0]:.1f} -> {gt_Z[-1]:.1f} м | "
              f"camera-depth ~{world_to_cam(gt[0,[0,3,6]],R,tvec)[2]:.1f}-"
              f"{world_to_cam(gt[-1,[0,3,6]],R,tvec)[2]:.1f} м | "
              f"bbox_w ~{K[0,0]*BALL_D/world_to_cam(gt[-1,[0,3,6]],R,tvec)[2]:.1f}px")
        print(f"  сходження |err|<1м: "
              f"{'кадр '+str(conv)+' ('+f'{conv*args.dt:.2f}с)' if conv is not None else 'НЕ зійшлось'}")
        print(f"  RMSE Z (останні 20 кадрів): "
              f"filter={rmse_filt:.2f} м  vs  size-only={rmse_size:.2f} м  "
              f"=> покращення x{rmse_size/max(rmse_filt,1e-6):.1f}")
        print("  k |  GT_Z | filt_Z | size_Z | err_filt err_size")
        for k in list(range(0, args.n, max(1, args.n // 12))):
            print(f"  {k:2d} | {gt_Z[k]:5.2f} | {filt_Z[k]:6.2f} | "
                  f"{size_only_Z[k]:6.2f} | {err_filt[k]:+6.2f}  {err_size[k]:+6.2f}")


if __name__ == '__main__':
    main()
