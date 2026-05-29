#!/usr/bin/env python3
"""
visualize_segment.py — візуалізація результату run_segment.py.

Читає JSONL із діагностикою (header + per-frame frame-рядки + summary),
відновлює часові ряди для домінантного треку та будує:

  1. PNG-діаграму з 9 панелями (matplotlib):
       X(t), Y(t), Z(t),  vx(t), vy(t), vz(t),
       μ_ballistic/hit/bounce(t), |residual|(t), Mahalanobis(t).
     Сирі детекції (raw_3d) накладаються на X/Y/Z як scatter, фільтровані
     стани (x_post) — суцільною лінією. Прогалини у детекціях лишаються
     прогалинами (NaN) — щоб було видно, де трекер екстраполює.

  2. Plotly-HTML з інтерактивною 3D-сценою:
       — сира 3D-траєкторія (scatter, маркер «.», прозорий)
       — фільтрована 3D-траєкторія (lines, кольорено по часу)
       — підлога-майданчик (9 м × 18 м, Y=0) як тонка квадратна сітка
       — позначка позиції камери (з calibration у header, якщо є)

Вибір «домінантного» треку: за замовчуванням — той, у якого найбільша
к-сть hits протягом усього сегмента. Можна форсувати через --track_id.

Приклад:
    python scripts/visualize_segment.py \\
        --jsonl logs/seg_ultrashort_full.jsonl \\
        --out_png logs/seg_ultrashort_full.png \\
        --out_html logs/seg_ultrashort_full.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")  # без X-сервера — лише PNG.
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import plotly.graph_objects as go


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Візуалізатор JSONL-виходу run_segment.py.",
    )
    p.add_argument("--jsonl", required=True, type=Path,
                   help="Вхідний JSONL з діагностикою.")
    p.add_argument("--out_png", required=True, type=Path,
                   help="Шлях до вихідного PNG (9 панелей).")
    p.add_argument("--out_html", required=True, type=Path,
                   help="Шлях до вихідного HTML (Plotly 3D-сцена).")
    p.add_argument("--track_id", type=int, default=None,
                   help="Якщо вказано — будувати лише цей track_id. "
                        "Інакше беремо домінантний (макс. hits).")
    p.add_argument("--show_all_tracks", action="store_true",
                   help="У 3D-сцені малювати ВСІ треки, не лише "
                        "домінантний.")
    return p.parse_args()


# ----------------------------------------------------------------------
def load_jsonl(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]],
                                     Optional[Dict[str, Any]]]:
    """Повертає (header, frames, summary). summary може бути None."""
    header: Optional[Dict[str, Any]] = None
    frames: List[Dict[str, Any]] = []
    summary: Optional[Dict[str, Any]] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get("type")
            if t == "header":
                header = rec
            elif t == "frame":
                frames.append(rec)
            elif t == "summary":
                summary = rec
    if header is None:
        raise ValueError(f"JSONL без header у {path}")
    return header, frames, summary


def pick_dominant_track(frames: List[Dict[str, Any]]) -> Optional[int]:
    """Track із максимальною к-стю hits-моментів (поява у списку tracks)."""
    counts: Dict[int, int] = {}
    confirmed_counts: Dict[int, int] = {}
    for fr in frames:
        for t in fr.get("tracks", []):
            tid = int(t["track_id"])
            counts[tid] = counts.get(tid, 0) + 1
            if t.get("state") == "Confirmed":
                confirmed_counts[tid] = confirmed_counts.get(tid, 0) + 1
    if not counts:
        return None
    # Перевага трекам, які встигли стати Confirmed.
    if confirmed_counts:
        return max(confirmed_counts.keys(),
                   key=lambda tid: (confirmed_counts[tid], counts[tid]))
    return max(counts.keys(), key=lambda tid: counts[tid])


def extract_track_series(frames: List[Dict[str, Any]],
                          track_id: int) -> Dict[str, np.ndarray]:
    """
    Витягуємо часові ряди для конкретного track_id. Кадри, в яких треку
    нема, заповнюємо NaN — щоб у matplotlib з'явилися прогалини.

    Стан-вектор у фільтрі: [x, vx, y, vy, z, vz].
    """
    n = len(frames)
    t = np.full(n, np.nan, dtype=float)
    frame_idx = np.zeros(n, dtype=int)
    raw_x = np.full(n, np.nan); raw_y = np.full(n, np.nan); raw_z = np.full(n, np.nan)
    post_x = np.full(n, np.nan); post_vx = np.full(n, np.nan)
    post_y = np.full(n, np.nan); post_vy = np.full(n, np.nan)
    post_z = np.full(n, np.nan); post_vz = np.full(n, np.nan)
    P_diag = np.full((n, 6), np.nan)
    mu_ball = np.full(n, np.nan); mu_hit = np.full(n, np.nan); mu_bnc = np.full(n, np.nan)
    mahal = np.full(n, np.nan)
    res_norm = np.full(n, np.nan)
    res_x = np.full(n, np.nan); res_y = np.full(n, np.nan); res_z = np.full(n, np.nan)
    state_str: List[Optional[str]] = [None] * n
    hits = np.full(n, np.nan)

    for i, fr in enumerate(frames):
        t[i] = fr["t"]
        frame_idx[i] = fr["frame"]
        rd = fr.get("raw_3d")
        if rd is not None:
            raw_x[i] = rd[0]; raw_y[i] = rd[1]; raw_z[i] = rd[2]
        for tr in fr.get("tracks", []):
            if int(tr["track_id"]) == track_id:
                xp = tr["x_post"]
                post_x[i], post_vx[i] = xp[0], xp[1]
                post_y[i], post_vy[i] = xp[2], xp[3]
                post_z[i], post_vz[i] = xp[4], xp[5]
                pd = tr.get("P_post_diag")
                if pd is not None:
                    P_diag[i] = pd
                mu = tr.get("mu")
                if mu is not None and len(mu) == 3:
                    mu_ball[i], mu_hit[i], mu_bnc[i] = mu
                m = tr.get("mahalanobis")
                if m is not None:
                    mahal[i] = m
                r = tr.get("residual")
                if r is not None:
                    res_x[i], res_y[i], res_z[i] = r
                    res_norm[i] = float(np.linalg.norm(r))
                state_str[i] = tr.get("state")
                if tr.get("hits") is not None:
                    hits[i] = tr["hits"]
                break

    return dict(
        t=t, frame_idx=frame_idx,
        raw_x=raw_x, raw_y=raw_y, raw_z=raw_z,
        post_x=post_x, post_vx=post_vx,
        post_y=post_y, post_vy=post_vy,
        post_z=post_z, post_vz=post_vz,
        P_diag=P_diag,
        mu_ball=mu_ball, mu_hit=mu_hit, mu_bnc=mu_bnc,
        mahal=mahal,
        res_x=res_x, res_y=res_y, res_z=res_z, res_norm=res_norm,
        state=state_str, hits=hits,
    )


def list_all_track_ids(frames: List[Dict[str, Any]]) -> List[int]:
    ids = set()
    for fr in frames:
        for tr in fr.get("tracks", []):
            ids.add(int(tr["track_id"]))
    return sorted(ids)


# ----------------------------------------------------------------------
def render_png(series: Dict[str, np.ndarray],
                header: Dict[str, Any],
                track_id: int,
                out_path: Path) -> None:
    """9-панельна діаграма."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    t = series["t"]

    # --- Ряд 1: позиції X, Y, Z ---
    for ax, label, raw, post in (
        (axes[0, 0], "X (м)", series["raw_x"], series["post_x"]),
        (axes[0, 1], "Y — висота (м)", series["raw_y"], series["post_y"]),
        (axes[0, 2], "Z (м)", series["raw_z"], series["post_z"]),
    ):
        ax.scatter(t, raw, s=8, alpha=0.45, color="#c0392b",
                   label="raw 3D", zorder=2)
        ax.plot(t, post, color="#2980b9", lw=1.5,
                label="IMM x_post", zorder=3)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    # --- Ряд 2: швидкості vx, vy, vz ---
    for ax, label, vel in (
        (axes[1, 0], "vx (м/с)", series["post_vx"]),
        (axes[1, 1], "vy (м/с)", series["post_vy"]),
        (axes[1, 2], "vz (м/с)", series["post_vz"]),
    ):
        ax.plot(t, vel, color="#16a085", lw=1.4)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color="k", lw=0.5, alpha=0.4)

    # --- Ряд 3: ваги IMM, |residual|, Махаланобіс ---
    ax_mu = axes[2, 0]
    ax_mu.plot(t, series["mu_ball"], color="#27ae60", lw=1.3,
               label="μ ballistic")
    ax_mu.plot(t, series["mu_hit"], color="#c0392b", lw=1.3,
               label="μ hit")
    ax_mu.plot(t, series["mu_bnc"], color="#8e44ad", lw=1.3,
               label="μ bounce")
    ax_mu.set_ylim(-0.05, 1.05)
    ax_mu.set_ylabel("IMM ваги")
    ax_mu.legend(loc="upper right", fontsize=8)
    ax_mu.grid(True, alpha=0.3)

    ax_res = axes[2, 1]
    ax_res.plot(t, series["res_norm"], color="#e67e22", lw=1.3)
    ax_res.set_ylabel("|residual| (м)")
    ax_res.grid(True, alpha=0.3)

    ax_mah = axes[2, 2]
    ax_mah.plot(t, series["mahal"], color="#34495e", lw=1.3)
    # Поріг гейтингу (sqrt тому що mahalanobis — це √(d²)).
    thr = header.get("tracker", {}).get("gating_threshold")
    if thr is not None:
        ax_mah.axhline(np.sqrt(thr), color="#c0392b",
                       linestyle="--", lw=1.0,
                       label=f"√gating={np.sqrt(thr):.2f}")
        ax_mah.legend(loc="upper right", fontsize=8)
    ax_mah.set_ylabel("Mahalanobis d")
    ax_mah.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("t (с)")

    # Шапка
    n_raw = int(np.sum(~np.isnan(series["raw_x"])))
    n_filt = int(np.sum(~np.isnan(series["post_x"])))
    frame_range = header.get("frame_range", [None, None])
    video = Path(header.get("video", "")).name
    fig.suptitle(
        f"Сегмент: {video}  "
        f"frames=[{frame_range[0]}, {frame_range[1]})  "
        f"FPS={header.get('fps')}  dt={header.get('dt'):.5f}s\n"
        f"track_id={track_id}  "
        f"raw_3d points={n_raw}  filtered points={n_filt}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[png] -> {out_path}")


# ----------------------------------------------------------------------
def render_html(header: Dict[str, Any],
                 frames: List[Dict[str, Any]],
                 dom_track_id: int,
                 all_tracks: bool,
                 out_path: Path) -> None:
    """Інтерактивна 3D-сцена."""
    fig = go.Figure()

    # Контур майданчика 9×18 на Y=0 (X-Z плоскість).
    court_x = [0, 9, 9, 0, 0]
    court_z = [0, 0, 18, 18, 0]
    court_y = [0, 0, 0, 0, 0]
    fig.add_trace(go.Scatter3d(
        x=court_x, y=court_y, z=court_z,
        mode="lines",
        line=dict(color="#7f8c8d", width=3),
        name="Майданчик 9×18 м",
    ))
    # Лінія сітки навпіл (X=4.5).
    fig.add_trace(go.Scatter3d(
        x=[4.5, 4.5, 4.5, 4.5], y=[0, 2.43, 2.43, 0],
        z=[0, 0, 18, 18],
        mode="lines",
        line=dict(color="#bdc3c7", width=2, dash="dash"),
        name="Сітка (2.43 м)",
    ))

    # Камера, якщо є.
    calib = header.get("calibration")
    if calib is not None and "camera_pos" in calib:
        cp = calib["camera_pos"]
        fig.add_trace(go.Scatter3d(
            x=[cp[0]], y=[cp[1]], z=[cp[2]],
            mode="markers+text",
            marker=dict(size=8, color="#34495e", symbol="diamond"),
            text=["camera"],
            textposition="top center",
            name=f"Camera ({cp[0]:.1f}, {cp[1]:.1f}, {cp[2]:.1f})",
        ))

    # Сира 3D-траєкторія (всі raw_3d).
    raw_xs = [fr.get("raw_3d", [None]*3)[0]
              for fr in frames if fr.get("raw_3d") is not None]
    raw_ys = [fr.get("raw_3d", [None]*3)[1]
              for fr in frames if fr.get("raw_3d") is not None]
    raw_zs = [fr.get("raw_3d", [None]*3)[2]
              for fr in frames if fr.get("raw_3d") is not None]
    raw_ts = [fr["t"]
              for fr in frames if fr.get("raw_3d") is not None]
    if raw_xs:
        fig.add_trace(go.Scatter3d(
            x=raw_xs, y=raw_ys, z=raw_zs,
            mode="markers",
            marker=dict(size=2.6, color=raw_ts, colorscale="Reds",
                        opacity=0.7,
                        colorbar=dict(title="raw t (с)", x=1.10, len=0.4)),
            name=f"raw 3D (n={len(raw_xs)})",
        ))

    # Фільтрована траєкторія.
    track_ids = (list_all_track_ids(frames) if all_tracks
                 else [dom_track_id])
    colorscales = ["Blues", "Greens", "Purples", "Oranges", "Greys"]
    for k, tid in enumerate(track_ids):
        series = extract_track_series(frames, tid)
        mask = ~np.isnan(series["post_x"])
        if mask.sum() < 2:
            continue
        fig.add_trace(go.Scatter3d(
            x=series["post_x"][mask],
            y=series["post_y"][mask],
            z=series["post_z"][mask],
            mode="lines+markers",
            line=dict(width=4),
            marker=dict(size=2.5,
                        color=series["t"][mask],
                        colorscale=colorscales[k % len(colorscales)],
                        showscale=(k == 0),
                        colorbar=(dict(title=f"track #{tid} t (с)",
                                        x=1.02, len=0.4)
                                   if k == 0 else None)),
            name=f"track #{tid} ({int(mask.sum())} pts)",
        ))

    video = Path(header.get("video", "")).name
    fig.update_layout(
        title=(f"3D-траєкторія м'яча — {video}<br>"
                f"<sub>frames={header.get('frame_range')} "
                f"FPS={header.get('fps')} "
                f"track={'all' if all_tracks else dom_track_id}</sub>"),
        scene=dict(
            xaxis=dict(title="X (м, поперек майданчика)", range=[-2, 11]),
            yaxis=dict(title="Y (м, висота)", range=[0, 8]),
            zaxis=dict(title="Z (м, вздовж майданчика)", range=[-3, 20]),
            aspectmode="data",
        ),
        legend=dict(yanchor="top", y=0.98, x=0.01),
        margin=dict(l=0, r=0, t=80, b=0),
        height=750,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"[html] -> {out_path}")


# ----------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    header, frames, summary = load_jsonl(args.jsonl)

    if not frames:
        sys.stderr.write("[ERR] У JSONL нема жодного frame-рядка.\n")
        return 2

    dom_id = args.track_id
    if dom_id is None:
        dom_id = pick_dominant_track(frames)
        if dom_id is None:
            sys.stderr.write(
                "[ERR] У сегменті не знайдено жодного треку.\n"
            )
            return 2
        sys.stderr.write(f"[i] Домінантний track_id = {dom_id}\n")

    series = extract_track_series(frames, dom_id)
    n_filt = int(np.sum(~np.isnan(series["post_x"])))
    if n_filt < 2:
        sys.stderr.write(
            f"[w] track #{dom_id} має лише {n_filt} fitted-точок; "
            f"графіки можуть бути порожніми.\n"
        )

    render_png(series, header, dom_id, args.out_png)
    render_html(header, frames, dom_id, args.show_all_tracks, args.out_html)

    if summary is not None:
        print(f"[summary] {summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
