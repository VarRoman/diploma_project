#!/usr/bin/env bash
# =============================================================================
#  run_pipeline.sh — the canonical end-to-end pipeline of the diploma project.
#
#  Video -> YOLO26 detection -> IMM/UKF 3D tracking (locked config) ->
#           game statistics -> overlay visualisation.
#
#  The tracking config was locked on 2026-06-04: depth-robust Gate A + a far-Z
#  detection ceiling at 21 m + a coast Z-fence against depth blow-up while
#  coasting. This script is the single entry point that reproduces the
#  reference result without typing ~20 flags by hand.
#
#  Usage:
#     scripts/run_pipeline.sh <video> [name] [t_start] [t_end]
#
#     <video>    path to the video file (required)
#     [name]     artifact prefix in runs/ (default: video name without ext.)
#     [t_start]  segment start, seconds (default: 0)
#     [t_end]    segment end, seconds (default: to the end)
#
#  Environment variables (optional):
#     PY      python interpreter    (default: conda env Volleyball_diploma)
#     MODEL   YOLO26 .pt weights    (default: main_model_april.pt)
#     TRAIL   overlay trail length  (default: 25)
#     NO_OVERLAY=1  skip the overlay render (tracking + statistics only)
# =============================================================================
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Вжиток: $0 <video> [name] [t_start] [t_end]" >&2
  exit 1
fi

# --- Paths relative to the script root (scripts/..) -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"          # ball_3d_analysis/
cd "$ROOT"

VIDEO="$1"
NAME="${2:-$(basename "${VIDEO%.*}")}"
T_START="${3:-0}"
T_END="${4:-}"

PY="${PY:-$HOME/miniconda3/envs/Volleyball_diploma/bin/python3}"
MODEL="${MODEL:-../training_models/models/main_model_april.pt}"
TRAIL="${TRAIL:-25}"

OUTDIR="runs"
mkdir -p "$OUTDIR"
TRACKS="$OUTDIR/${NAME}_tracks.jsonl"
STATS="$OUTDIR/${NAME}_stats.json"
OVERLAY="$OUTDIR/${NAME}_overlay.mp4"

# Optional time bounds.
TIME_ARGS=()
[[ -n "$T_START" ]] && TIME_ARGS+=(--t_start "$T_START")
[[ -n "$T_END"   ]] && TIME_ARGS+=(--t_end "$T_END")

echo "============================================================"
echo " ПАЙПЛАЙН: $NAME"
echo "   відео   : $VIDEO"
echo "   модель  : $MODEL"
echo "   сегмент : [${T_START}s, ${T_END:-кінець}]"
echo "============================================================"

# --- Stage 1: detection + 3D tracking (locked config) -----------------------
echo "[1/3] YOLO26 детекція + IMM/UKF 3D-трекінг ..."
"$PY" scripts/run_segment.py \
  --video "$VIDEO" \
  --model "$MODEL" \
  "${TIME_ARGS[@]}" \
  --out_jsonl "$TRACKS" \
  --conf 0.4 --iou 0.45 --imgsz 1920 \
  --max_age 80 --min_hits 3 --gating 16.0 \
  --floor_eps -0.5 --ceiling_max 15.0 --court_margin 10.0 \
  --court_z_far_max 21.0 \
  --max_assoc_residual 3.0 \
  --enable_adaptive_depth_R --sigma_w 1.5 --sigma_w_speed_gain 0.15 \
  --z_min 15.0 --z_max 60.0 --smooth_wbox 3 --reject_truncated 3.0 \
  --wbox_debias 0.099 --ball_diameter 0.27 \
  --enable_depth_robust_gate --depth_hold_gain 0.5 --depth_innov_free 3.0 \
  --enable_coast_z_fence --coast_z_min -3.0 --coast_z_max 21.0 \
  --save_calibration

# --- Stage 2: game statistics -----------------------------------------------
echo "[2/3] Просторово-часова статистика ..."
"$PY" scripts/compute_statistics.py --jsonl "$TRACKS" --out_json "$STATS"

# --- Stage 3: overlay visualisation -----------------------------------------
if [[ "${NO_OVERLAY:-0}" == "1" ]]; then
  echo "[3/3] Оверлей пропущено (NO_OVERLAY=1)."
else
  echo "[3/3] Рендер оверлею ..."
  "$PY" scripts/overlay_tracks.py \
    --video "$VIDEO" --jsonl "$TRACKS" --out "$OVERLAY" --trail "$TRAIL"
fi

echo "============================================================"
echo " ГОТОВО. Артефакти:"
echo "   трекінг   : $TRACKS"
echo "   статистика: $STATS"
[[ "${NO_OVERLAY:-0}" == "1" ]] || echo "   оверлей   : $OVERLAY"
echo "============================================================"
