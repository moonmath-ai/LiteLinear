#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
set -euo pipefail

# Run FFN calibration (capture R matrices) in the `ltx_radial` conda env.
#
# Defaults are intentionally small/safe; override via env vars:
#   PIPELINE_CONFIG=... MAX_PROMPTS=... BLOCKS=... WHICH=... CHECKPOINT_EVERY=...
#   SAVE_R=1
#   OUT=... CKPT=... RESUME=1
#
# Notes:
# - We default to WHICH=w1 to keep calibration costs reasonable. To capture both FFN linears, set:
#     WHICH=w1,w2
#   (or run twice, once per which, if you prefer separate output files).
# - The prompt file is expected at:
#     third_party/data/vidprom_filtered_extended.txt
#   (not committed due to size; see `ffn/capture_r.py` default).
#
# Example:
#   MAX_PROMPTS=200 BLOCKS=0-1 WHICH=w1 CHECKPOINT_EVERY=10 ./ffn/run_capture_r_ltx_radial.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Setup logging - redirect all output to both stdout and log file
LOG_FILE="${LOG_FILE:-$REPO_ROOT/ffn/r_capture.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

# Calibration knobs
PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/ltxv-13b-0.9.8-distilled.yaml}"
MAX_PROMPTS="${MAX_PROMPTS:-25000}"
BLOCKS="${BLOCKS:-0-1}"
WHICH="${WHICH:-w1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
SAVE_R="${SAVE_R:-0}"
RESUME="${RESUME:-0}"
NUM_FRAMES="${NUM_FRAMES:-17}"
GPU_ACCUM="${GPU_ACCUM:-1}" # Set to 1 to enable GPU accumulation
SEED="${SEED:-42}" # Random seed for deterministic prompt selection
ACCUM_DTYPE="${ACCUM_DTYPE:-}" # float64, float32, float16
ACCUM_DEVICE="${ACCUM_DEVICE:-}" # cpu, cuda, cuda:N

# Outputs (kept inside repo; ignored by git)
mkdir -p "$REPO_ROOT/ffn_delta_checkpoints" "$REPO_ROOT/ffn_delta_outputs"
OUT="${OUT:-$REPO_ROOT/ffn_delta_outputs/r_${WHICH}_b${BLOCKS}_n${MAX_PROMPTS}.pt}"
CKPT="${CKPT:-$REPO_ROOT/ffn_delta_checkpoints/r_${WHICH}_b${BLOCKS}_n${MAX_PROMPTS}.ckpt.pt}"

echo "[run_capture_r] repo=$REPO_ROOT"
echo "[run_capture_r] log_file=$LOG_FILE"
echo "[run_capture_r] pipeline_config=$PIPELINE_CONFIG"
echo "[run_capture_r] blocks=$BLOCKS which=$WHICH max_prompts=$MAX_PROMPTS"
echo "[run_capture_r] num_frames=$NUM_FRAMES gpu_accum=$GPU_ACCUM seed=$SEED"
echo "[run_capture_r] save_r=$SAVE_R"
echo "[run_capture_r] out=$OUT"
echo "[run_capture_r] ckpt=$CKPT checkpoint_every=$CHECKPOINT_EVERY resume=$RESUME"
echo

cd "$REPO_ROOT"

# Expect the user to activate the env manually (e.g. `conda activate ltx_radial`) before running.
python -c "import ltx_video" >/dev/null 2>&1 || {
  echo "ERROR: Python cannot import ltx_video. Activate your env first (e.g.: conda activate ltx_radial)." >&2
  exit 1
}

ARGS=(
  --no_compile
  --pipeline_config "$PIPELINE_CONFIG"
  --max_prompts "$MAX_PROMPTS"
  --blocks "$BLOCKS"
  --which "$WHICH"
  --out "$OUT"
  --checkpoint_path "$CKPT"
  --checkpoint_every "$CHECKPOINT_EVERY"
  --num_frames "$NUM_FRAMES"
  --seed "$SEED"
)

if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi

if [[ "$SAVE_R" == "1" ]]; then
  ARGS+=(--save_r)
fi

if [[ "$GPU_ACCUM" == "1" ]]; then
  ARGS+=(--gpu_accum)
fi

if [[ -n "$ACCUM_DTYPE" ]]; then
  ARGS+=(--accum_dtype "$ACCUM_DTYPE")
fi

if [[ -n "$ACCUM_DEVICE" ]]; then
  ARGS+=(--accum_device "$ACCUM_DEVICE")
fi

python ffn/capture_r.py "${ARGS[@]}"


