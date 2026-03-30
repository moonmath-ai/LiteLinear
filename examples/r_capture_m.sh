#!/usr/bin/env bash
set -euo pipefail

# Run distributed FFN calibration (capture R matrices) in the `ltx_radial` conda env.
#
# Defaults are intentionally small/safe; override via env vars:
#   PIPELINE_CONFIG=... MAX_PROMPTS=... BLOCKS=... WHICH=... CHECKPOINT_EVERY=...
#   REDUCE_EVERY=... SAVE_R=1
#   OUT=... OUT_RSQRT=... OUT_RSQRT_INV=... CKPT=... RESUME=1
#   OUT_DIR=... OUT_RSQRT_DIR=... OUT_RSQRT_INV_DIR=...
#   GPU_ACCUM=1 NUM_GPUS=... CUDA_VISIBLE_DEVICES=...
#
# Notes:
# - We default to WHICH=w1 to keep calibration costs reasonable. To capture both FFN linears, set:
#     WHICH=w1,w2
#   (or run twice, once per which, if you prefer separate output files).
# - The prompt file is expected at:
#     third_party/data/vidprom_filtered_extended.txt
#   (not committed due to size; see `ffn/capture_r_ddp.py` default).
#
# Example:
#   CUDA_VISIBLE_DEVICES=4,5,6 MAX_PROMPTS=200 BLOCKS=0-1 WHICH=w1 SEED=42 ./ffn/r_capture_m.sh
#   (NUM_GPUS is auto-detected from CUDA_VISIBLE_DEVICES if not set)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Setup logging - redirect all output to both stdout and log file
LOG_FILE="${LOG_FILE:-$REPO_ROOT/ffn/r_capture_m.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

# Calibration knobs
PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/ltxv-13b-0.9.8-distilled.yaml}"
MAX_PROMPTS="${MAX_PROMPTS:-25000}"
BLOCKS="${BLOCKS:-0-47}"
WHICH="${WHICH:-w1,w2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-}"
REDUCE_EVERY="${REDUCE_EVERY:-0}"
SAVE_R="${SAVE_R:-0}"
RESUME="${RESUME:-0}"
NUM_FRAMES="${NUM_FRAMES:-8}"
GPU_ACCUM="${GPU_ACCUM:-1}" # Set to 1 to enable GPU accumulation
ACCUM_DTYPE="${ACCUM_DTYPE:-}" # float64, float32, float16
ACCUM_DEVICE="${ACCUM_DEVICE:-}" # cpu, cuda, cuda:N
ACCUM_DEVICE_RANK0_ONLY="${ACCUM_DEVICE_RANK0_ONLY:-0}"
RANK0_ACCUM_ONLY="${RANK0_ACCUM_ONLY:-0}"
SKIP_RSQRT="${SKIP_RSQRT:-0}"
RSQRT_DEVICE="${RSQRT_DEVICE:-cuda:0}"

if [[ -z "${NUM_GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_GPUS="${#DEVICES[@]}"
  else
    NUM_GPUS=1
  fi
fi

INFER_RANKS="$NUM_GPUS"
if [[ "$RANK0_ACCUM_ONLY" == "1" && "$NUM_GPUS" -gt 1 ]]; then
  INFER_RANKS=$((NUM_GPUS - 1))
fi
if [[ "$INFER_RANKS" -le 0 ]]; then
  INFER_RANKS=1
fi
if [[ -z "${CHECKPOINT_EVERY:-}" ]]; then
  CHECKPOINT_EVERY=$((200 * INFER_RANKS))
fi

# Outputs (kept inside repo; ignored by git)
mkdir -p "$REPO_ROOT/ffn_delta_checkpoints" "$REPO_ROOT/ffn_delta_outputs"
OUT_DIR="${OUT_DIR:-}"
OUT_RSQRT_DIR="${OUT_RSQRT_DIR:-}"
OUT_RSQRT_INV_DIR="${OUT_RSQRT_INV_DIR:-}"
if [[ -n "$OUT_DIR" && -z "$OUT_RSQRT_DIR" ]]; then
  OUT_RSQRT_DIR="$OUT_DIR"
fi
if [[ -n "$OUT_DIR" ]]; then
  OUT="${OUT:-}"
else
  OUT="${OUT:-$REPO_ROOT/ffn_delta_outputs/r_${WHICH}_b${BLOCKS}_n${MAX_PROMPTS}.pt}"
fi
if [[ -n "$OUT" ]]; then
  OUT_RSQRT="${OUT_RSQRT:-${OUT%.pt}_rsqrt.pt}"
  OUT_RSQRT_INV="${OUT_RSQRT_INV:-${OUT%.pt}_rsqrt_inv.pt}"
else
  OUT_RSQRT="${OUT_RSQRT:-}"
  OUT_RSQRT_INV="${OUT_RSQRT_INV:-}"
fi
CKPT="${CKPT:-$REPO_ROOT/ffn_delta_checkpoints/r_${WHICH}_b${BLOCKS}_n${MAX_PROMPTS}.ckpt.pt}"

echo "[run_capture_r_ddp] repo=$REPO_ROOT"
echo "[run_capture_r_ddp] log_file=$LOG_FILE"
echo "[run_capture_r_ddp] pipeline_config=$PIPELINE_CONFIG"
echo "[run_capture_r_ddp] blocks=$BLOCKS which=$WHICH max_prompts=$MAX_PROMPTS"
echo "[run_capture_r_ddp] num_frames=$NUM_FRAMES gpu_accum=$GPU_ACCUM"
echo "[run_capture_r_ddp] cuda_visible_devices=$CUDA_VISIBLE_DEVICES num_gpus=$NUM_GPUS"
echo "[run_capture_r_ddp] accum_device_rank0_only=$ACCUM_DEVICE_RANK0_ONLY"
echo "[run_capture_r_ddp] rank0_accum_only=$RANK0_ACCUM_ONLY"
echo "[run_capture_r_ddp] skip_rsqrt=$SKIP_RSQRT"
echo "[run_capture_r_ddp] rsqrt_device=$RSQRT_DEVICE"
echo "[run_capture_r_ddp] out=${OUT:-<disabled>}"
echo "[run_capture_r_ddp] out_rsqrt=${OUT_RSQRT:-<disabled>}"
echo "[run_capture_r_ddp] out_rsqrt_inv=${OUT_RSQRT_INV:-<disabled>}"
echo "[run_capture_r_ddp] out_dir=${OUT_DIR:-<none>}"
echo "[run_capture_r_ddp] out_rsqrt_dir=${OUT_RSQRT_DIR:-<none>}"
echo "[run_capture_r_ddp] out_rsqrt_inv_dir=${OUT_RSQRT_INV_DIR:-<none>}"
echo "[run_capture_r_ddp] ckpt=$CKPT checkpoint_every=$CHECKPOINT_EVERY resume=$RESUME"
echo "[run_capture_r_ddp] reduce_every=$REDUCE_EVERY"
echo "[run_capture_r_ddp] save_r=$SAVE_R"
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
  --checkpoint_path "$CKPT"
  --checkpoint_every "$CHECKPOINT_EVERY"
  --reduce_every "$REDUCE_EVERY"
  --num_frames "$NUM_FRAMES"
  --seed "$SEED"
)

if [[ "$SAVE_R" == "1" ]]; then
  ARGS+=(--save_r)
fi

if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
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
if [[ "$ACCUM_DEVICE_RANK0_ONLY" == "1" ]]; then
  ARGS+=(--accum_device_rank0_only)
fi
if [[ "$RANK0_ACCUM_ONLY" == "1" ]]; then
  ARGS+=(--rank0_accum_only)
fi
if [[ "$SKIP_RSQRT" == "1" ]]; then
  ARGS+=(--skip_rsqrt)
fi
if [[ -n "$RSQRT_DEVICE" ]]; then
  ARGS+=(--rsqrt_device "$RSQRT_DEVICE")
fi

if [[ -n "$OUT" ]]; then
  ARGS+=(--out "$OUT")
fi

if [[ -n "$OUT_RSQRT" ]]; then
  ARGS+=(--out_rsqrt "$OUT_RSQRT")
fi

if [[ -n "$OUT_RSQRT_INV" ]]; then
  ARGS+=(--out_rsqrt_inv "$OUT_RSQRT_INV")
fi

if [[ -n "$OUT_DIR" ]]; then
  ARGS+=(--out_dir "$OUT_DIR")
fi

if [[ -n "$OUT_RSQRT_DIR" ]]; then
  ARGS+=(--out_rsqrt_dir "$OUT_RSQRT_DIR")
fi

if [[ -n "$OUT_RSQRT_INV_DIR" ]]; then
  ARGS+=(--out_rsqrt_inv_dir "$OUT_RSQRT_INV_DIR")
fi

torchrun --standalone --nproc_per_node="$NUM_GPUS" ffn/capture_r_ddp.py "${ARGS[@]}"
