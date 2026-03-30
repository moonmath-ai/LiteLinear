#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
set -e
set -o pipefail

# Be robust to being invoked from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Redirect all output to both terminal and log file
exec > >(tee -a devcl_nsys.log) 2>&1

export PATH=/root/miniconda3/envs/ltx-ffn/bin:$PATH
export PYTHONPATH=/root/users/vh/LTX-Video:${PYTHONPATH:-}

OUT_DIR="${OUT_DIR:-./nsys_reports}"
mkdir -p "$OUT_DIR"

SHAPES_JSON_DEFAULT="$SCRIPT_DIR/captured_shapes_2.json"
COMMON_ARGS=(
    --iters "${ITERS:-10}"
    --warmup "${WARMUP:-0}"
    --shapes-json "${SHAPES_JSON:-$SHAPES_JSON_DEFAULT}"
)
if [ "${NO_COMPILE:-0}" != "1" ]; then
    COMMON_ARGS+=(
        --compile
        --compile-mode "${COMPILE_MODE:-max-autotune}"
        --compile-backend "${COMPILE_BACKEND:-inductor}"
    )
fi

Q_DEQUANT_TILE_K=4096 \
USE_FP8_REMAINDER=1 \
CUBLASLT_AUTOTUNE_RESET=${CUBLASLT_AUTOTUNE_RESET:-0} \
CUBLASLT_AUTOTUNE_VERBOSE=1 \
nsys profile --trace=cuda,nvtx --cuda-event-trace=false --force-overwrite true -o "$OUT_DIR/fused_forward" \
    python profile_fused_forward.py "${COMMON_ARGS[@]}" "$@"
