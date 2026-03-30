#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
set -e
set -o pipefail

# Be robust to being invoked from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Redirect all output to both terminal and log file
exec > >(tee -a devcl_ncu.log) 2>&1

export PATH=/root/miniconda3/envs/ltx-ffn/bin:$PATH
export PYTHONPATH=/root/users/vh/LTX-Video:${PYTHONPATH:-}

OUT_DIR="${OUT_DIR:-./ncu_reports}"
mkdir -p "$OUT_DIR"

SHAPES_JSON_DEFAULT="$SCRIPT_DIR/captured_shapes_2.json"
COMMON_ARGS=(
    --iters "${ITERS:-5}"
    --warmup "${WARMUP:-2}"
    --shapes-json "${SHAPES_JSON:-$SHAPES_JSON_DEFAULT}"
)
if [ "${NO_COMPILE:-0}" != "1" ]; then
    COMMON_ARGS+=(
        --compile
        --compile-mode "${COMPILE_MODE:-max-autotune}"
        --compile-backend "${COMPILE_BACKEND:-inductor}"
    )
fi

NCU_ARGS=(
    --set "${NCU_SET:-basic}"
    --target-processes all
    --nvtx
    --nvtx-push-pop-scope process
    --nvtx-include "${NCU_NVTX_LITE:-lite_linear_iter_0}"
    --nvtx-include "${NCU_NVTX_TE:-TE_iter_0}"
)
if [ -n "${NCU_LAUNCH_COUNT:-}" ] || [ -n "${NCU_LAUNCH_SKIP:-}" ]; then
    NCU_ARGS+=(
        --launch-skip "${NCU_LAUNCH_SKIP:-0}"
        --launch-count "${NCU_LAUNCH_COUNT:-1}"
    )
fi

Q_DEQUANT_TILE_K=4096 \
USE_FP8_REMAINDER=1 \
CUBLASLT_AUTOTUNE_RESET=${CUBLASLT_AUTOTUNE_RESET:-0} \
CUBLASLT_AUTOTUNE_VERBOSE=1 \
ncu "${NCU_ARGS[@]}" \
    -f -o "$OUT_DIR/fused_forward" \
    python profile_fused_forward.py "${COMMON_ARGS[@]}" "$@"
