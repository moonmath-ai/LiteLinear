#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
# Dev cycle: Run devcl_lrdelta.sh with specific environment variables
# Usage: ./devcl_do.sh

set -e
set -o pipefail  # Catch errors through pipes

# Be robust to being invoked from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Redirect all output to both terminal and log file
exec > >(tee -a devcl_do.log) 2>&1

# Benchmark mode controls:
#   BENCH_CUDA_ONLY=1 -> pass --bench-cuda-only
#   BENCH_LR_ONLY=1   -> pass --bench-lr-only (unless CUDA-only is enabled)
# Defaults keep prior behavior (CUDA-only) unless explicitly overridden.
BENCH_CUDA_ONLY="${BENCH_CUDA_ONLY:-0}"
BENCH_LR_ONLY="${BENCH_LR_ONLY:-0}"
BENCH_ARGS=()
if [ "$BENCH_CUDA_ONLY" -eq 1 ]; then
  BENCH_ARGS+=(--bench-cuda-only)
elif [ "$BENCH_LR_ONLY" -eq 1 ]; then
  BENCH_ARGS+=(--bench-lr-only)
fi

Q_DEQUANT_TILE_K=4096 \
USE_FP8_REMAINDER=1 \
CUBLASLT_AUTOTUNE_RESET=${CUBLASLT_AUTOTUNE_RESET:-0} \
CUBLASLT_AUTOTUNE_VERBOSE=${CUBLASLT_AUTOTUNE_VERBOSE:-1} \
./devcl_lrdelta.sh \
"${BENCH_ARGS[@]}" \
--compile \
--compile-mode max-autotune \
--compile-backend cudagraphs \
--shapes-json captured_shapes_2.json \
--iters 5 \
"$@"