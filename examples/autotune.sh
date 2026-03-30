#!/bin/bash
set -e
set -o pipefail

# Runtime autotune for v13 kernel (older commit).
export CUDA_VISIBLE_DEVICES=7
export CUBLASLT_AUTOTUNE_BF16=1
export CUBLASLT_AUTOTUNE_FP8=1
# Force re-profile even when persistent cache entries exist.
export CUBLASLT_AUTOTUNE_RESET="${CUBLASLT_AUTOTUNE_RESET:-0}"
export CUBLASLT_AUTOTUNE_WARMUP=2
export CUBLASLT_AUTOTUNE_ITERS=200
export CUBLASLT_AUTOTUNE_VERBOSE=1

if [ $# -gt 0 ]; then
  exec "$@"
else
  exec /root/miniconda3/envs/ltx-ffn/bin/python bench_ffn.py
fi
