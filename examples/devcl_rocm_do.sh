#!/bin/bash
set -euo pipefail

# AMD dev cycle benchmark (Lin vs ROCM) on captured FFN shapes.
# Usage: ./devcl_rocm_do.sh [extra args passed to bench_lrdelta_amd.py]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export LITELINEAR_ENABLE_ROCM_EXT=1
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" bench_lrdelta_amd.py \
  --shapes-json captured_shapes_2.json \
  --iters 100 \
  --warmup 20 \
  "$@"
