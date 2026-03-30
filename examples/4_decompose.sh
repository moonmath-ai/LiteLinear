#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
# Apply Low-Rank Decomposition + FP8 Quantization using captured R matrices

REPO_ROOT="$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"
export PYTHONPATH="$REPO_ROOT"

PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/ltxv-13b-0.9.8-distilled-fp8.yaml}"
R_PATH="${R_PATH:-$REPO_ROOT/ffn_delta_outputs/r_w1_b0-1_n25000.pt}"
LOWRANK_RANK="${LOWRANK_RANK:-128}"
BLOCKS="${BLOCKS:-0-5}" 

echo "[run_decompose] repo=$REPO_ROOT"
echo "[run_decompose] cmd_args=$@"
echo

cd "$REPO_ROOT"

# Check environment
python -c "import ltx_video" >/dev/null 2>&1 || {
  echo "ERROR: Python cannot import ltx_video. Activate your env first (e.g.: conda activate ltx_radial)." >&2
  exit 1
}

# Run
python ffn/decompose.py \
  --pipeline_config "$PIPELINE_CONFIG" \
  --r_path "$R_PATH" \
  --rank "$LOWRANK_RANK" \
  --blocks "$BLOCKS" \
  --benchmark \
  "$@"
