#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
set -e

# Configuration
NUM_FRAMES=8
COMPILE_MODE="reduce-overhead"
OUT_DIR="./ffn_delta_outputs"
REPO_DIR="/root/users/vh/LTX-Video"

# Ensure environment is correct
export PATH="/root/miniconda3/envs/ltx-ffn/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "Starting Full-Capture Profiling Workflow (8 Frames)"
echo "=========================================================="

# 1. Profile Baseline
echo "[1/2] Profiling Baseline Model..."
python "$REPO_DIR/ffn/decompose.py" \
    --num_frames $NUM_FRAMES \
    --frame_rate 25 \
    --compile \
    --compile_mode $COMPILE_MODE \
    --profile \
    --profile_baseline \
    --skip_baseline \
    --output_video "$OUT_DIR/baseline_profile.mp4"

# 2. Profile Rank 128
echo "[2/2] Profiling Decomposed Model (Rank 128)..."
python "$REPO_DIR/ffn/decompose.py" \
    --num_frames $NUM_FRAMES \
    --frame_rate 25 \
    --compile \
    --compile_mode $COMPILE_MODE \
    --profile \
    --rank 128 \
    --skip_baseline \
    --output_video "$OUT_DIR/r128_profile.mp4"

echo "=========================================================="
echo "Profiling Complete!"
echo "Traces saved to $OUT_DIR"
echo "=========================================================="
