#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
# Fast script to find optimal static scale
# Runs 1 benchmark iteration, 24 frames, no baseline export

set -e

BLOCKS="0-27"
NUM_FRAMES=48
BENCHMARK_RUNS=1
RANK=32
export RANK

if [ -z "$1" ]; then
    echo "Usage: $0 <static_scale>"
    exit 1
fi

SCALE=$1
OUT_VIDEO="ffn_delta_outputs/test_r${RANK}_scale_${SCALE}.mp4"
if [ -f "$OUT_VIDEO" ]; then
    rm "$OUT_VIDEO"
fi

echo "========================================="
echo "Testing Static Scale: $SCALE"
echo "Rank: $RANK, Frames: $NUM_FRAMES"
echo "========================================="

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PATH=/root/miniconda3/envs/ltx-ffn/bin:$PATH \
    ./ffn/4_decompose.sh \
    --blocks $BLOCKS \
    --compile \
    --output_video $OUT_VIDEO \
    --benchmark_runs $BENCHMARK_RUNS \
    --rank $RANK \
    --num_frames $NUM_FRAMES \
    --static_scale $SCALE \
    --skip_baseline

# Check video validity
if [ -f "$OUT_VIDEO" ]; then
    SIZE=$(ls -lh $OUT_VIDEO | awk '{print $5}')
    echo ""
    echo ">>> Output Video Size: $SIZE"
    if [[ "$SIZE" == *"M"* ]]; then # Simple check for M suffix
        echo ">>> STATUS: VALID (likely)"
    else
        echo ">>> STATUS: INVALID (likely black/NaN)"
    fi
else
    echo ">>> STATUS: FAILED (no video generated)"
fi
