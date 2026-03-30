#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
# Rank sweep benchmark script for FFN decomposition
# Tests ranks: 32, 64, 128, 256
# Runs baseline once, then experimental passes in separate processes.

set -e

# Full logging
LOG_FILE="rank_sweep_full.log"
exec > >(tee "$LOG_FILE") 2>&1

# Defaults (override via environment for wrappers)
# BLOCKS="0-27"
BLOCKS="${BLOCKS:-0-47}"
NUM_FRAMES="${NUM_FRAMES:-16}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-3}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/ltxv-13b-0.9.8-distilled.yaml}"
FRAME_RATE="${FRAME_RATE:-24}"
SEED="${SEED:-171198}"
PROMPT="${PROMPT:-}"
REF_PATH="${REF_PATH:-}"


# Set local HF cache to avoid PermissionError on system cache
mkdir -p ./cache
export HF_HOME="$(pwd)/cache"
export TRANSFORMERS_CACHE="$(pwd)/cache"
export HF_HUB_CACHE="$(pwd)/cache"

# Use both w1 and w2 R-matrix files as requested by the user.
R_PATH="ffn_delta_outputs/r_w1_b0-1_n25000.ckpt.pt,ffn_delta_outputs/r_w2_b0-1_n25000.ckpt.pt"

# Check if test_prompts.json exists and read prompts from it
# Try multiple possible locations
PROMPTS_FILE="${PROMPTS_FILE:-}"
if [ -z "$PROMPTS_FILE" ]; then
    # Try current directory first, then script directory, then project root
    if [ -f "test_prompts.json" ]; then
        PROMPTS_FILE="test_prompts.json"
    elif [ -f "$(dirname "$0")/../test_prompts.json" ]; then
        PROMPTS_FILE="$(dirname "$0")/../test_prompts.json"
    elif [ -f "$(dirname "$0")/../../test_prompts.json" ]; then
        PROMPTS_FILE="$(dirname "$0")/../../test_prompts.json"
    fi
fi

if [ -n "$PROMPTS_FILE" ] && [ -f "$PROMPTS_FILE" ]; then
    echo "Reading prompts from $PROMPTS_FILE"
    # Read prompts from JSON file using Python - check if it's empty or has entries
    PROMPT_DATA=$(python3 -c "
import json, sys
try:
    data = json.load(open('$PROMPTS_FILE'))
    if not data or len(data) == 0:
        sys.exit(1)
    # Check format: if it's a list of dicts with seed/prompt, use that format
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'seed' in data[0] and 'prompt' in data[0]:
        # New format: list of dicts with seed and prompt
        for entry in data:
            seed_val = str(entry.get('seed', ''))
            prompt_val = str(entry.get('prompt', ''))
            print(f\"{seed_val}|{prompt_val}\")
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
        # Old format: list of strings (backward compatibility)
        for prompt in data:
            print(f\"|{prompt}\")
    else:
        sys.exit(1)
except Exception as e:
    sys.exit(1)
")
    if [ $? -eq 0 ] && [ -n "$PROMPT_DATA" ]; then
        PROMPTS="$PROMPT_DATA"
        PROMPT_COUNT=$(echo "$PROMPTS" | wc -l)
        echo "Found $PROMPT_COUNT entries in $PROMPTS_FILE"
    else
        # JSON file is empty or invalid, fall back to default behavior
        PROMPTS=""
        PROMPT_COUNT=0
        echo "JSON file is empty or invalid, falling back to default behavior"
    fi
else
    # No file found, fall back to default behavior
    PROMPTS=""
    PROMPT_COUNT=0
fi

# If no prompts from JSON, use environment variables (previous behavior)
if [ "$PROMPT_COUNT" -eq 0 ]; then
    if [ -n "$PROMPT" ]; then
        PROMPTS="|$PROMPT"
        PROMPT_COUNT=1
        echo "Using single prompt from PROMPT environment variable"
    else
        echo "No prompts file found and no PROMPT variable set. Using default seed."
    fi
fi

# Function to generate baseline for a single prompt
generate_baseline_for_prompt() {
    local ENTRY_SEED="$1"
    local CURRENT_PROMPT="$2"
    local PROMPT_INDEX="$3"
    local PROMPT_ID="$4"
    
    # Use entry seed if provided, otherwise use default SEED
    local CURRENT_SEED="${ENTRY_SEED:-$SEED}"
    
    EXTRA_INFER_ARGS=(
        --frame_rate "$FRAME_RATE"
        --seed "$CURRENT_SEED"
    )
    if [ -n "$CURRENT_PROMPT" ]; then
        EXTRA_INFER_ARGS+=(--prompt "$CURRENT_PROMPT")
    fi
    if [ -n "$REF_PATH" ]; then
        EXTRA_INFER_ARGS+=(--conditioning_image "$REF_PATH")
    fi
    
    mkdir -p ffn_delta_outputs
    BASELINE_VIDEO="ffn_delta_outputs/verify_baseline_${PROMPT_ID}_real.mp4"
    BASELINE_VIDEO_TMP="${BASELINE_VIDEO%.mp4}_tmp.mp4"
    
    echo ""
    echo ">>> Generating Baseline for Prompt $PROMPT_INDEX (Seed: $CURRENT_SEED)..."
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PATH=/root/miniconda3/envs/ltx-ffn/bin:$PATH \
        PYTHONPATH=/root/users/vh/LTX-Video:${PYTHONPATH:-} \
        python ffn/decompose.py \
        --pipeline_config "$PIPELINE_CONFIG" \
        --num_frames $NUM_FRAMES \
        --benchmark_runs $BENCHMARK_RUNS \
        --benchmark \
        --compile \
        --blocks "empty" \
        --output_video "$BASELINE_VIDEO" \
        --export_baseline \
        "${EXTRA_INFER_ARGS[@]}" \
        --skip_decomposition
    
    # decompose.py exports the baseline as <output>_baseline.mp4 when --export_baseline is set.
    # Rename it so we end up with a single stable baseline reference filename.
    if [ -f "$BASELINE_VIDEO_TMP" ]; then
        mv -f "$BASELINE_VIDEO_TMP" "$BASELINE_VIDEO"
    fi
}

# Function to run rank tests for a single prompt at a specific rank
run_rank_test_for_prompt() {
    local ENTRY_SEED="$1"
    local CURRENT_PROMPT="$2"
    local PROMPT_INDEX="$3"
    local PROMPT_ID="$4"
    local LOWRANK_RANK="$5"
    
    # Use entry seed if provided, otherwise use default SEED
    local CURRENT_SEED="${ENTRY_SEED:-$SEED}"
    
    EXTRA_INFER_ARGS=(
        --frame_rate "$FRAME_RATE"
        --seed "$CURRENT_SEED"
    )
    if [ -n "$CURRENT_PROMPT" ]; then
        EXTRA_INFER_ARGS+=(--prompt "$CURRENT_PROMPT")
    fi
    if [ -n "$REF_PATH" ]; then
        EXTRA_INFER_ARGS+=(--conditioning_image "$REF_PATH")
    fi
    
    # 1. Custom CUDA Kernels
    echo ""
    echo ">>> [1/2] Prompt $PROMPT_INDEX - Custom CUDA Kernels (Fused) - Rank $LOWRANK_RANK..."
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PATH=/root/miniconda3/envs/ltx-ffn/bin:$PATH \
        ./ffn/4_decompose.sh \
        --pipeline_config "$PIPELINE_CONFIG" \
        --r_path "$R_PATH" \
        --blocks $BLOCKS \
        --compile \
        --output_video ffn_delta_outputs/verify_r${LOWRANK_RANK}_kernel_${PROMPT_ID}_real.mp4 \
        --benchmark_runs $BENCHMARK_RUNS \
        --rank $LOWRANK_RANK \
        --num_frames $NUM_FRAMES \
        "${EXTRA_INFER_ARGS[@]}" \
        --skip_baseline
    
    # 2. Torch Fallback
    echo ""
    echo ">>> [2/2] Prompt $PROMPT_INDEX - Torch Fallback (No Custom Kernel) - Rank $LOWRANK_RANK..."
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PATH=/root/miniconda3/envs/ltx-ffn/bin:$PATH \
        ./ffn/4_decompose.sh \
        --pipeline_config "$PIPELINE_CONFIG" \
        --r_path "$R_PATH" \
        --blocks $BLOCKS \
        --compile \
        --output_video ffn_delta_outputs/verify_r${LOWRANK_RANK}_nokernel_${PROMPT_ID}_real.mp4 \
        --benchmark_runs $BENCHMARK_RUNS \
        --rank $LOWRANK_RANK \
        --num_frames $NUM_FRAMES \
        "${EXTRA_INFER_ARGS[@]}" \
        --no_kernel \
        --skip_baseline
}

echo "========================================="
echo "FFN Decomposition Comparative Rank Sweep"
echo "Blocks: $BLOCKS"
echo "Frames: $NUM_FRAMES"
echo "Ranks: $RANKS"
echo "Log: $LOG_FILE"
echo "========================================="

# Step 1: Generate baselines for all prompts first
echo ""
echo "=================================================================="
echo ">>> STEP 1: Generating Baselines for All Prompts"
echo "=================================================================="

# Parse and store all prompts with their seeds and IDs
declare -a PROMPT_SEEDS
declare -a PROMPT_TEXTS
declare -a PROMPT_IDS

ACTUAL_PROMPT_COUNT=0

if [ "$PROMPT_COUNT" -gt 0 ]; then
    PROMPT_INDEX=1
    while IFS= read -r entry; do
        if [ -n "$entry" ]; then
            # Parse entry: format is "seed|prompt" or "|prompt" (if no seed)
            ENTRY_SEED=""
            CURRENT_PROMPT=""
            if [[ "$entry" == *"|"* ]]; then
                ENTRY_SEED="${entry%%|*}"
                CURRENT_PROMPT="${entry#*|}"
                # If seed is empty, use default
                if [ -z "$ENTRY_SEED" ]; then
                    ENTRY_SEED=""
                fi
            else
                # Old format: just prompt
                CURRENT_PROMPT="$entry"
            fi
            
            # Create prompt-safe identifier for filenames
            PROMPT_ID=$(echo "$CURRENT_PROMPT" | head -c 50 | tr -cd '[:alnum:]_' | head -c 30)
            if [ -z "$PROMPT_ID" ]; then
                PROMPT_ID="prompt${PROMPT_INDEX}"
            fi
            
            PROMPT_SEEDS+=("$ENTRY_SEED")
            PROMPT_TEXTS+=("$CURRENT_PROMPT")
            PROMPT_IDS+=("$PROMPT_ID")
            
            generate_baseline_for_prompt "$ENTRY_SEED" "$CURRENT_PROMPT" "$PROMPT_INDEX" "$PROMPT_ID"
            PROMPT_INDEX=$((PROMPT_INDEX + 1))
            ACTUAL_PROMPT_COUNT=$((ACTUAL_PROMPT_COUNT + 1))
        fi
    done <<< "$PROMPTS"
else
    # Original behavior: run once without prompts, use default seed
    PROMPT_ID="prompt1"
    PROMPT_SEEDS+=("")
    PROMPT_TEXTS+=("")
    PROMPT_IDS+=("$PROMPT_ID")
    generate_baseline_for_prompt "" "" "1" "$PROMPT_ID"
    ACTUAL_PROMPT_COUNT=1
fi

# Step 2: Run rank sweeps - rank loop first, then prompt loop
echo ""
echo "=================================================================="
echo ">>> STEP 2: Running Rank Sweeps (Rank Loop First)"
echo "=================================================================="

for LOWRANK_RANK in $RANKS; do
    echo ""
    echo "=================================================================="
    echo ">>> RANK $LOWRANK_RANK - Testing All Prompts"
    echo "=================================================================="
    
    # For each prompt, test this rank
    for idx in "${!PROMPT_IDS[@]}"; do
        PROMPT_INDEX=$((idx + 1))
        ENTRY_SEED="${PROMPT_SEEDS[$idx]}"
        CURRENT_PROMPT="${PROMPT_TEXTS[$idx]}"
        PROMPT_ID="${PROMPT_IDS[$idx]}"
        
        echo ""
        echo ">>> Prompt $PROMPT_INDEX of $ACTUAL_PROMPT_COUNT (Seed: ${ENTRY_SEED:-$SEED})"
        run_rank_test_for_prompt "$ENTRY_SEED" "$CURRENT_PROMPT" "$PROMPT_INDEX" "$PROMPT_ID" "$LOWRANK_RANK"
    done
    
    echo ""
    echo ">>> Rank $LOWRANK_RANK Complete for All Prompts"
done

echo ""
echo "========================================="
echo "Rank sweep complete!"
echo "Full log saved to: $LOG_FILE"
echo "========================================="
