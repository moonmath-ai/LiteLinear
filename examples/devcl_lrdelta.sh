#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
# Dev cycle (LowRankDeltaLinear):
#   - Rebuild kernel (if changed)
#   - Verify correctness: Linear vs TE Linear vs LowRankDeltaLinear (PT) vs LowRankDeltaLinear (CUDA)
#   - Benchmark: same 4-way comparison on real FFN shapes (w1/w2)
#
# Usage: ./devcl_lrdelta.sh [--r=64|--ranks=64,32] [--skip-verify] [--skip-bench] [--force-rebuild] [--nocompile] [--compile-mode=max-autotune] [--compile-backend=cudagraphs] [--factor-source=random|decompose|load] [--factors=path] [--factor-block=0] [--factor-layer=name] [--compare-factors] [--bench-lr-only] [--bench-cuda-only]
# Env: VERIFY_COMPILE_BACKEND overrides verify-only backend when compile-backend=cudagraphs (default: inductor)

set -e
set -o pipefail

# Be robust to being invoked from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Redirect all output to both terminal and log file
exec > >(tee -a devcl_lrdelta.log) 2>&1

export CUBLASLT_AUTOTUNE_RESET="${CUBLASLT_AUTOTUNE_RESET:-0}"

SKIP_VERIFY=0
SKIP_BENCH=0
FORCE_REBUILD=0
USE_COMPILE=1
COMPILE_MODE="max-autotune"
COMPILE_BACKEND="inductor"
SHAPES_JSON="${SHAPES_JSON:-captured_shapes_2.json}"
VERIFY_ITERS=5
BENCH_ITERS=""
RANKS_STR=""
FACTOR_SOURCE="random"
FACTORS_PATH=""
FACTOR_BLOCK=""
FACTOR_LAYER=""
COMPARE_FACTORS=0
BENCH_LR_ONLY="${BENCH_LR_ONLY:-0}"
BENCH_CUDA_ONLY="${BENCH_CUDA_ONLY:-0}"

# Normalize env-provided boolean-ish values to 0/1.
case "$BENCH_LR_ONLY" in
    1|true|TRUE|yes|YES) BENCH_LR_ONLY=1 ;;
    *) BENCH_LR_ONLY=0 ;;
esac
case "$BENCH_CUDA_ONLY" in
    1|true|TRUE|yes|YES) BENCH_CUDA_ONLY=1 ;;
    *) BENCH_CUDA_ONLY=0 ;;
esac
if [ "$BENCH_CUDA_ONLY" -eq 1 ]; then
    BENCH_LR_ONLY=1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-verify) SKIP_VERIFY=1; shift ;;
        --skip-bench) SKIP_BENCH=1; shift ;;
        --force-rebuild) FORCE_REBUILD=1; shift ;;
        --no-compile) USE_COMPILE=0; shift ;;
        --compile-mode) COMPILE_MODE="$2"; shift 2 ;;
        --compile-mode=*) COMPILE_MODE="${1#*=}"; shift ;;
        --compile-backend) COMPILE_BACKEND="$2"; shift 2 ;;
        --compile-backend=*) COMPILE_BACKEND="${1#*=}"; shift ;;
        --iters) VERIFY_ITERS="$2"; BENCH_ITERS="$2"; shift 2 ;;
        --iters=*) VERIFY_ITERS="${1#*=}"; BENCH_ITERS="${1#*=}"; shift ;;
        --r|--ranks) RANKS_STR="$2"; shift 2 ;;
        --r=*|--ranks=*) RANKS_STR="${1#*=}"; shift ;;
        --factor-source) FACTOR_SOURCE="$2"; shift 2 ;;
        --factor-source=*) FACTOR_SOURCE="${1#*=}"; shift ;;
        --factors) FACTORS_PATH="$2"; shift 2 ;;
        --factors=*) FACTORS_PATH="${1#*=}"; shift ;;
        --factor-block) FACTOR_BLOCK="$2"; shift 2 ;;
        --factor-block=*) FACTOR_BLOCK="${1#*=}"; shift ;;
        --factor-layer) FACTOR_LAYER="$2"; shift 2 ;;
        --factor-layer=*) FACTOR_LAYER="${1#*=}"; shift ;;
        --compare-factors) COMPARE_FACTORS=1; shift ;;
        --bench-lr-only) BENCH_LR_ONLY=1; shift ;;
        --bench-cuda-only) BENCH_CUDA_ONLY=1; BENCH_LR_ONLY=1; shift ;;
        *) shift ;;
    esac
done

if [ -z "$RANKS_STR" ]; then
    RANKS_STR="${RANKS:-64}"
fi
RANKS_STR="${RANKS_STR//,/ }"
read -r -a RANK_LIST <<< "$RANKS_STR"

export PATH=/root/miniconda3/envs/ltx-ffn/bin:$PATH
export PYTHONPATH=/root/users/vh/LTX-Video:${PYTHONPATH:-}

echo "========================================="
echo "LowRankDeltaLinear Dev Cycle"
echo "========================================="

if [ "$COMPARE_FACTORS" -eq 1 ] && [ -z "$FACTORS_PATH" ]; then
    echo "❌ --compare-factors requires --factors=PATH"
    exit 1
fi

FACTOR_SOURCES=("$FACTOR_SOURCE")
if [ "$COMPARE_FACTORS" -eq 1 ]; then
    FACTOR_SOURCES=("decompose" "load")
fi

# 1. Rebuild (only if kernel source changed)
KERNEL_SRCS=(
    "../lite_linear/csrc/kernel_v13.cu"
    "../lite_linear/csrc/binding.cpp"
    "../lite_linear/csrc/kernel_utils.cuh"
    "../lite_linear/csrc/ltx_cuda.h"
)
KERNEL_HASH_FILE=".kernel_hash"
missing=0
for f in "${KERNEL_SRCS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "❌ Missing kernel source: $f"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    exit 1
fi

if ! CURRENT_HASH=$(md5sum "${KERNEL_SRCS[@]}" 2>/dev/null | md5sum | cut -d' ' -f1); then
    echo "❌ Failed to hash kernel sources"
    exit 1
fi
LAST_HASH=$(cat "$KERNEL_HASH_FILE" 2>/dev/null || echo "")

rebuild_kernel() {
    if ! ./rbld.sh; then
        echo ""
        echo "❌ BUILD FAILED - check rbld.log for details"
        exit 1
    fi
    echo "$CURRENT_HASH" > "$KERNEL_HASH_FILE"
}

if [ "$FORCE_REBUILD" -eq 1 ]; then
    echo ""
    echo "[1/3] Rebuilding kernel (--force-rebuild)..."
    rebuild_kernel
elif [ "$CURRENT_HASH" != "$LAST_HASH" ]; then
    echo ""
    echo "[1/3] Rebuilding kernel (source changed)..."
    rebuild_kernel
else
    echo ""
    echo "[1/3] Skipping rebuild (no changes to extension sources)"
fi

# 2. Verify
if [ $SKIP_VERIFY -eq 0 ]; then
    echo ""
    echo "[2/3] Verifying LowRankDeltaLinear correctness (Linear vs TE vs LR-PT vs LR-CUDA)..."
    if [ "$USE_COMPILE" -eq 1 ] && [ "$COMPILE_BACKEND" = "cudagraphs" ]; then
        echo "[verify] compile backend override: using ${VERIFY_COMPILE_BACKEND:-inductor} (cudagraphs is unstable across varying shapes)"
    fi
    echo "Ranks: ${RANK_LIST[*]}"
    echo "Factor sources: ${FACTOR_SOURCES[*]}"
    # Use a few randomized iterations by default to catch intermittent issues.
    for RANK in "${RANK_LIST[@]}"; do
        for SRC in "${FACTOR_SOURCES[@]}"; do
            echo ""
            echo ">>> Verify rank ${RANK} | source=${SRC}"
            VERIFY_ARGS=(--iters "$VERIFY_ITERS" --r "$RANK" --factor-source "$SRC")
            VERIFY_ARGS+=(--shapes-json "$SHAPES_JSON")
            if [ -n "$FACTOR_BLOCK" ]; then
                VERIFY_ARGS+=(--factor-block "$FACTOR_BLOCK")
            fi
            if [ -n "$FACTOR_LAYER" ]; then
                VERIFY_ARGS+=(--factor-layer "$FACTOR_LAYER")
            fi
            if [ "$SRC" = "load" ]; then
                VERIFY_ARGS+=(--factors "$FACTORS_PATH")
            fi
            if [ "$USE_COMPILE" -eq 1 ]; then
                VERIFY_BACKEND="$COMPILE_BACKEND"
                if [ "$VERIFY_BACKEND" = "cudagraphs" ]; then
                    # Verification runs many distinct shapes; cudagraphs can fail due shape replays.
                    VERIFY_BACKEND="${VERIFY_COMPILE_BACKEND:-inductor}"
                fi
                VERIFY_ARGS+=(--compile --compile-mode "$COMPILE_MODE" --compile-backend "$VERIFY_BACKEND")
            fi
            python verify_lrdelta.py "${VERIFY_ARGS[@]}"
            if [ $? -ne 0 ]; then
                echo ""
                echo "❌ VERIFICATION FAILED"
                exit 1
            fi
        done
    done
else
    echo ""
    echo "[2/3] Skipping verification (--skip-verify)"
fi

# 3. Benchmark
if [ $SKIP_BENCH -eq 0 ]; then
    echo ""
    if [ "$BENCH_CUDA_ONLY" -eq 1 ]; then
        echo "[3/3] Benchmarking (LR-CUDA only)..."
    elif [ "$BENCH_LR_ONLY" -eq 1 ]; then
        echo "[3/3] Benchmarking (LR-PT vs LR-CUDA only)..."
    else
        echo "[3/3] Benchmarking (Linear vs TE vs LR-PT vs LR-CUDA)..."
    fi
    echo "Ranks: ${RANK_LIST[*]}"
    echo "Factor sources: ${FACTOR_SOURCES[*]}"
    for RANK in "${RANK_LIST[@]}"; do
        for SRC in "${FACTOR_SOURCES[@]}"; do
            echo ""
            echo ">>> Bench rank ${RANK} | source=${SRC}"
            BENCH_ARGS=(--r "$RANK" --factor-source "$SRC")
            BENCH_ARGS+=(--shapes-json "$SHAPES_JSON")
            if [ -n "$BENCH_ITERS" ]; then
                BENCH_ARGS+=(--iters "$BENCH_ITERS")
            fi
            if [ -n "$FACTOR_BLOCK" ]; then
                BENCH_ARGS+=(--factor-block "$FACTOR_BLOCK")
            fi
            if [ -n "$FACTOR_LAYER" ]; then
                BENCH_ARGS+=(--factor-layer "$FACTOR_LAYER")
            fi
            if [ "$SRC" = "load" ]; then
                BENCH_ARGS+=(--factors "$FACTORS_PATH")
            fi
            if [ "$USE_COMPILE" -eq 1 ]; then
                BENCH_ARGS+=(--compile --compile-mode "$COMPILE_MODE" --compile-backend "$COMPILE_BACKEND")
            fi
            if [ "$BENCH_LR_ONLY" -eq 1 ]; then
                BENCH_ARGS+=(--bench-lr-only)
            fi
            if [ "$BENCH_CUDA_ONLY" -eq 1 ]; then
                BENCH_ARGS+=(--bench-cuda-only)
            fi
            python bench_lrdelta.py "${BENCH_ARGS[@]}"
        done
    done
else
    echo ""
    echo "[3/3] Skipping benchmark (--skip-bench)"
fi

echo ""
echo "========================================="
echo "Dev cycle complete!"
echo "========================================="

