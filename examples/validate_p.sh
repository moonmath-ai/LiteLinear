#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
# Minimal validation: baseline + CUDA (LR-PT optional) per prompt.
# Usage: ./validate_p.sh [RANK=32] [FRAMES=24] [--force-baseline] [--force-factors] [--no-factors] [--fp8] [--only-issues]

export Q_DEQUANT_TILE_K=4096
export USE_FP8_REMAINDER=1
export CUBLASLT_AUTOTUNE_RESET="${CUBLASLT_AUTOTUNE_RESET:-0}"

export FRAMES=${FRAMES:-120}
export NUM_FRAMES=${NUM_FRAMES:-$FRAMES}
export PROMPTS_FILE="/tmp/no_prompts.json"
# export PROMPT="A gleaming silver Porsche 911 aggressively pursues a black Dodge Challenger and red Ford Mustang across a vast, blindingly white salt flat. Studio lighting creates sharp contrasts with long shadows stretching from each vehicle as they drift through dust clouds. Camera tracks alongside the Porsche, low angle emphasizing speed, while the truck looms distantly ahead; tires kick up plumes of salt crystals. Vehicles exhibit subtle body roll and suspension compression with each rapid movement"
export PROMPT="A charming animated scene of a fluffy, white kitten with bright green eyes and soft, curly fur catching a delicate butterfly in a field of tall grass. The kitten has playful, curious expressions as it reaches out with its front paw to gently grab the fluttering butterfly. The butterfly has vibrant orange and black wings and flutters around before being caught. The grass sways gently in the breeze, creating a serene and lively atmosphere. Close-up view, focusing on the interaction between the kitten and the butterfly."
export BASELINE=${BASELINE:-1} 
export LR_PT=${LR_PT:-0}     
export RANKS=64
export COMPILE=${COMPILE:-1}
export PRECOMPUTE_FACTORS=${PRECOMPUTE_FACTORS:-1}

set -e
set -o pipefail

if [ -f validate_p.log ]; then
    python3 - <<'PY'
import re
from pathlib import Path

log = Path("validate_p.log")
existing = list(Path(".").glob("validate_p.log.bak*.log"))
nums = []
for p in existing:
    m = re.search(r"validate_p\.log\.bak(\d+)\.log$", p.name)
    if m:
        nums.append(int(m.group(1)))
next_num = (max(nums) + 1) if nums else 1
log.rename(f"validate_p.log.bak{next_num}.log")
PY
fi

# Redirect all output to both terminal and log file
exec > >(tee -a validate_p.log) 2>&1

RUN_ID="$(date +%s)"
echo ""
echo "========================================="
echo "==== RUN_ID ${RUN_ID} START ===="
echo "Command: $0 $*"
echo "Timestamp: $(date -Is)"
echo "========================================="
echo ""

FORCE_BASELINE=${FORCE_BASELINE:-0}
FORCE_FACTORS=${FORCE_FACTORS:-0}
FP8_CONFIG=${FP8_CONFIG:-0}
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --force-baseline)
            FORCE_BASELINE=1
            ;;
        --force-factors)
            FORCE_FACTORS=1
            ;;
        --no-factors)
            PRECOMPUTE_FACTORS=0
            ;;
        --fp8)
            FP8_CONFIG=1
            ;;
        --only-issues)
            ONLY_ISSUES=1
            ;;
        *)
            POSITIONAL+=("$arg")
            ;;
    esac
done

RANK="${POSITIONAL[0]:-$RANKS}"
FRAMES="${POSITIONAL[1]:-$NUM_FRAMES}"
BLOCKS="${BLOCKS:-0-47}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/ltxv-13b-0.9.8-distilled.yaml}"
BASELINE_PIPELINE_CONFIG="$PIPELINE_CONFIG"
if [ "$FP8_CONFIG" -eq 1 ]; then
    BASELINE_PIPELINE_CONFIG="configs/ltxv-13b-0.9.8-distilled-fp8.yaml"
fi
FRAME_RATE="${FRAME_RATE:-24}"
SEED="${SEED:-171198}"
PROMPT="${PROMPT:-}"
REF_PATH="${REF_PATH:-}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
BASELINE="${BASELINE:-1}"
LR_PT="${LR_PT:-0}"
COMPILE="${COMPILE:-1}"
ONLY_ISSUES="${ONLY_ISSUES:-0}"
RANKS="${RANKS:-$RANK}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-10}"
BENCHMARK_ENABLED=0
if [ "$COMPILE" -eq 1 ]; then
    if [ "$BENCHMARK_RUNS" -gt 0 ]; then
        BENCHMARK_ENABLED=1
    else
        echo "INFO: BENCHMARK_RUNS=$BENCHMARK_RUNS, skipping benchmark."
    fi
fi

COMPILE_ARGS=()
if [ "$COMPILE" -eq 1 ]; then
    COMPILE_ARGS+=(--compile)
fi

# Set local HF cache
mkdir -p ./cache
export HF_HOME="$(pwd)/cache"
export HF_HUB_CACHE="$(pwd)/cache"
unset TRANSFORMERS_CACHE

R_PATH="ffn_delta_outputs/r_w1_b0-1_n25000.ckpt.pt,ffn_delta_outputs/r_w2_b0-1_n25000.ckpt.pt"

ensure_local_model_file() {
    local filename="$1"
    local dest="$PWD/$filename"
    if [ -f "$dest" ]; then
        return
    fi
    local cached_path
    cached_path=$(python3 - <<'PY' "$filename"
from huggingface_hub import hf_hub_download
import sys
fname = sys.argv[1]
path = hf_hub_download(repo_id="Lightricks/LTX-Video", filename=fname, repo_type="model")
print(path)
PY
)
    if [ -n "$cached_path" ] && [ -f "$cached_path" ]; then
        ln -s "$cached_path" "$dest" 2>/dev/null || {
            echo "WARNING: Failed to create symlink for $filename (no copy fallback)." >&2
        }
    fi
}

# Ensure local files exist to avoid repeated HF download logs
ensure_local_model_file "ltxv-13b-0.9.8-distilled.safetensors"
ensure_local_model_file "ltxv-13b-0.9.8-distilled-fp8.safetensors"
ensure_local_model_file "ltxv-spatial-upscaler-0.9.8.safetensors"

resolve_checkpoint_path() {
    python3 - <<'PY' "$1"
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
if not cfg_path.exists():
    print("")
    sys.exit(0)
data = yaml.safe_load(cfg_path.read_text())
ckpt = data.get("checkpoint_path")
print(ckpt or "")
PY
}

CKPT_PATH="$(resolve_checkpoint_path "$PIPELINE_CONFIG")"
CKPT_FILE=""
if [ -n "$CKPT_PATH" ]; then
    if [[ "$CKPT_PATH" = /* ]]; then
        CKPT_FILE="$CKPT_PATH"
    else
        ensure_local_model_file "$CKPT_PATH"
        CKPT_FILE="$PWD/$CKPT_PATH"
    fi
fi

MODEL_NAME="unknown_model"
if [ -n "$CKPT_FILE" ]; then
    CKPT_BASENAME="$(basename "$CKPT_FILE")"
    MODEL_NAME="${CKPT_BASENAME%.*}"
fi
PIPELINE_NAME="unknown_pipeline"
if [ -n "$PIPELINE_CONFIG" ]; then
    PIPELINE_BASENAME="$(basename "$PIPELINE_CONFIG")"
    PIPELINE_NAME="${PIPELINE_BASENAME%.*}"
fi
FACTORS_DIR_DEFAULT="ffn_delta_outputs/lr_data/${MODEL_NAME}"
if [ "$PIPELINE_NAME" != "$MODEL_NAME" ]; then
    FACTORS_DIR_DEFAULT="ffn_delta_outputs/lr_data/${MODEL_NAME}/${PIPELINE_NAME}"
fi
FACTORS_DIR="${FACTORS_DIR:-$FACTORS_DIR_DEFAULT}"
export FACTORS_DIR

if [ -z "$PROMPTS_FILE" ]; then
    if [ -f "test_prompts.json" ]; then
        PROMPTS_FILE="test_prompts.json"
    elif [ -f "$(dirname "$0")/../test_prompts.json" ]; then
        PROMPTS_FILE="$(dirname "$0")/../test_prompts.json"
    elif [ -f "$(dirname "$0")/../../test_prompts.json" ]; then
        PROMPTS_FILE="$(dirname "$0")/../../test_prompts.json"
    fi
fi

PROMPTS=""
if [ -n "$PROMPTS_FILE" ] && [ -f "$PROMPTS_FILE" ]; then
    echo "Reading prompts from $PROMPTS_FILE"
    PROMPTS=$(python3 -c "
import json, sys
try:
    data = json.load(open('$PROMPTS_FILE'))
    if not data:
        sys.exit(1)
    if isinstance(data, list) and isinstance(data[0], dict) and 'prompt' in data[0]:
        for entry in data:
            seed = entry.get('seed','')
            prompt = entry.get('prompt','')
            is_issue = 1 if entry.get('is_issue') else 0
            print(f\"{seed}|{prompt}|{is_issue}\")
    elif isinstance(data, list) and isinstance(data[0], str):
        for prompt in data:
            print(f\"|{prompt}|0\")
    else:
        sys.exit(1)
except Exception:
    sys.exit(1)
")
fi

if [ -z "$PROMPTS" ]; then
    if [ -n "$PROMPT" ]; then
        PROMPTS="|$PROMPT|0"
    else
        PROMPTS="||0"
    fi
fi

unique_output_path() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "$path"
        return 0
    fi

    local dir base stem ext
    dir="$(dirname "$path")"
    base="$(basename "$path")"
    ext=""
    stem="$base"
    if [[ "$base" == *.* ]]; then
        ext=".${base##*.}"
        stem="${base%$ext}"
    fi

    local idx=2
    local candidate
    while true; do
        candidate="${dir}/${stem}_${idx}${ext}"
        if [ ! -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
        idx=$((idx + 1))
    done
}

factors_has_r() {
    local path="$1"
    if [ -z "$path" ] || [ ! -f "$path" ]; then
        echo "0"
        return 0
    fi

    /root/miniconda3/envs/ltx-ffn/bin/python - <<'PY' "$path"
import sys
from safetensors import safe_open

path = sys.argv[1]
with safe_open(path, framework="pt", device="cpu") as f:
    meta = f.metadata() or {}
val = str(meta.get("with_r", "")).lower()
print("1" if val in ("1", "true", "yes", "y") else "0")
PY
}

run_baseline() {
    local ENTRY_SEED="$1"
    local CURRENT_PROMPT="$2"
    local PROMPT_ID="$3"
    local CURRENT_SEED="${ENTRY_SEED:-$SEED}"
    local CONFIG_FP8_TAG="bf16"
    if [ "${FP8_CONFIG:-0}" -eq 1 ]; then
        CONFIG_FP8_TAG="fp8"
    fi
    local COMPILE_TAG="eager"

    if [ "$COMPILE" -eq 1 ]; then
        COMPILE_TAG="default"
    fi
    local BENCHMARK_ARGS=()
    if [ "$BENCHMARK_ENABLED" -eq 1 ]; then
        BENCHMARK_ARGS+=(--benchmark --benchmark_runs "$BENCHMARK_RUNS")
    else
        BENCHMARK_ARGS+=(--skip_benchmark)
    fi
    local FORCE_ARGS=()
    if [ "$FORCE_BASELINE" -eq 1 ]; then
        FORCE_ARGS+=(--force_output)
    fi

    EXTRA_INFER_ARGS=(--frame_rate "$FRAME_RATE" --seed "$CURRENT_SEED")
    if [ -n "$CURRENT_PROMPT" ]; then
        EXTRA_INFER_ARGS+=(--prompt "$CURRENT_PROMPT")
    fi
    if [ -n "$REF_PATH" ]; then
        EXTRA_INFER_ARGS+=(--conditioning_image "$REF_PATH")
    fi

    mkdir -p ffn_delta_outputs
    local BASELINE_VIDEO_BASE="ffn_delta_outputs/validate_s${CURRENT_SEED}_f${FRAMES}_${COMPILE_TAG}_${CONFIG_FP8_TAG}_baseline_${PROMPT_ID}.mp4"
    if [ -f "$BASELINE_VIDEO_BASE" ] && [ "$FORCE_BASELINE" -eq 0 ]; then
        echo "[baseline] Exists, skipping (use --force-baseline to regenerate): $BASELINE_VIDEO_BASE"
        return
    fi
    BASELINE_VIDEO="$(unique_output_path "$BASELINE_VIDEO_BASE")"

    echo ""
    echo "========================================="
    echo "==== BASELINE-${COMPILE_TAG}-${CONFIG_FP8_TAG} seed=${CURRENT_SEED} ===="
    echo "========================================="
    echo ""
    env PYTORCH_ALLOC_CONF=expandable_segments:True \
        PATH=/root/miniconda3/envs/ltx-ffn/bin:$PATH \
        PYTHONPATH=/root/users/vh/LTX-Video:${PYTHONPATH:-} \
        python ffn/decompose.py \
        --pipeline_config "$BASELINE_PIPELINE_CONFIG" \
        --num_frames "$FRAMES" \
        "${COMPILE_ARGS[@]}" \
        "${BENCHMARK_ARGS[@]}" \
        "${FORCE_ARGS[@]}" \
        --blocks "empty" \
        --output_video "$BASELINE_VIDEO" \
        "${EXTRA_INFER_ARGS[@]}" \
        --skip_decomposition
}

ensure_factors() {
    local CURRENT_RANK="$1"
    if [ "$PRECOMPUTE_FACTORS" -eq 0 ]; then
        return 0
    fi

    mkdir -p "$FACTORS_DIR"
    local FACTORS_PATH="$FACTORS_DIR/ffn_lowrank_factors_r${CURRENT_RANK}.safetensors"
    if [ -f "$FACTORS_PATH" ] && [ "$FORCE_FACTORS" -eq 0 ]; then
        echo "[factors] Exists, reusing: $FACTORS_PATH"
        return 0
    fi

    if [ -z "$CKPT_FILE" ] || [ ! -f "$CKPT_FILE" ]; then
        echo "[factors] ERROR: checkpoint not found (PIPELINE_CONFIG=$PIPELINE_CONFIG, ckpt=$CKPT_FILE)"
        return 1
    fi

    echo "[factors] Building low-rank factors (rank=${CURRENT_RANK})..."
    local R_ARGS=()
    if [ -n "$R_PATH" ]; then
        R_ARGS+=(--r_matrices "$R_PATH")
    fi

    env PYTORCH_ALLOC_CONF=expandable_segments:True \
        PATH=/root/miniconda3/envs/ltx-ffn/bin:$PATH \
        PYTHONPATH=/root/users/vh/LTX-Video:${PYTHONPATH:-} \
        python ffn/decompose_weights.py \
        --ckpt "$CKPT_FILE" \
        --out "$FACTORS_DIR" \
        --rank "$CURRENT_RANK" \
        "${R_ARGS[@]}"
}

run_cuda() {
    local ENTRY_SEED="$1"
    local CURRENT_PROMPT="$2"
    local PROMPT_ID="$3"
    local CURRENT_RANK="$4"
    local CURRENT_SEED="${ENTRY_SEED:-$SEED}"
    local Q_FP8_TAG="deq"
    if [ "${USE_FP8_REMAINDER:-0}" -eq 1 ]; then
        Q_FP8_TAG="cxfp8"
    fi
    local COMPILE_TAG="eager"
    if [ "$COMPILE" -eq 1 ]; then
        COMPILE_TAG="default"
    fi
    local BENCHMARK_ARGS=()
    if [ "$BENCHMARK_ENABLED" -eq 1 ]; then
        BENCHMARK_ARGS+=(--benchmark_runs "$BENCHMARK_RUNS")
    else
        BENCHMARK_ARGS+=(--skip_benchmark)
    fi

    EXTRA_INFER_ARGS=(--frame_rate "$FRAME_RATE" --seed "$CURRENT_SEED")
    if [ -n "$CURRENT_PROMPT" ]; then
        EXTRA_INFER_ARGS+=(--prompt "$CURRENT_PROMPT")
    fi
    if [ -n "$REF_PATH" ]; then
        EXTRA_INFER_ARGS+=(--conditioning_image "$REF_PATH")
    fi

    FACTOR_ARGS=()
    local CALIB_TAG=""
    if [ "$PRECOMPUTE_FACTORS" -eq 1 ]; then
        ensure_factors "$CURRENT_RANK"
        FACTORS_PATH="$FACTORS_DIR/ffn_lowrank_factors_r${CURRENT_RANK}.safetensors"
        if [ -f "$FACTORS_PATH" ]; then
            FACTOR_ARGS+=(--factors "$FACTORS_PATH")
            if [ "$(factors_has_r "$FACTORS_PATH")" -eq 0 ]; then
                CALIB_TAG="noR_"
            fi
        fi
    else
        FACTORS_PATH="$FACTORS_DIR/ffn_lowrank_factors_r${CURRENT_RANK}.safetensors"
        if [ ! -f "$FACTORS_PATH" ]; then
            echo "[factors] ERROR: PRECOMPUTE_FACTORS=0 but missing $FACTORS_PATH" >&2
            exit 1
        fi
        FACTOR_ARGS+=(--factors "$FACTORS_PATH")
        if [ "$(factors_has_r "$FACTORS_PATH")" -eq 0 ]; then
            CALIB_TAG="noR_"
        fi
    fi

    echo ""
    echo "========================================="
    echo "==== LR-CUDA-${COMPILE_TAG}-${Q_FP8_TAG} rank=${CURRENT_RANK} seed=${CURRENT_SEED} ===="
    echo "========================================="
    echo ""
    local OUTPUT_VIDEO_BASE="ffn_delta_outputs/validate_r${CURRENT_RANK}_s${CURRENT_SEED}_f${FRAMES}_${COMPILE_TAG}_${Q_FP8_TAG}_lr-cu_${CALIB_TAG}${PROMPT_ID}.mp4"
    local OUTPUT_VIDEO
    OUTPUT_VIDEO="$(unique_output_path "$OUTPUT_VIDEO_BASE")"

    env PYTORCH_ALLOC_CONF=expandable_segments:True \
        PATH=/root/miniconda3/envs/ltx-ffn/bin:$PATH \
        USE_LITE_LINEAR=1 \
        ./ffn/4_decompose.sh \
        --pipeline_config "$PIPELINE_CONFIG" \
        --r_path "$R_PATH" \
        --blocks "$BLOCKS" \
        "${COMPILE_ARGS[@]}" \
        "${BENCHMARK_ARGS[@]}" \
        "${FACTOR_ARGS[@]}" \
        --output_video "$OUTPUT_VIDEO" \
        --rank "$CURRENT_RANK" \
        --num_frames "$FRAMES" \
        "${EXTRA_INFER_ARGS[@]}" \
        --use_lite_linear \
        --skip_baseline
}

run_lr_pt() {
    local ENTRY_SEED="$1"
    local CURRENT_PROMPT="$2"
    local PROMPT_ID="$3"
    local CURRENT_RANK="$4"
    local CURRENT_SEED="${ENTRY_SEED:-$SEED}"
    local Q_FP8_TAG="deq"
    local COMPILE_TAG="eager"
    if [ "${USE_FP8_REMAINDER:-0}" -eq 1 ]; then
        Q_FP8_TAG="fp8"
    fi
    if [ "$COMPILE" -eq 1 ]; then
        COMPILE_TAG="default"
    fi
    local BENCHMARK_ARGS=()
    if [ "$BENCHMARK_ENABLED" -eq 1 ]; then
        BENCHMARK_ARGS+=(--benchmark_runs "$BENCHMARK_RUNS")
    else
        BENCHMARK_ARGS+=(--skip_benchmark)
    fi

    EXTRA_INFER_ARGS=(--frame_rate "$FRAME_RATE" --seed "$CURRENT_SEED")
    if [ -n "$CURRENT_PROMPT" ]; then
        EXTRA_INFER_ARGS+=(--prompt "$CURRENT_PROMPT")
    fi
    if [ -n "$REF_PATH" ]; then
        EXTRA_INFER_ARGS+=(--conditioning_image "$REF_PATH")
    fi

    FACTOR_ARGS=()
    local CALIB_TAG=""
    if [ "$PRECOMPUTE_FACTORS" -eq 1 ]; then
        ensure_factors "$CURRENT_RANK"
        FACTORS_PATH="$FACTORS_DIR/ffn_lowrank_factors_r${CURRENT_RANK}.safetensors"
        if [ -f "$FACTORS_PATH" ]; then
            FACTOR_ARGS+=(--factors "$FACTORS_PATH")
            if [ "$(factors_has_r "$FACTORS_PATH")" -eq 0 ]; then
                CALIB_TAG="noR_"
            fi
        fi
    else
        FACTORS_PATH="$FACTORS_DIR/ffn_lowrank_factors_r${CURRENT_RANK}.safetensors"
        if [ ! -f "$FACTORS_PATH" ]; then
            echo "[factors] ERROR: PRECOMPUTE_FACTORS=0 but missing $FACTORS_PATH" >&2
            exit 1
        fi
        FACTOR_ARGS+=(--factors "$FACTORS_PATH")
        if [ "$(factors_has_r "$FACTORS_PATH")" -eq 0 ]; then
            CALIB_TAG="noR_"
        fi
    fi

    echo ""
    echo "========================================="
    echo "==== LR-PT-${COMPILE_TAG}-${Q_FP8_TAG} rank=${CURRENT_RANK} seed=${CURRENT_SEED} ===="
    echo "========================================="
    echo ""
    local OUTPUT_VIDEO_BASE="ffn_delta_outputs/validate_r${CURRENT_RANK}_s${CURRENT_SEED}_f${FRAMES}_${COMPILE_TAG}_${Q_FP8_TAG}_lr-pt_${CALIB_TAG}${PROMPT_ID}.mp4"
    local OUTPUT_VIDEO
    OUTPUT_VIDEO="$(unique_output_path "$OUTPUT_VIDEO_BASE")"

    env PYTORCH_ALLOC_CONF=expandable_segments:True \
        PATH=/root/miniconda3/envs/ltx-ffn/bin:$PATH \
        USE_LITE_LINEAR=1 \
        ./ffn/4_decompose.sh \
        --pipeline_config "$PIPELINE_CONFIG" \
        --r_path "$R_PATH" \
        --blocks "$BLOCKS" \
        "${COMPILE_ARGS[@]}" \
        "${BENCHMARK_ARGS[@]}" \
        "${FACTOR_ARGS[@]}" \
        --output_video "$OUTPUT_VIDEO" \
        --rank "$CURRENT_RANK" \
        --num_frames "$FRAMES" \
        "${EXTRA_INFER_ARGS[@]}" \
        --use_lite_linear \
        --no_kernel \
        --skip_baseline
}

echo "========================================="
echo "Fast Validation"
echo "Rank: $RANK"
echo "Frames: $FRAMES"
echo "Ranks: $RANKS"
echo "Baseline: $BASELINE"
echo "Compile: $COMPILE"
echo "LR-PT: $LR_PT"
echo "Only issues: $ONLY_ISSUES"
echo "Precompute factors: $PRECOMPUTE_FACTORS"
echo "Factors dir: $FACTORS_DIR"
echo "========================================="

INDEX=1
while IFS= read -r entry; do
    ENTRY_SEED=""
    CURRENT_PROMPT=""
    ISSUE_FLAG="0"
    IFS="|" read -r ENTRY_SEED CURRENT_PROMPT ISSUE_FLAG <<< "$entry"
    ISSUE_FLAG="${ISSUE_FLAG:-0}"
    if [ "$ONLY_ISSUES" -eq 1 ] && [ "$ISSUE_FLAG" -ne 1 ]; then
        continue
    fi
    PROMPT_ID=$(echo "$CURRENT_PROMPT" | head -c 50 | tr -cd '[:alnum:]_' | head -c 30)
    if [ -z "$PROMPT_ID" ]; then
        PROMPT_ID="prompt${INDEX}"
    fi

    echo ""
    echo ">>> Prompt $INDEX (Seed: ${ENTRY_SEED:-$SEED})"
    for CURRENT_RANK in $RANKS; do
        run_cuda "$ENTRY_SEED" "$CURRENT_PROMPT" "$PROMPT_ID" "$CURRENT_RANK"
        if [ "$LR_PT" -eq 1 ]; then
            run_lr_pt "$ENTRY_SEED" "$CURRENT_PROMPT" "$PROMPT_ID" "$CURRENT_RANK"
        fi
    done
    if [ "$BASELINE" -eq 1 ]; then
        run_baseline "$ENTRY_SEED" "$CURRENT_PROMPT" "$PROMPT_ID"
    fi
    INDEX=$((INDEX + 1))
done <<< "$PROMPTS"

echo ""
echo "Validation complete!"
echo "Outputs in: ffn_delta_outputs/"
echo ""
echo "========================================="
echo "==== RUN_ID ${RUN_ID} END ===="
echo "Timestamp: $(date -Is)"
echo "========================================="
echo ""

python3 - <<'PY' "$RUN_ID" "validate_p.log" "validate_results.log"
import re
import sys

run_id = sys.argv[1]
log_path = sys.argv[2]
out_path = sys.argv[3]

start_marker = f"==== RUN_ID {run_id} START ===="
end_marker = f"==== RUN_ID {run_id} END ===="

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.read().splitlines()

try:
    start_idx = lines.index(start_marker)
    end_idx = lines.index(end_marker, start_idx + 1)
except ValueError:
    sys.exit(0)

section = lines[start_idx + 1 : end_idx]

backend_re = re.compile(
    r"^==== (LR-CUDA|LR-PT|BASELINE)-([^-]+)-([^- ]+)(?: rank=(\d+))? seed=(\d+) ====$"
)
bench_decomp_re = re.compile(r"\[bench\] Decomposed: ([0-9.]+) s/run")
bench_base_re = re.compile(r"\[bench\] Baseline: ([0-9.]+) s/run")

current = None
summary = {}

for line in section:
    m = backend_re.match(line)
    if m:
        backend, compile_tag, cfg_tag, rank, seed = m.groups()
        current = (backend, compile_tag, cfg_tag, rank, seed)
        continue

    if current and bench_decomp_re.search(line):
        time = bench_decomp_re.search(line).group(1)
        backend, compile_tag, cfg_tag, rank, seed = current
        key = (backend, compile_tag, cfg_tag, rank)
        summary.setdefault(key, []).append((seed, time))
        current = None
        continue

    if current and bench_base_re.search(line):
        time = bench_base_re.search(line).group(1)
        backend, compile_tag, cfg_tag, rank, seed = current
        key = (backend, compile_tag, cfg_tag, rank)
        summary.setdefault(key, []).append((seed, time))
        current = None

def label(key):
    backend, compile_tag, cfg_tag, rank = key
    compiled = "compiled" if compile_tag == "default" else "eager"
    if backend == "BASELINE":
        return f"Baseline ({compiled}, {cfg_tag} config)"
    return f"{backend} ({compiled}, rank {rank}, {cfg_tag})"

lines = []
if summary:
    lines.append("============")
    for key in summary:
        lines.append(label(key))
        for seed, time in summary[key]:
            metric = "Baseline" if key[0] == "BASELINE" else "Decomposed"
            lines.append(f"Seed {seed}: [bench] {metric}: {time} s/run")
    lines.append("============")
else:
    lines.append("============")
    lines.append("No benchmark entries found for this RUN_ID.")
    lines.append("============")

text = "\n".join(lines)
print(text)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text + "\n")
PY
