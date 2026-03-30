#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
set -e
set -o pipefail  # Catch errors through pipes
export PIP_CONSTRAINT=""

# Be robust to being invoked from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Cleaning build artifacts to ensure fresh compilation
echo "[rebuild] Cleaning build artifacts..."
rm -rf ../build
rm -rf ../lite_linear.egg-info

# Reinstalling
echo "[rebuild] Reinstalling lite-linear ..."
# Using --no-build-isolation to use current environment packages
# Using --force-reinstall to overwrite existing installation
PY_BIN="/root/miniconda3/envs/ltx-ffn/bin/python"
env PYTHONPATH=. "$PY_BIN" -m pip install -v ../ --no-build-isolation --no-deps --force-reinstall 2>&1 | tee rbld.log

echo "[rebuild] Done. Kernel updated."
