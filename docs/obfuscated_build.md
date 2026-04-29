# Obfuscated Wheel Build

This document describes the wheel-only obfuscated build pipeline for `LiteLinear`.

## What This Pipeline Produces

- A platform wheel that includes:
  - Compiled native runtime modules
  - Cython-compiled core package modules (`linear`, `checkpoint_patch_core`, `ffn_delta`, `ffn_patch`)
  - PyArmor-obfuscated `lite_linear/__init__.py`
  - Plaintext support modules `lite_linear/offline_patch.py` and `lite_linear/patched_checkpoint.py`
- No native source files in the wheel payload.

## Prerequisites

- CUDA toolkit available in your environment (`CUDA_HOME` set when needed)
- NVIDIA driver/runtime compatible with your CUDA toolkit
- Python build environment with:
  - `build`, `wheel`, `setuptools`
  - `Cython`
  - `pyarmor`
- A CUDA-enabled PyTorch install

## 1) Create Obfuscated Build/Test Environment

```bash
./scripts/create_obf_env.sh litelinear-obf
```

Optional: pick a different PyTorch wheel index:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./scripts/create_obf_env.sh litelinear-obf
```

Optional: override the Python version (the LTX-2 runner in this fork uses Python 3.10):

```bash
PYTHON_VERSION=3.10 ./scripts/create_obf_env.sh litelinear-obf
```

## 2) Build Obfuscated Wheel

Run inside the conda env, or point `PYTHON_BIN` at an existing project venv:

```bash
conda run -n litelinear-obf ./scripts/build_obfuscated_wheel.sh
```

```bash
PYTHON_BIN=.venv/bin/python ./scripts/build_obfuscated_wheel.sh
```

Artifact output:

- `dist/*.whl`

The build script validates wheel contents with `scripts/validate_wheel_contents.py` and fails if:

- Native sources/headers are present in the wheel
- Plaintext core modules are present
- Expected binary modules are missing
- `lite_linear/__init__.py` is not PyArmor-obfuscated
- `lite_linear.offline_patch` or its supporting Python modules are missing from the wheel

## 3) Test Installed Wheel with Benchmark

```bash
./scripts/test_obfuscated_wheel.sh litelinear-obf dist/<your-wheel>.whl
```

This script:

- Reinstalls the built wheel into the chosen conda env
- Verifies `torch.cuda.is_available()`
- Verifies `import lite_linear`, `import lite_linear.checkpoint_patch_core`, and `import lite_linear.offline_patch`
- Verifies `python -m lite_linear.offline_patch --help`
- Builds a tiny synthetic patched checkpoint and verifies online redirect through `safetensors.safe_open`
- Runs `extras/bench_ffn.py`

## 4) Run LTX-2 Against The Wheel

After building the wheel into `dist/`, the runner can consume it without changing the workspace dependency sources:

```bash
./scripts/runners/run_ltx2.sh
```

The runner resolves the newest wheel from `libs/LiteLinear/dist/` and invokes:

```bash
uv run --with <wheel> --no-sources-package lite-linear python -m ltx_pipelines.ti2vid_two_stages
```

This keeps normal editable development intact while making the inference run use the built wheel instead of `libs/LiteLinear` source code.

## Troubleshooting

- Native runtime import errors
  - Ensure CUDA toolkit is visible and PyTorch is CUDA-enabled.
  - Rebuild wheel in a compatible CUDA build environment.
- Wheel validation fails due to source leakage
  - Re-run `scripts/build_obfuscated_wheel.sh` and inspect stage tree at `build/obfuscated_stage`.
- Benchmark fails with CUDA runtime errors
  - Confirm driver/runtime compatibility and that the selected env has the right CUDA-enabled torch wheel.
