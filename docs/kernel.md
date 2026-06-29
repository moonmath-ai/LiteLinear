# Runtime Notes

This release intentionally omits native runtime internals from the public
documentation. Use the packaged wheels for deployment and the build
validation scripts to confirm the wheel payload policy.

## Compatibility

- Wheels are platform-specific and must match the target Python ABI.
- Use a CUDA-enabled PyTorch environment whose torch version matches the
  wheel build (the 0.3.0+cu128 wheels are built against torch 2.11 cu128;
  see `lite_linear-0.3.0+cu128.dist-info/METADATA` for the exact pin).
- Rebuild wheels when changing Python, platform, CUDA/PyTorch
  compatibility, or deployment hardware assumptions.

### Wheel filename convention

Wheel filenames follow the standard format
([PEP 491](https://peps.python.org/pep-0491/#file-name-convention)):

```
{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
```

LiteLinear does not use the optional build tag, so:

```
lite_linear-{version}+{flavor}-cp{py}-cp{py}-{platform}.whl
```

Where:

| Component | Meaning |
| --- | --- |
| `{distribution}` | `lite_linear` (import name `lite_linear`, distribution name `lite-linear`) |
| `{version}` | PEP 440 version (e.g. `0.3.0`, `0.2.0`) |
| `{flavor}` | local version label after `+` (PEP 440): `cu128` for NVIDIA, `rocm63` for AMD |
| `cp{py}-cp{py}` | Python tag / ABI tag — both identical for CPython ABI-tagged builds (`cp310`, `cp312`, …) |
| `{platform}` | platform tag (`linux_x86_64` for the wheels shipped here) |

For example:

- `lite_linear-0.3.0+cu128-cp310-cp310-linux_x86_64.whl` — 0.3.0 release,
  built for CUDA 12.8 torch wheels, Python 3.10.
- `lite_linear-0.2.0+cu128-cp312-cp312-linux_x86_64.whl` — 0.2.0 release,
  built for CUDA 12.8 torch wheels, Python 3.12.
- `lite_linear-0.1.0+rocm7-cp310-cp310-linux_x86_64.whl` — 0.1.0 release,
  built for ROCm 7 torch wheels, Python 3.10.

The PEP 440 local label (`+cu128`, `+rocm63`, `+rocm7`) is informational;
pip does not use it for resolution. Pip matches on the public version +
the Python / ABI / platform tags, so the CUDA wheels will be picked over
the ROCm wheels automatically as long as the host's PyTorch build is the
matching flavor.

## Validation

The release wheel validation checks that:

- Required runtime modules are present (`_cuda` for NVIDIA, `_rocm` for AMD).
- Source files for the native runtime are not included in the wheel
  payload (`lite_linear/csrc/`, `lite_linear/csrc_rocm/`, and any
  `.cu` / `.cpp` / `.cuh` / `.h` under `lite_linear/`).
- The public Python entrypoints needed by integration flows remain
  available (`lite_linear.LiteLinear`, `lite_linear.calibration`,
  `lite_linear.cli`, `lite_linear.converter`, `lite_linear.decompose`,
  `lite_linear.inspect`, `lite_linear.manifest`).

Run validation with:

```bash
python scripts/validate_wheel_contents.py --wheel dist/<wheel>.whl
```

(from the private build repo).

## Cross-platform FP8 variants

`Q_fp8` is platform-pinned: NVIDIA builds use `float8_e4m3fn` (max value
448), AMD ROCm builds use `float8_e4m3fnuz` (max value 240). PyTorch's
default `copy_` would silently cast between the two variants; the
`LiteLinear._check_fp8_dtype` `load_state_dict` pre-hook raises on a
mismatch and points at `lite-linear convert --fp8-dtype {e4m3fn,e4m3fnuz}`
to produce a checkpoint with the correct variant.

## Benchmarking

For stable timing comparisons:

- Warm up the target workload before collecting timings (the kernel's
  first call at each shape runs cuBLASLt heuristic selection).
- Compare against the same model, prompt, scheduler settings, precision,
  and hardware.
- Treat first-run setup costs separately from steady-state inference
  timings.

Useful entry points:

- `examples/bench_ffn.py` — kernel microbench (calls
  `lite_linear._cuda.fused_forward` directly) on the captured LTX-Video
  FFN shape set.
- `examples/bench_lrdelta.py` — module-level bench (`LiteLinear` vs
  `nn.Linear`, optional TE comparison).
- `examples/bench_lrdelta_amd.py` — same for the ROCm path.

## Known caveats

- The fused kernel hardcodes the x→FP8 cast at `scale=1.0`. Activations
  with `|x| > 448` (NVIDIA) / `|x| > 240` (AMD) saturate silently. If
  this is a problem for your workload, see the input-scale discussion in
  the upstream `lite_linear/linear.py` docstring.
- LiteLinear is inference-only: the autograd `Function` wrapping the
  fused kernel raises on `.backward()`.
- LiteLinear requires CUDA inputs; running `forward` on CPU raises.
