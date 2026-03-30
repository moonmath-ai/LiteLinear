#!/usr/bin/env python
"""
Verify LowRankDeltaLinear correctness:

We compare:
  1) Linear baseline: F.linear(x, W, bias)
  2) LowRankDeltaLinear (PyTorch path): HAS_CUDA_EXT=False
  3) LowRankDeltaLinear (CUDA fused kernel): HAS_CUDA_EXT=True

We primarily assert LR-PT ~= LR-CUDA (kernel integration correctness).
We also report LR-* vs Linear for context (expected to differ with FP8).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from lite_linear.ffn_delta import LowRankDeltaLinear, decompose_weight, load_lowrank_factors

try:
    import transformer_engine.pytorch as te

    _HAS_TE = True
    _TE_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - optional dependency
    te = None
    _HAS_TE = False
    _TE_IMPORT_ERROR = e


def _shape_name(n: int, k: int) -> str:
    if n == 16384 and k == 4096:
        return "w1"
    if n == 4096 and k == 16384:
        return "w2"
    return f"N{n}_K{k}"


def _infer_part(n: int, k: int) -> str | None:
    name = _shape_name(n, k)
    if name in ("w1", "w2"):
        return name
    return None


def _select_factor_entry(
    factors: dict[str, dict[str, torch.Tensor]],
    *,
    part: str | None,
    block: int,
    layer_name: str | None,
) -> tuple[str, dict[str, torch.Tensor]]:
    if layer_name is not None:
        if layer_name not in factors:
            raise KeyError(f"Requested layer not found in factors: {layer_name}")
        return layer_name, factors[layer_name]
    if part is None:
        raise KeyError("Cannot infer factor entry without part or layer_name")
    preferred = f"transformer_blocks.{block}.ff.{part}"
    if preferred in factors:
        return preferred, factors[preferred]
    for key in sorted(factors.keys()):
        if key.endswith(f".ff.{part}"):
            return key, factors[key]
    raise KeyError(f"No factors found for part={part} (block={block})")


def _get_field(entry: dict, key: str) -> int:
    if key in entry:
        return entry[key]
    key_l = key.lower()
    if key_l in entry:
        return entry[key_l]
    raise KeyError(f"Missing field '{key}' in shape entry: {entry}")


def load_configs_from_json(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    configs = []
    for entry in data:
        m = int(_get_field(entry, "M"))
        n = int(_get_field(entry, "N"))
        k = int(_get_field(entry, "K"))
        configs.append(
            {
                "name": f"{_shape_name(n, k)} M={m}",
                "m": m,
                "n": n,
                "k": k,
            }
        )
    return configs


def seed_list(iters: int, seed_override: int | None) -> list[int]:
    if seed_override is not None:
        return [seed_override] * iters
    # Deterministic seed sweep to make failures reproducible.
    preset = [84337, 34640, 31874, 78857, 94940]
    if iters <= len(preset):
        return preset[:iters]
    extra = torch.randint(0, 100000, (iters - len(preset),)).tolist()
    return preset + extra


def run_one(
    m: int,
    n: int,
    k: int,
    r: int,
    seed: int,
    *,
    device: torch.device,
    use_compile: bool,
    compile_mode: str,
    compile_backend: str,
    use_te: bool,
    factor_source: str,
    factors: dict[str, dict[str, torch.Tensor]] | None,
    factor_block: int,
    factor_layer: str | None,
):
    torch.manual_seed(seed)

    dtype = torch.bfloat16

    x = torch.randn(m, k, device=device, dtype=dtype)

    # Full Linear weights (baseline)
    W = torch.randn(n, k, device=device, dtype=dtype) * 0.1
    bias = torch.randn(n, device=device, dtype=dtype) * 0.01

    Q_fp8 = None
    Q_scale_inv = None
    if factor_source == "random":
        A = torch.randn(n, r, device=device, dtype=dtype) * 0.1
        B = torch.randn(r, k, device=device, dtype=dtype) * 0.1
        Q = torch.randn(n, k, device=device, dtype=dtype) * 0.1
    elif factor_source == "decompose":
        A, B, Q = decompose_weight(W, rank=r)
        A = A.to(device=device, dtype=dtype)
        B = B.to(device=device, dtype=dtype)
        Q = Q.to(device=device, dtype=dtype)
    elif factor_source == "load":
        if factors is None:
            raise ValueError("factor_source=load requires factors to be provided")
        part = _infer_part(n, k)
        key, entry = _select_factor_entry(
            factors, part=part, block=factor_block, layer_name=factor_layer
        )
        A = entry["A"].to(device=device, dtype=dtype)
        B = entry["B"].to(device=device, dtype=dtype)
        Q_fp8 = entry.get("Q_fp8")
        Q_scale_inv = entry.get("Q_scale_inv")
        Q = entry.get("Q")
        if Q_fp8 is not None:
            Q_fp8 = Q_fp8.to(device=device)
            if Q_scale_inv is None:
                raise KeyError(f"Missing Q_scale_inv for {key}")
            Q_scale_inv = Q_scale_inv.to(device=device, dtype=torch.float32)
        elif Q is not None:
            Q = Q.to(device=device, dtype=dtype)
        loaded_bias = entry.get("bias")
        if loaded_bias is not None:
            bias = loaded_bias.to(device=device, dtype=dtype)
        if A.shape[0] != n or B.shape[1] != k:
            raise ValueError(
                f"Loaded factors '{key}' shape mismatch: A={tuple(A.shape)} B={tuple(B.shape)} "
                f"expected n={n} k={k}"
            )
    else:
        raise ValueError(f"Unknown factor_source: {factor_source}")

    # Single module: run both paths without duplicating weights/Q.
    mod = LowRankDeltaLinear(
        A=A,
        B=B,
        Q=Q,
        Q_fp8=Q_fp8 if factor_source == "load" else None,
        Q_scale_inv=Q_scale_inv if factor_source == "load" else None,
        bias=bias,
        quantize_q=True,  # enable FP8 remainder path
        fp8_dtype=torch.float8_e4m3fn,
        static_input_scale=1.0,  # match kernel's x->fp8 (scale=1) semantics
        layer_name=f"verify_{m}_{n}_{k}",
    ).to(device)
    mod.eval()
    # Free the original BF16 remainder (we keep only the FP8 quantized copy inside the module).
    del Q

    te_mod = None
    if use_te:
        try:
            try:
                te_mod = te.Linear(k, n, bias=True, device=device, params_dtype=dtype)
            except TypeError:
                te_mod = te.Linear(k, n, bias=True)
                te_mod.to(device=device, dtype=dtype)
            with torch.no_grad():
                te_mod.weight.copy_(W)
                if te_mod.bias is not None:
                    te_mod.bias.copy_(bias)
            te_mod.eval()
        except Exception as exc:
            print(f"[verify] TE Linear init failed; skipping TE for this run ({exc})")
            te_mod = None

    def pt_call(inp: torch.Tensor) -> torch.Tensor:
        return mod.forward_pt(inp)

    def cuda_call(inp: torch.Tensor) -> torch.Tensor:
        return mod.forward_cuda(inp)

    if use_compile:
        if compile_backend == "inductor":
            pt_call_f = torch.compile(pt_call, backend=compile_backend, mode=compile_mode)
            cuda_call_f = torch.compile(cuda_call, backend=compile_backend, mode=compile_mode)
        else:
            pt_call_f = torch.compile(pt_call, backend=compile_backend)
            cuda_call_f = torch.compile(cuda_call, backend=compile_backend)
    else:
        pt_call_f = pt_call
        cuda_call_f = cuda_call

    with torch.no_grad():
        y_linear = F.linear(x, W, bias)

        y_pt = pt_call_f(x)
        y_cuda = cuda_call_f(x)
        y_te = te_mod(x) if te_mod is not None else None

    # Primary: kernel matches torch path
    diff = (y_cuda.float() - y_pt.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Context: LR vs Linear (expected non-zero due to FP8 + unrelated W)
    lin_diff = (y_pt.float() - y_linear.float()).abs()
    lin_max = lin_diff.max().item()
    lin_mean = lin_diff.mean().item()
    cuda_lin_diff = (y_cuda.float() - y_linear.float()).abs()
    cuda_lin_max = cuda_lin_diff.max().item()
    cuda_lin_mean = cuda_lin_diff.mean().item()

    te_lin_max = te_lin_mean = None
    if y_te is not None:
        te_lin_diff = (y_te.float() - y_linear.float()).abs()
        te_lin_max = te_lin_diff.max().item()
        te_lin_mean = te_lin_diff.mean().item()

    return {
        "max_diff_cuda_vs_pt": max_diff,
        "mean_diff_cuda_vs_pt": mean_diff,
        "max_diff_pt_vs_linear": lin_max,
        "mean_diff_pt_vs_linear": lin_mean,
        "max_diff_cuda_vs_linear": cuda_lin_max,
        "mean_diff_cuda_vs_linear": cuda_lin_mean,
        "max_diff_te_vs_linear": te_lin_max,
        "mean_diff_te_vs_linear": te_lin_mean,
    }


def main():
    ap = argparse.ArgumentParser(description="Verify LowRankDeltaLinear (PT vs CUDA) on real FFN shapes.")
    ap.add_argument("--iters", type=int, default=1, help="Random iterations per config")
    ap.add_argument("--r", type=int, default=64, help="Low-rank dimension R")
    # No quick mode: always use full stress configs.
    ap.add_argument("--seed", type=int, default=None, help="Fixed seed (for debugging)")
    ap.add_argument("--compile", action="store_true", help="Run the PT/CUDA paths under torch.compile")
    ap.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        help="torch.compile mode: default, reduce-overhead, max-autotune",
    )
    ap.add_argument(
        "--compile-backend",
        type=str,
        default="inductor",
        help="torch.compile backend: inductor, cudagraphs, aot_eager, eager",
    )
    ap.add_argument(
        "--shapes-json",
        type=str,
        default=None,
        help="Path to captured_shapes*.json to override default verification shapes",
    )
    ap.add_argument(
        "--factor-source",
        choices=["random", "decompose", "load"],
        default="random",
        help="How to build low-rank factors: random, decompose (SVD), or load",
    )
    ap.add_argument(
        "--factors",
        type=str,
        default=None,
        help="Path to precomputed low-rank factors (safetensors or pt)",
    )
    ap.add_argument(
        "--factor-block",
        type=int,
        default=0,
        help="Block index to load factors from (w1/w2 selection)",
    )
    ap.add_argument(
        "--factor-layer",
        type=str,
        default=None,
        help="Exact layer name to load factors from (overrides block/part selection)",
    )
    ap.add_argument(
        "--pt-max",
        type=float,
        default=1.5,
        help="Max abs diff allowed for CUDA vs PT (gating)",
    )
    ap.add_argument(
        "--pt-mean",
        type=float,
        default=0.05,
        help="Mean abs diff allowed for CUDA vs PT (gating)",
    )
    ap.add_argument(
        "--lin-max",
        type=float,
        default=130.0,
        help="Max abs diff allowed for CUDA vs Linear (gating)",
    )
    ap.add_argument(
        "--lin-mean",
        type=float,
        default=17.0,
        help="Mean abs diff allowed for CUDA vs Linear (gating)",
    )
    args = ap.parse_args()

    device = torch.device("cuda")

    factors = None
    if args.factor_source == "load":
        if not args.factors:
            raise SystemExit("--factors is required when --factor-source=load")
        factors, metadata = load_lowrank_factors(args.factors)
        print(f"[verify] Loaded factors ({len(factors)} layers), meta={metadata}")

    configs = [
        {"name": "w1 M=14000", "m": 14000, "n": 16384, "k": 4096},
        {"name": "w1 M=56000", "m": 56000, "n": 16384, "k": 4096},
        {"name": "w2 M=14000", "m": 14000, "n": 4096, "k": 16384},
        {"name": "w2 M=56000", "m": 56000, "n": 4096, "k": 16384},
    ]
    if args.shapes_json:
        configs = load_configs_from_json(args.shapes_json)
        print(f"Loaded {len(configs)} shapes from {args.shapes_json}")

    print("LowRankDeltaLinear Verification")
    print(
        f"R={args.r} iters={args.iters} compile={args.compile} backend={args.compile_backend} mode={args.compile_mode}"
    )
    print(
        f"CUDA vs PT thresholds: max<={args.pt_max} mean<={args.pt_mean} (gating)"
    )
    print(
        f"CUDA vs Linear thresholds: max<={args.lin_max} mean<={args.lin_mean} (gating)"
    )
    if not _HAS_TE:
        print(f"TE Linear: not available ({_TE_IMPORT_ERROR})")
    else:
        print("TE vs Linear thresholds: using CUDA-vs-Linear thresholds")
    print("=" * 70)

    # Warm up CUDA
    _ = torch.randn(16, 16, device=device) @ torch.randn(16, 16, device=device)
    torch.cuda.synchronize()

    all_ok = True
    use_te = _HAS_TE
    seeds = seed_list(args.iters, args.seed)
    for cfg in configs:
        name, m, n, k = cfg["name"], cfg["m"], cfg["n"], cfg["k"]
        print(f"\n--- {name} (M={m}, N={n}, K={k}) ---")

        for seed in seeds:
            stats = run_one(
                m,
                n,
                k,
                args.r,
                seed,
                device=device,
                use_compile=args.compile,
                compile_mode=args.compile_mode,
                compile_backend=args.compile_backend,
                use_te=use_te,
                factor_source=args.factor_source,
                factors=factors,
                factor_block=args.factor_block,
                factor_layer=args.factor_layer,
            )

            ok_cuda_pt = (
                stats["max_diff_cuda_vs_pt"] <= args.pt_max
                and stats["mean_diff_cuda_vs_pt"] <= args.pt_mean
            )
            ok_cuda_lin = (
                stats["max_diff_cuda_vs_linear"] <= args.lin_max
                and stats["mean_diff_cuda_vs_linear"] <= args.lin_mean
            )
            ok_te_lin = True
            if use_te and stats["max_diff_te_vs_linear"] is not None:
                ok_te_lin = (
                    stats["max_diff_te_vs_linear"] <= args.lin_max
                    and stats["mean_diff_te_vs_linear"] <= args.lin_mean
                )
            all_ok = all_ok and ok_cuda_pt and ok_cuda_lin and ok_te_lin
            status_pt = "✓" if ok_cuda_pt else "✗"
            status_lin = "✓" if ok_cuda_lin else "✗"
            if not use_te or stats["max_diff_te_vs_linear"] is None:
                status_te = "-"
            else:
                status_te = "✓" if ok_te_lin else "✗"
            print(
                f"  pt:{status_pt} lin:{status_lin} te:{status_te} seed={seed} | "
                f"cuda-vs-pt max={stats['max_diff_cuda_vs_pt']:.3f} mean={stats['mean_diff_cuda_vs_pt']:.3f} | "
                f"pt-vs-lin max={stats['max_diff_pt_vs_linear']:.3f} mean={stats['mean_diff_pt_vs_linear']:.3f} | "
                f"cuda-vs-lin max={stats['max_diff_cuda_vs_linear']:.3f} mean={stats['mean_diff_cuda_vs_linear']:.3f}"
                + (
                    ""
                    if not use_te or stats["max_diff_te_vs_linear"] is None
                    else (
                        f" | te-vs-lin max={stats['max_diff_te_vs_linear']:.3f} "
                        f"mean={stats['mean_diff_te_vs_linear']:.3f}"
                    )
                )
            )

    print("\n" + "=" * 70)
    if all_ok:
        print("All LowRankDeltaLinear PT-vs-CUDA checks PASSED!")
        return 0
    print("Some LowRankDeltaLinear PT-vs-CUDA checks FAILED!")
    return 1


if __name__ == "__main__":
    sys.exit(main())

