#!/usr/bin/env python
"""
Benchmark Linear vs LowRankDeltaLinear:

  - Linear baseline: F.linear(x, W, bias)
  - LowRankDeltaLinear (PT fallback): HAS_CUDA_EXT=False
  - LowRankDeltaLinear (CUDA fused): HAS_CUDA_EXT=True

Shapes match real LTX 13B FFN sizes captured from validate.sh.
"""

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
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


R_DEFAULT = 64

# (name, M, K, N, call_count)
REAL_CONFIGS = [
    ("w1", 1400, 4096, 16384, 195),
    ("w1", 5600, 4096, 16384, 84),
    ("w2", 1400, 16384, 4096, 196),
    ("w2", 5600, 16384, 4096, 84),
]


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


def load_configs_from_json(path: str) -> list[tuple[str, int, int, int, int]]:
    data = json.loads(Path(path).read_text())
    configs = []
    for entry in data:
        m = int(_get_field(entry, "M"))
        n = int(_get_field(entry, "N"))
        k = int(_get_field(entry, "K"))
        calls = int(entry.get("count", 1))
        configs.append((_shape_name(n, k), m, k, n, calls))
    return configs


_EMPTY_CUDAGRAPH_WARNED: set[str] = set()


def report_empty_cudagraph(name: str, caught: list[warnings.WarningMessage]) -> None:
    if any("The CUDA Graph is empty" in str(w.message) for w in caught):
        if name not in _EMPTY_CUDAGRAPH_WARNED:
            print(f"[bench] cudagraphs empty for {name}")
            _EMPTY_CUDAGRAPH_WARNED.add(name)


def bench(
    fn,
    warmup=20,
    iters=100,
    *,
    name: str | None = None,
    log_empty_cudagraphs: bool = False,
):
    def _run() -> tuple[float, float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        start_wall = time.perf_counter()
        for i in range(iters):
            fn()
            if i == iters - 1:
                end_event.record()
                torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start_wall) * 1000.0
        event_ms = start_event.elapsed_time(end_event)
        return wall_ms / iters, event_ms / iters

    if log_empty_cudagraphs:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ms = _run()
        if name:
            report_empty_cudagraph(name, caught)
        return ms

    return _run()


def _bench_ms(t_ms: tuple[float, float] | float | None) -> float | None:
    if t_ms is None:
        return None
    if isinstance(t_ms, tuple):
        return max(t_ms[0], t_ms[1])
    return t_ms


def _add_total(
    total: tuple[float, float] | None,
    t_ms: tuple[float, float] | None,
    calls: int,
) -> tuple[float, float] | None:
    if t_ms is None:
        return total
    if total is None:
        return (t_ms[0] * calls, t_ms[1] * calls)
    return (total[0] + t_ms[0] * calls, total[1] + t_ms[1] * calls)


def pct_faster(
    baseline_ms: tuple[float, float] | float | None,
    candidate_ms: tuple[float, float] | float | None,
) -> float | None:
    """
    Return percent faster vs baseline (positive means faster, negative means slower).
      pct = (baseline - candidate) / baseline * 100
    """
    baseline = _bench_ms(baseline_ms)
    candidate = _bench_ms(candidate_ms)
    if baseline is None or candidate is None or baseline <= 0.0:
        return None
    return (baseline - candidate) / baseline * 100.0


def fmt_us_dual(t_ms: tuple[float, float] | None, width: int = 14) -> str:
    if t_ms is None:
        return f"{'n/a':>{width}}"
    bench_us = _bench_ms(t_ms) * 1000.0
    return f"{bench_us:.0f}".rjust(width)


def fmt_ms_dual(t_ms: tuple[float, float] | None, width: int = 14) -> str:
    if t_ms is None:
        return f"{'n/a':>{width}}"
    return f"{_bench_ms(t_ms):.0f}".rjust(width)


def fmt_pct(pct: float | None, width: int = 8) -> str:
    if pct is None:
        return f"{'n/a':>{width}}"
    return f"{pct:+{width - 1}.1f}%"


def reset_compile_state_if_needed(backend: str) -> None:
    if backend != "cudagraphs":
        return
    # cudagraphs backend is sensitive to shape changes across compiled calls.
    # Reset caches between shapes to avoid reusing stale graphs.
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        if hasattr(torch, "_inductor") and hasattr(torch._inductor, "cudagraph_trees"):
            torch._inductor.cudagraph_trees.clear_cudagraph_cache()  # type: ignore[attr-defined]
    except Exception:
        pass


def precompile_shape(
    named_fns: list[tuple[str, callable]],
    *,
    log_empty_cudagraphs: bool = False,
) -> None:
    # Force compilation/cudagraph capture for this shape before timing.
    for name, fn in named_fns:
        if fn is None:
            continue
        if log_empty_cudagraphs:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fn()
            report_empty_cudagraph(name, caught)
        else:
            fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_cudagraph_fn(fn, x: torch.Tensor):
    # Build a CUDA graph replay wrapper for a fixed-shape input.
    x_static = x.clone()
    # Warmup
    fn(x_static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(x_static)

    def replay(x_ref=x_static, out_ref=out):
        graph.replay()
        return out_ref

    return replay


def main():
    ap = argparse.ArgumentParser(description="Benchmark LowRankDeltaLinear vs Linear")
    ap.add_argument("--r", type=int, default=R_DEFAULT, help="Low-rank dimension R")
    ap.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    ap.add_argument("--iters", type=int, default=100, help="Timed iterations")
    ap.add_argument("--compile", action="store_true", help="Use torch.compile for all three paths")
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
        help="Path to captured_shapes*.json to override default benchmark shapes",
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
        "--bench-lr-only",
        action="store_true",
        help="Benchmark only LR-PT and LR-CUDA (skip Linear/TE timing)",
    )
    ap.add_argument(
        "--bench-cuda-only",
        action="store_true",
        help="Benchmark only LR-CUDA timing (skip Linear/TE/LR-PT timing)",
    )
    args = ap.parse_args()
    if args.bench_cuda_only:
        args.bench_lr_only = True

    device = torch.device("cuda")
    dtype = torch.bfloat16

    factors = None
    if args.factor_source == "load":
        if not args.factors:
            raise SystemExit("--factors is required when --factor-source=load")
        factors, metadata = load_lowrank_factors(args.factors)
        print(f"[bench] Loaded factors ({len(factors)} layers), meta={metadata}")

    print(
        f"LowRankDeltaLinear Benchmark | R={args.r} | source={args.factor_source} | "
        f"compile={args.compile} backend={args.compile_backend} mode={args.compile_mode}"
    )
    if not _HAS_TE:
        print(f"[bench] transformer_engine not available; skipping TE Linear ({_TE_IMPORT_ERROR})")
    if args.compile and args.compile_backend == "cudagraphs":
        print("[bench] cudagraphs backend: LR-PT/LR-CUDA use manual CUDA graph capture")
    if args.bench_lr_only:
        if args.bench_cuda_only:
            print("=" * 64)
            print(
                f"{'Cfg':<4} {'M':>5} {'Count':>5} {'LR-CUDA':>10}"
            )
            print(
                f"{'':<4} {'':>5} {'':>5} {'(us)':>10}"
            )
            print("[units] per-shape rows: us | TOTAL row: ms")
            print("-" * 64)
        else:
            print("=" * 92)
            print(
                f"{'Cfg':<4} {'M':>5} {'Count':>5} {'LR-PT':>10} {'LR-CUDA':>10} {'CUDA vs PT':>12}"
            )
            print(
                f"{'':<4} {'':>5} {'':>5} {'(us)':>10} {'(us)':>10} {'(%)':>12}"
            )
            print("[units] per-shape rows: us | TOTAL row: ms")
            print("-" * 92)
    else:
        print("=" * 146)
        print(
            f"{'Cfg':<4} {'M':>5} {'Count':>5} {'Linear':>8} {'TE-Linear':>10} {'LR-PT':>8} {'LR-CUDA':>8} {'TE vs Lin':>10} {'PT vs Lin':>10} {'CUDA vs Lin':>12}"
        )
        print(
            f"{'':<4} {'':>5} {'':>5} {'(us)':>8} {'(us)':>10} {'(us)':>8} {'(us)':>8} {'(%)':>10} {'(%)':>10} {'(%)':>12}"
        )
        print("[units] per-shape rows: us | TOTAL row: ms")
        print("-" * 146)

    total_linear: tuple[float, float] | None = None
    total_pt: tuple[float, float] | None = None
    total_cuda: tuple[float, float] | None = None
    total_te: tuple[float, float] | None = None
    total_calls = 0
    geo_cuda_log_sum = 0.0
    geo_weight_sum = 0

    configs = REAL_CONFIGS
    if args.shapes_json:
        configs = load_configs_from_json(args.shapes_json)
        print(f"Loaded {len(configs)} shapes from {args.shapes_json}")

    for name, m, k, n, calls in configs:
        if args.compile:
            reset_compile_state_if_needed(args.compile_backend)
        total_calls += int(calls)
        # Inputs
        x = torch.randn(m, k, device=device, dtype=dtype)

        # Linear baseline weights
        W = torch.randn(n, k, device=device, dtype=dtype) * 0.1
        bias = torch.randn(n, device=device, dtype=dtype) * 0.01

        Q_fp8 = None
        Q_scale_inv = None
        if args.factor_source == "random":
            A = torch.randn(n, args.r, device=device, dtype=dtype) * 0.1
            B = torch.randn(args.r, k, device=device, dtype=dtype) * 0.1
            Q = torch.randn(n, k, device=device, dtype=dtype) * 0.1
        elif args.factor_source == "decompose":
            A, B, Q = decompose_weight(W, rank=args.r)
            A = A.to(device=device, dtype=dtype)
            B = B.to(device=device, dtype=dtype)
            Q = Q.to(device=device, dtype=dtype)
        elif args.factor_source == "load":
            if factors is None:
                raise ValueError("factor_source=load requires factors to be provided")
            part = name if name in ("w1", "w2") else _infer_part(n, k)
            key, entry = _select_factor_entry(
                factors, part=part, block=args.factor_block, layer_name=args.factor_layer
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
            raise ValueError(f"Unknown factor_source: {args.factor_source}")

        linear_mod = None
        te_mod = None
        if not args.bench_lr_only:
            # Linear module (so we can torch.compile it)
            linear_mod = nn.Linear(k, n, bias=True, device=device, dtype=dtype)
            with torch.no_grad():
                linear_mod.weight.copy_(W)
                linear_mod.bias.copy_(bias)
            linear_mod.eval()

            if _HAS_TE:
                try:
                    try:
                        te_mod = te.Linear(
                            k, n, bias=True, device=device, params_dtype=dtype
                        )
                    except TypeError:
                        te_mod = te.Linear(k, n, bias=True)
                        te_mod.to(device=device, dtype=dtype)
                    with torch.no_grad():
                        te_mod.weight.copy_(W)
                        if te_mod.bias is not None:
                            te_mod.bias.copy_(bias)
                    te_mod.eval()
                except Exception as exc:
                    print(f"[bench] TE Linear init failed; skipping TE for this run ({exc})")
                    te_mod = None

        # Single module: benchmark both paths without duplicating weights/Q.
        mod = LowRankDeltaLinear(
            A=A,
            B=B,
            Q=Q,
            Q_fp8=Q_fp8 if args.factor_source == "load" else None,
            Q_scale_inv=Q_scale_inv if args.factor_source == "load" else None,
            bias=bias,
            quantize_q=True,
            fp8_dtype=torch.float8_e4m3fn,
            static_input_scale=1.0,  # match kernel semantics
            layer_name=f"bench_{name}_{m}",
        ).to(device)
        mod.eval()
        # Free the original BF16 remainder (we keep only the FP8 quantized copy inside the module).
        del Q

        # Linear
        def linear_fn():
            return linear_mod(x)

        def te_fn():
            if te_mod is None:
                return None
            return te_mod(x)

        # LowRankDeltaLinear forward (PT fallback vs CUDA fused)
        def lr_pt_fn():
            return mod.forward_pt(x)

        def lr_cuda_fn():
            return mod.forward_cuda(x)

        lr_pt_graph = None
        lr_cuda_graph = None
        if args.compile and args.compile_backend == "cudagraphs":
            lr_pt_graph = build_cudagraph_fn(mod.forward_pt, x)
            lr_cuda_graph = build_cudagraph_fn(mod.forward_cuda, x)

        lr_pt_call_c = None
        lr_cuda_call_c = None
        linear_fn_compiled = None
        te_mod_c = None

        if args.compile:
            def lr_pt_call(inp: torch.Tensor) -> torch.Tensor:
                return mod.forward_pt(inp)

            def lr_cuda_call(inp: torch.Tensor) -> torch.Tensor:
                return mod.forward_cuda(inp)

            if args.compile_backend != "cudagraphs":
                if args.compile_backend == "inductor":
                    lr_pt_call_c = torch.compile(
                        lr_pt_call, backend=args.compile_backend, mode=args.compile_mode
                    )
                    lr_cuda_call_c = torch.compile(
                        lr_cuda_call, backend=args.compile_backend, mode=args.compile_mode
                    )
                else:
                    lr_pt_call_c = torch.compile(lr_pt_call, backend=args.compile_backend)
                    lr_cuda_call_c = torch.compile(lr_cuda_call, backend=args.compile_backend)

            if not args.bench_lr_only:
                if args.compile_backend == "inductor":
                    linear_mod_c = torch.compile(
                        linear_mod, backend=args.compile_backend, mode=args.compile_mode
                    )
                else:
                    linear_mod_c = torch.compile(linear_mod, backend=args.compile_backend)

                def linear_fn_compiled():
                    return linear_mod_c(x)

            def lr_pt_fn_compiled():
                if lr_pt_graph is not None:
                    return lr_pt_graph()
                if lr_pt_call_c is not None:
                    return lr_pt_call_c(x)
                return lr_pt_fn()

            def lr_cuda_fn_compiled():
                if lr_cuda_graph is not None:
                    return lr_cuda_graph()
                if lr_cuda_call_c is not None:
                    return lr_cuda_call_c(x)
                return lr_cuda_fn()

            if (not args.bench_lr_only) and te_mod is not None:
                try:
                    if args.compile_backend == "inductor":
                        te_mod_c = torch.compile(
                            te_mod, backend=args.compile_backend, mode=args.compile_mode
                        )
                    else:
                        te_mod_c = torch.compile(te_mod, backend=args.compile_backend)
                except Exception as exc:
                    print(f"[bench] TE torch.compile failed; using eager ({exc})")
                    te_mod_c = None
        else:
            linear_fn_compiled = None if args.bench_lr_only else linear_fn
            lr_pt_fn_compiled = lr_pt_fn
            lr_cuda_fn_compiled = lr_cuda_fn

        if args.compile:
            precompile_items: list[tuple[str, callable]] = []
            if linear_fn_compiled is not None:
                precompile_items.append(("linear", linear_fn_compiled))
            if te_mod_c is not None:
                precompile_items.append(("te", lambda: te_mod_c(x)))
            if (not args.bench_cuda_only) and lr_pt_call_c is not None:
                precompile_items.append(("lr-pt", lr_pt_fn_compiled))
            if lr_cuda_call_c is not None:
                precompile_items.append(("lr-cu", lr_cuda_fn_compiled))
            if precompile_items:
                precompile_shape(
                    precompile_items,
                    log_empty_cudagraphs=(args.compile_backend == "cudagraphs"),
                )

        log_empty = args.compile_backend == "cudagraphs"
        t_linear = None
        if linear_fn_compiled is not None:
            t_linear = bench(
                linear_fn_compiled,
                warmup=args.warmup,
                iters=args.iters,
                name="linear",
                log_empty_cudagraphs=log_empty,
            )
        t_te = None
        if (not args.bench_lr_only) and te_mod is not None:
            if te_mod_c is not None:
                t_te = bench(
                    lambda: te_mod_c(x),
                    warmup=args.warmup,
                    iters=args.iters,
                    name="te",
                    log_empty_cudagraphs=log_empty,
                )
            else:
                t_te = bench(
                    te_fn,
                    warmup=args.warmup,
                    iters=args.iters,
                    name="te",
                    log_empty_cudagraphs=log_empty,
                )
        t_pt = None
        if not args.bench_cuda_only:
            t_pt = bench(
                lr_pt_fn_compiled,
                warmup=args.warmup,
                iters=args.iters,
                name="lr-pt",
                log_empty_cudagraphs=log_empty,
            )
        t_cuda = bench(
            lr_cuda_fn_compiled,
            warmup=args.warmup,
            iters=args.iters,
            name="lr-cu",
            log_empty_cudagraphs=log_empty,
        )

        pt_ms = _bench_ms(t_pt)
        cuda_ms = _bench_ms(t_cuda)
        if cuda_ms is not None and cuda_ms > 0.0:
            geo_cuda_log_sum += math.log(cuda_ms * 1000.0) * calls
            geo_weight_sum += calls

        total_linear = _add_total(total_linear, t_linear, calls)
        total_te = _add_total(total_te, t_te, calls)
        total_pt = _add_total(total_pt, t_pt, calls)
        total_cuda = _add_total(total_cuda, t_cuda, calls)

        if args.bench_cuda_only:
            print(f"{name:<4} {m:>5} {calls:>5} {fmt_us_dual(t_cuda, 10)}")
        elif args.bench_lr_only:
            cuda_vs_pt = pct_faster(t_pt, t_cuda)
            print(
                f"{name:<4} {m:>5} {calls:>5} {fmt_us_dual(t_pt, 10)} {fmt_us_dual(t_cuda, 10)} {fmt_pct(cuda_vs_pt, 12)}"
            )
        else:
            te_vs_lin = pct_faster(t_linear, t_te)
            pt_vs_lin = pct_faster(t_linear, t_pt)
            cuda_vs_lin = pct_faster(t_linear, t_cuda)
            print(
                f"{name:<4} {m:>5} {calls:>5} {fmt_us_dual(t_linear, 8)} {fmt_us_dual(t_te, 10)} {fmt_us_dual(t_pt, 8)} {fmt_us_dual(t_cuda, 8)} {fmt_pct(te_vs_lin, 10)} {fmt_pct(pt_vs_lin, 10)} {fmt_pct(cuda_vs_lin, 12)}"
            )

    if args.bench_lr_only:
        if args.bench_cuda_only:
            print("-" * 64)
            print(
                f"{'TOTAL':<4} {'':<5} {total_calls:>5} {fmt_ms_dual(total_cuda, 10)}"
            )
        else:
            print("-" * 92)
            total_cuda_vs_pt = pct_faster(total_pt, total_cuda)
            print(
                f"{'TOTAL':<4} {'':<5} {total_calls:>5} "
                f"{fmt_ms_dual(total_pt, 10)} {fmt_ms_dual(total_cuda, 10)} {fmt_pct(total_cuda_vs_pt, 12)}"
            )
        if geo_weight_sum > 0:
            geo_cuda_us = math.exp(geo_cuda_log_sum / geo_weight_sum)
            print(f"LR-CUDA GEOMEAN(us): {geo_cuda_us:.0f}")
        if args.bench_cuda_only:
            print("=" * 64)
        else:
            print("=" * 92)
    else:
        print("-" * 146)
        total_te_vs_lin = pct_faster(total_linear, total_te)
        total_pt_vs_lin = pct_faster(total_linear, total_pt)
        total_cuda_vs_lin = pct_faster(total_linear, total_cuda)
        print(
            f"{'TOTAL':<4} {'':<5} {total_calls:>5} "
            f"{fmt_ms_dual(total_linear, 8)} {fmt_ms_dual(total_te, 10)} "
            f"{fmt_ms_dual(total_pt, 8)} {fmt_ms_dual(total_cuda, 8)} "
            f"{fmt_pct(total_te_vs_lin, 10)} {fmt_pct(total_pt_vs_lin, 10)} {fmt_pct(total_cuda_vs_lin, 12)}"
        )
        print("=" * 146)

    return 0


if __name__ == "__main__":
    sys.exit(main())

