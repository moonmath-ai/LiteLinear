#!/usr/bin/env python
"""
Profile LiteLinear (fused_forward) vs te.Linear for distinct shapes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from contextlib import contextmanager

from lite_linear import LiteLinear

try:
    import transformer_engine.pytorch as te

    _HAS_TE = True
    _TE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    te = None
    _HAS_TE = False
    _TE_IMPORT_ERROR = exc


REAL_CONFIGS = [
    ("w1", 1400, 4096, 16384),
    ("w1", 5600, 4096, 16384),
    ("w2", 1400, 16384, 4096),
    ("w2", 5600, 16384, 4096),
]


def _shape_name(n: int, k: int) -> str:
    if n == 16384 and k == 4096:
        return "w1"
    if n == 4096 and k == 16384:
        return "w2"
    return f"N{n}_K{k}"


def _get_field(entry: dict, key: str) -> int:
    if key in entry:
        return entry[key]
    key_l = key.lower()
    if key_l in entry:
        return entry[key_l]
    raise KeyError(f"Missing field '{key}' in shape entry: {entry}")


def load_configs_from_json(path: Path) -> list[tuple[str, int, int, int]]:
    data = json.loads(path.read_text())
    configs: list[tuple[str, int, int, int]] = []
    for entry in data:
        m = int(_get_field(entry, "M"))
        n = int(_get_field(entry, "N"))
        k = int(_get_field(entry, "K"))
        configs.append((_shape_name(n, k), m, k, n))
    return configs


def distinct_configs(
    configs: Iterable[tuple[str, int, int, int]]
) -> list[tuple[str, int, int, int]]:
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[str, int, int, int]] = []
    for name, m, k, n in configs:
        key = (m, k, n)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, m, k, n))
    return out


def _quantize_fp8_per_tensor(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This PyTorch build does not expose FP8 dtypes")
    amax = t.abs().max()
    fp8_max = 448.0
    scale = (fp8_max / amax.clamp(min=1e-12)).to(dtype=torch.float32)
    q_fp8 = (t * scale).to(dtype=torch.float8_e4m3fn)
    return q_fp8, scale.reciprocal().to(dtype=torch.float32)


def build_cudagraph_fn(fn, x: torch.Tensor):
    x_static = x.clone()
    fn(x_static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(x_static)

    def replay(x_ref=x_static, out_ref=out):
        graph.replay()
        return out_ref

    return replay


@contextmanager
def _nvtx_range(label: str):
    if hasattr(torch.cuda, "nvtx"):
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


@torch.no_grad()
def run_loops(fn, warmup: int, iters: int, *, nvtx_prefix: str | None = None) -> None:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if nvtx_prefix is None:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return
    with _nvtx_range(f"{nvtx_prefix}_section"):
        for i in range(iters):
            if i == 0:
                with _nvtx_range(f"{nvtx_prefix}_iter_0"):
                    fn()
            else:
                fn()
    torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile LiteLinear (fused_forward)")
    ap.add_argument("--iters", type=int, default=10, help="Profile iterations per shape")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup iterations per shape")
    ap.add_argument("--rank", type=int, default=64, help="Low-rank dimension R")
    ap.add_argument(
        "--shapes-json",
        type=str,
        default=None,
        help="Path to captured_shapes*.json; defaults to extras/captured_shapes_2.json if present",
    )
    ap.add_argument(
        "--distinct-shapes",
        action="store_true",
        help="Deduplicate shapes by (M,K,N)",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile (or cudagraphs replay) for the forward calls",
    )
    ap.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        help="torch.compile mode: default, reduce-overhead, max-autotune",
    )
    ap.add_argument(
        "--compile-backend",
        type=str,
        default="cudagraphs",
        help="torch.compile backend: inductor, cudagraphs, aot_eager, eager",
    )
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    shapes_path = Path(args.shapes_json) if args.shapes_json else None
    if shapes_path is None:
        default_path = Path(__file__).resolve().parent / "captured_shapes_2.json"
        shapes_path = default_path if default_path.exists() else None
    if args.shapes_json and shapes_path is not None and not shapes_path.exists():
        # Try resolving relative to this script directory for convenience.
        candidate = Path(__file__).resolve().parent / shapes_path
        if candidate.exists():
            shapes_path = candidate
        else:
            raise SystemExit(f"Shapes file not found: {shapes_path}")

    configs = REAL_CONFIGS
    if shapes_path is not None and shapes_path.exists():
        configs = load_configs_from_json(shapes_path)
        print(f"Loaded {len(configs)} shapes from {shapes_path}")
    if args.distinct_shapes:
        configs = distinct_configs(configs)
        print(f"Using {len(configs)} distinct shapes")
    else:
        print(f"Using {len(configs)} shapes")

    if not _HAS_TE:
        print(f"[profile] transformer_engine not available; skipping TE ({_TE_IMPORT_ERROR})")

    for name, m, k, n in configs:
        print(f"\n[shape] {name} M={m} K={k} N={n}")
        shape_label = f"shape_{name}_M{m}_K{k}_N{n}"
        with _nvtx_range(shape_label):
            x = torch.randn(m, k, device=device, dtype=dtype)

            lite = LiteLinear(k, n, bias=True, device=device, dtype=dtype, rank=args.rank)

            A = torch.randn(n, args.rank, device=device, dtype=dtype) * 0.1
            B = torch.randn(args.rank, k, device=device, dtype=dtype) * 0.1
            Q = torch.randn(n, k, device=device, dtype=dtype) * 0.1
            bias = torch.randn(n, device=device, dtype=dtype) * 0.01
            Q_fp8, Q_scale_inv = _quantize_fp8_per_tensor(Q)
            lite._install_factors(A=A, B=B, Q_fp8=Q_fp8, Q_scale_inv=Q_scale_inv, bias=bias)

            te_mod = None
            if _HAS_TE:
                try:
                    try:
                        te_mod = te.Linear(k, n, bias=True, device=device, params_dtype=dtype)
                    except TypeError:
                        te_mod = te.Linear(k, n, bias=True)
                        te_mod.to(device=device, dtype=dtype)
                    te_mod.eval()
                except Exception as exc:
                    print(f"[profile] TE init failed; skipping TE ({exc})")
                    te_mod = None

            def lite_call(inp: torch.Tensor) -> torch.Tensor:
                return lite(inp)

            lite_fn = None
            te_fn = None
            if args.compile:
                if args.compile_backend == "cudagraphs":
                    lite_fn = build_cudagraph_fn(lite_call, x)
                    if te_mod is not None:
                        te_fn = build_cudagraph_fn(lambda inp: te_mod(inp), x)
                else:
                    try:
                        lite_c = torch.compile(
                            lite, backend=args.compile_backend, mode=args.compile_mode
                        )
                        lite_fn = lambda: lite_c(x)
                    except Exception as exc:
                        print(f"[profile] LiteLinear torch.compile failed; using eager ({exc})")
                        lite_fn = lambda: lite(x)
                    if te_mod is not None:
                        try:
                            te_c = torch.compile(
                                te_mod, backend=args.compile_backend, mode=args.compile_mode
                            )
                            te_fn = lambda: te_c(x)
                        except Exception as exc:
                            print(f"[profile] TE torch.compile failed; using eager ({exc})")
                            te_fn = lambda: te_mod(x)
            else:
                lite_fn = lambda: lite(x)
                if te_mod is not None:
                    te_fn = lambda: te_mod(x)

            print(f"[profile] LiteLinear: iters={args.iters} warmup={args.warmup}")
            run_loops(
                lite_fn,
                warmup=args.warmup,
                iters=args.iters,
                nvtx_prefix="lite_linear",
            )
            if te_fn is not None:
                print(f"[profile] TE Linear: iters={args.iters} warmup={args.warmup}")
                run_loops(
                    te_fn, warmup=args.warmup, iters=args.iters, nvtx_prefix="TE"
                )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
