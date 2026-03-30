#!/usr/bin/env python
"""
Profile LowRankDeltaLinear forward_cuda (lr-cu) for distinct shapes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

from lite_linear.ffn_delta import LowRankDeltaLinear


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


@torch.no_grad()
def run_loops(fn, warmup: int, iters: int) -> None:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile LowRankDeltaLinear (lr-cu)")
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
        "--compile",
        action="store_true",
        help="Use torch.compile (or cudagraphs replay) for the forward call",
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

    configs = REAL_CONFIGS
    if shapes_path is not None and shapes_path.exists():
        configs = load_configs_from_json(shapes_path)
        print(f"Loaded {len(configs)} shapes from {shapes_path}")
    configs = distinct_configs(configs)
    print(f"Using {len(configs)} distinct shapes")

    for name, m, k, n in configs:
        print(f"\n[shape] {name} M={m} K={k} N={n}")
        x = torch.randn(m, k, device=device, dtype=dtype)
        A = torch.randn(n, args.rank, device=device, dtype=dtype) * 0.1
        B = torch.randn(args.rank, k, device=device, dtype=dtype) * 0.1
        Q = torch.randn(n, k, device=device, dtype=dtype) * 0.1
        bias = torch.randn(n, device=device, dtype=dtype) * 0.01

        mod = LowRankDeltaLinear(
            A=A,
            B=B,
            Q=Q,
            bias=bias,
            quantize_q=True,
            fp8_dtype=torch.float8_e4m3fn,
            static_input_scale=1.0,
            layer_name=f"profile_{name}_{m}",
        ).to(device)
        mod.eval()

        def lr_cuda_call(inp: torch.Tensor) -> torch.Tensor:
            return mod.forward_cuda(inp)

        if args.compile:
            if args.compile_backend == "cudagraphs":
                lr_fn = build_cudagraph_fn(lr_cuda_call, x)
            else:
                try:
                    lr_c = torch.compile(
                        lr_cuda_call, backend=args.compile_backend, mode=args.compile_mode
                    )
                    lr_fn = lambda: lr_c(x)
                except Exception as exc:
                    print(f"[profile] LR-CUDA torch.compile failed; using eager ({exc})")
                    lr_fn = lambda: mod.forward_cuda(x)
        else:
            lr_fn = lambda: mod.forward_cuda(x)

        print(f"[profile] LR-CUDA: iters={args.iters} warmup={args.warmup}")
        run_loops(lr_fn, warmup=args.warmup, iters=args.iters)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
