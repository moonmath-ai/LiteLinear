#!/usr/bin/env python
"""
AMD benchmark table for FFN projection shapes captured from LTX-Video.

Compares:
  - Lin: baseline torch.nn.Linear
  - ROCM: LiteLinear ROCm fused path

Units:
  - per-shape rows: us
  - TOTAL row: ms (count-weighted aggregate)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from lite_linear import LiteLinear


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


def load_configs_from_json(path: str) -> list[tuple[str, int, int, int, int]]:
    data = json.loads(Path(path).read_text())
    configs: list[tuple[str, int, int, int, int]] = []
    for entry in data:
        m = int(_get_field(entry, "M"))
        n = int(_get_field(entry, "N"))
        k = int(_get_field(entry, "K"))
        calls = int(entry.get("count", 1))
        configs.append((_shape_name(n, k), m, k, n, calls))
    return configs


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def fmt_us(ms: float, width: int = 7) -> str:
    return f"{ms * 1000:>{width}.0f}"


def fmt_ms(ms: float, width: int = 7) -> str:
    return f"{ms:>{width}.0f}"


def fmt_x(val: float, width: int = 7) -> str:
    return f"{val:>{width}.3f}x"


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark Lin vs ROCM LiteLinear on captured shapes")
    ap.add_argument("--shapes-json", type=str, default="captured_shapes_2.json")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0, help="Optional row limit for quick dry-runs")
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm is required for this benchmark")

    # Force production ROCm LiteLinear path.
    import os

    os.environ.setdefault("LITELINEAR_ENABLE_ROCM_EXT", "1")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    configs = load_configs_from_json(args.shapes_json)
    if args.limit > 0:
        configs = configs[: args.limit]

    print("Performance Benchmark on shapes captured from LTX-Video")
    print("Units:")
    print("  per-shape rows: us")
    print("  TOTAL row: ms")
    print("Column glossary:")
    print("  Cfg: FFN projection shape (w1 = up-proj, w2 = down-proj).")
    print("  M: flattened activation rows for that GEMM shape.")
    print("  Count: number of calls for that shape in the captured workload.")
    print("  Lin: baseline nn.Linear latency.")
    print("  ROCM: LiteLinear ROCm path latency.")
    print("  ROCM_x: speedup multiplier vs baseline torch.nn.linear (>1 is faster, <1 is slower).")
    print("  TOTAL: count-weighted aggregate across listed shapes.")
    print()
    print("Cfg      M  Count |    Lin   ROCM |  ROCM_x")
    print("---------------------------------------------")

    total_calls = 0
    total_lin_ms = 0.0
    total_rocm_ms = 0.0
    rows: list[dict] = []

    for cfg, m, k, n, count in configs:
        total_calls += int(count)
        x = torch.randn(m, k, device=device, dtype=dtype)

        lin_mod = nn.Linear(k, n, bias=True, device=device, dtype=dtype)

        rocm_mod = LiteLinear(k, n, bias=True, device=device, dtype=dtype, rank=args.r)
        with torch.no_grad():
            rocm_mod.weight.copy_(lin_mod.weight)
            if rocm_mod.bias is not None and lin_mod.bias is not None:
                rocm_mod.bias.copy_(lin_mod.bias)
        rocm_mod.materialize_from_weight(rank=args.r)

        t_lin = bench(lambda: lin_mod(x), warmup=args.warmup, iters=args.iters)
        t_rocm = bench(lambda: rocm_mod(x), warmup=args.warmup, iters=args.iters)
        rocm_x = t_lin / t_rocm

        total_lin_ms += t_lin * count
        total_rocm_ms += t_rocm * count

        print(
            f"{cfg:<4} {m:>6} {count:>6} | {fmt_us(t_lin):>6} {fmt_us(t_rocm):>6} | {fmt_x(rocm_x):>7}"
        )

        rows.append(
            {
                "cfg": cfg,
                "m": int(m),
                "k": int(k),
                "n": int(n),
                "count": int(count),
                "lin_us": t_lin * 1000.0,
                "rocm_us": t_rocm * 1000.0,
                "rocm_x": rocm_x,
            }
        )

    total_rocm_x = total_lin_ms / total_rocm_ms
    print("---------------------------------------------")
    print(f"TOTAL        {total_calls:>4} | {fmt_ms(total_lin_ms):>6} {fmt_ms(total_rocm_ms):>6} | {fmt_x(total_rocm_x):>7}")

    if args.out_json:
        out = {
            "rows": rows,
            "total": {
                "calls": total_calls,
                "lin_ms": total_lin_ms,
                "rocm_ms": total_rocm_ms,
                "rocm_x": total_rocm_x,
            },
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
