#!/usr/bin/env python
"""
FFN Module Benchmark - Real Video Generation Sizes

All sizes captured from validate.sh (24 frames) with actual call counts.
Uses raw PyTorch ops to match what torch.compile optimizes to.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import lite_linear._cuda as _cuda_ext

    HAS_CUDA = True
except ImportError:
    raise ImportError("lite_linear._cuda not found")

try:
    import transformer_engine.pytorch as te

    _HAS_TE = True
    _TE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    te = None
    _HAS_TE = False
    _TE_IMPORT_ERROR = exc

# TE Linear is benched in bf16 mode (no `te.fp8_autocast(...)` wrap),
# matching the convention in extras/profile_fused_forward.py. The point
# is to compare LiteLinear's fused FP8 path against TE Linear's stock
# bf16 matmul; comparing to TE's FP8 mode is a deliberate follow-up
# (adds DelayedScaling recipe setup + per-call autocast overhead).
#
# TODO(pr-deletions-validate): the TE column added here was only
# smoke-tested with transformer_engine NOT installed (the n/a fallback
# path). The live-TE bench path (`te.Linear(...)` build + `.copy_(W)`
# + bench) hasn't been exercised. Verify against a TE-enabled env
# before publishing numbers. Remove this note once validated.

R = 32

# Real sizes from validate.sh 24 frames with call counts
REAL_CONFIGS = [
    # (name, M, K, N, call_count)
    ("w1", 1400, 4096, 16384, 195),
    ("w1", 5600, 4096, 16384, 84),
    ("w2", 1400, 16384, 4096, 196),
    ("w2", 5600, 16384, 4096, 84),
]


def bench(fn, warmup=20, iters=100):
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
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt-cast-inside",
        action="store_true",
        help="Include X->FP8 cast cost inside LR-PT benchmark loop (more apples-to-apples vs CUDA kernel).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write per-config + totals timings as JSON to this path "
        "(in addition to the human-readable stdout). Used as the diff target "
        "for PR-litelinear's manual perf gate.",
    )
    args = parser.parse_args()

    device = torch.device("cuda")

    print(f"FFN Benchmark (Raw Ops) R={R}")
    if args.pt_cast_inside:
        print("NOTE: LR-PT includes X->FP8 cast inside timed loop.")
    if not _HAS_TE:
        print(
            f"NOTE: transformer_engine not available ({_TE_IMPORT_ERROR}); TE column = n/a"
        )
    print("Real sizes from validate.sh (24 frames)")
    print("=" * 110)
    print(
        f"{'Config':<6} {'M':>5} {'Count':>5} {'Orig':>7} {'LR-PT':>7} {'LR-CUDA':>8} {'TE-bf16':>8} {'PT/Orig':>8} {'C/PT':>6} {'C/TE':>6}"
    )
    print("-" * 110)

    # Benchmark torch baselines once.
    base_rows = []
    for name, m, k, n, calls in REAL_CONFIGS:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        A = torch.randn(n, R, device=device, dtype=torch.bfloat16) * 0.1
        B = torch.randn(R, k, device=device, dtype=torch.bfloat16) * 0.1
        Q_fp8 = (torch.randn(n, k, device=device) * 0.1).to(torch.float8_e4m3fn)
        W = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        bias = torch.randn(n, device=device, dtype=torch.bfloat16) * 0.01
        scale = torch.tensor(1.0, device=device)

        def orig_fn():
            return F.linear(x, W, bias)

        t_orig = bench(orig_fn)

        def lr_pt_fn():
            lr = F.linear(F.linear(x, B), A)
            x_fp8 = x.to(torch.float8_e4m3fn) if args.pt_cast_inside else x_fp8_cached
            rem = torch._scaled_mm(
                x_fp8, Q_fp8.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16
            )
            return lr + rem + bias

        x_fp8_cached = x.to(torch.float8_e4m3fn)

        t_pt = bench(lr_pt_fn)

        t_te = None
        if _HAS_TE:
            try:
                # Older TE versions reject the device=/params_dtype= kwargs.
                try:
                    te_mod = te.Linear(
                        k, n, bias=True, device=device, params_dtype=torch.bfloat16
                    )
                except TypeError:
                    te_mod = te.Linear(k, n, bias=True).to(
                        device=device, dtype=torch.bfloat16
                    )
                with torch.no_grad():
                    te_mod.weight.copy_(W)
                    if te_mod.bias is not None:
                        te_mod.bias.copy_(bias)
                te_mod.eval()

                def te_fn():
                    return te_mod(x)

                t_te = bench(te_fn)
            except Exception as exc:
                print(f"[bench] TE init/bench failed for {name} M={m}: {exc}")

        base_rows.append(
            (name, m, k, n, calls, x, A, B, Q_fp8, bias, t_orig, t_pt, t_te)
        )

    total_orig = total_pt = total_cuda = 0.0
    total_te = 0.0
    total_te_calls = 0  # call-weighted denominator; TE may be missing for some shapes
    config_rows: list[dict] = []
    for name, m, k, n, calls, x, A, B, Q_fp8, bias, t_orig, t_pt, t_te in base_rows:
        if HAS_CUDA:

            def lr_cuda_fn():
                return _cuda_ext.fused_forward(x, Q_fp8, A, B, bias, 1.0)

            t_cuda = bench(lr_cuda_fn)
        else:
            t_cuda = float("inf")

        total_orig += t_orig * calls
        total_pt += t_pt * calls
        total_cuda += t_cuda * calls
        if t_te is not None:
            total_te += t_te * calls
            total_te_calls += calls

        te_us_str = f"{t_te * 1000:>6.0f}us" if t_te is not None else f"{'n/a':>8}"
        cuda_over_te_str = (
            f"{t_cuda / t_te:>5.2f}x" if t_te is not None else f"{'n/a':>6}"
        )

        config_rows.append(
            {
                "name": name, "M": m, "K": k, "N": n, "call_count": calls,
                "t_orig_ms": t_orig, "t_pt_ms": t_pt, "t_cuda_ms": t_cuda,
                "t_te_ms": t_te,
            },
        )  # fmt: skip

        print(
            f"{name:<6} {m:>5} {calls:>5} {t_orig * 1000:>5.0f}us {t_pt * 1000:>5.0f}us "
            f"{t_cuda * 1000:>6.0f}us {te_us_str} "
            f"{t_pt / t_orig:>7.2f}x {t_cuda / t_pt:>5.2f}x {cuda_over_te_str}"
        )

    print("-" * 110)
    te_total_str = f"{total_te:>6.0f}ms" if total_te_calls > 0 else f"{'n/a':>8}"
    cuda_over_te_total = (
        f"{total_cuda / total_te:>5.2f}x" if total_te_calls > 0 else f"{'n/a':>6}"
    )
    print(
        f"{'TOTAL':<6} {'':<5} {559:>5} {total_orig:>5.0f}ms {total_pt:>5.0f}ms "
        f"{total_cuda:>6.0f}ms {te_total_str} "
        f"{total_pt / total_orig:>7.2f}x {total_cuda / total_pt:>5.2f}x {cuda_over_te_total}"
    )
    print("=" * 110)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "rank": R,
                    "pt_cast_inside": args.pt_cast_inside,
                    "configs": config_rows,
                    "totals_ms": {
                        "t_orig": total_orig,
                        "t_pt": total_pt,
                        "t_cuda": total_cuda,
                        "t_te": total_te if total_te_calls > 0 else None,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
