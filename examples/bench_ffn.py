#!/usr/bin/env python
"""
FFN Module Benchmark - Real Video Generation Sizes

All sizes captured from validate.sh (24 frames) with actual call counts.
Uses raw PyTorch ops to match what torch.compile optimizes to.
"""
import torch
import torch.nn.functional as F
import argparse

try:
    import LiteFFN._cuda as LiteFFN_cuda

    HAS_CUDA = True
except ImportError:
    raise ImportError("LiteFFN._cuda not found")

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
    args = parser.parse_args()

    device = torch.device("cuda")

    print(f"FFN Benchmark (Raw Ops) R={R}")
    if args.pt_cast_inside:
        print("NOTE: LR-PT includes X->FP8 cast inside timed loop.")
    print("Real sizes from validate.sh (24 frames)")
    print("=" * 90)
    print(
        f"{'Config':<6} {'M':>5} {'Count':>5} {'Orig':>7} {'LR-PT':>7} {'LR-CUDA':>8} {'PT/Orig':>8} {'C/PT':>6}"
    )
    print("-" * 90)

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

        base_rows.append((name, m, k, n, calls, x, A, B, Q_fp8, bias, t_orig, t_pt))

    total_orig = total_pt = total_cuda = 0
    for name, m, k, n, calls, x, A, B, Q_fp8, bias, t_orig, t_pt in base_rows:
        if HAS_CUDA:

            def lr_cuda_fn():
                return LiteFFN_cuda.fused_forward(x, Q_fp8, A, B, bias, 1.0)

            t_cuda = bench(lr_cuda_fn)
        else:
            t_cuda = float("inf")

        total_orig += t_orig * calls
        total_pt += t_pt * calls
        total_cuda += t_cuda * calls

        print(
            f"{name:<6} {m:>5} {calls:>5} {t_orig*1000:>5.0f}us {t_pt*1000:>5.0f}us {t_cuda*1000:>6.0f}us {t_pt/t_orig:>7.2f}x {t_cuda/t_pt:>5.2f}x"
        )

    print("-" * 90)
    print(
        f"{'TOTAL':<6} {'':<5} {559:>5} {total_orig:>5.0f}ms {total_pt:>5.0f}ms {total_cuda:>6.0f}ms {total_pt/total_orig:>7.2f}x {total_cuda/total_pt:>5.2f}x"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
