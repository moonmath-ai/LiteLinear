#!/usr/bin/env python3
"""
SpecDoc0 helper: verify cast behavior and NaN safety assumptions.

Checks:
1) Plain PyTorch float8_e4m3fn cast behavior on large finite values.
2) (Optional) Installed LiteLinear fused kernel behavior under large finite inputs.
"""

from __future__ import annotations

import argparse

import torch


def check_torch_cast() -> None:
    vals = torch.tensor(
        [
            -1e6,
            -1e5,
            -1e4,
            -1e3,
            -1e2,
            -10.0,
            -1.0,
            -0.1,
            0.0,
            0.1,
            1.0,
            10.0,
            1e2,
            1e3,
            1e4,
            1e5,
            1e6,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    fp8_back = vals.to(torch.float8_e4m3fn).float()
    print("[torch-cast] finfo max:", torch.finfo(torch.float8_e4m3fn).max)
    print("[torch-cast] sample back:", fp8_back.tolist())
    print("[torch-cast] has_nan:", bool(torch.isnan(fp8_back).any().item()))
    print("[torch-cast] has_inf:", bool(torch.isinf(fp8_back).any().item()))


def check_fused_forward(m: int, k: int, n: int, r: int, scale: float) -> None:
    try:
        import lite_linear._cuda as _cuda_ext
    except Exception as exc:
        print("[fused-forward] lite_linear._cuda import failed; skipping kernel check.")
        print(f"[fused-forward] reason: {exc}")
        return

    torch.manual_seed(0)
    device = "cuda"
    x = (torch.randn(m, k, device=device, dtype=torch.float32) * scale).to(torch.bfloat16)
    q = (torch.randn(n, k, device=device, dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn)
    a = (torch.randn(n, r, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    b = (torch.randn(r, k, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    bias = (torch.randn(n, device=device, dtype=torch.float32) * 0.01).to(torch.bfloat16)

    y = _cuda_ext.fused_forward(x, q, a, b, bias, 1.0)
    y_f = y.float()
    print("[fused-forward] output shape:", tuple(y.shape))
    print("[fused-forward] finite_ratio:", float(torch.isfinite(y_f).float().mean().item()))
    print("[fused-forward] nan_count:", int(torch.isnan(y_f).sum().item()))
    print("[fused-forward] inf_count:", int(torch.isinf(y_f).sum().item()))
    print("[fused-forward] min/max:", float(y_f.min().item()), float(y_f.max().item()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Check castfinite / NaN safety assumptions.")
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument(
        "--activation-scale",
        type=float,
        default=5000.0,
        help="Scale of random finite BF16 activations for stress testing.",
    )
    args = ap.parse_args()

    check_torch_cast()
    check_fused_forward(args.m, args.k, args.n, args.r, args.activation_scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
