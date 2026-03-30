#!/usr/bin/env python3
"""
Build low-rank+remainder decompositions for LTX FFN layers and save them.

This is the "offline" half of the FFN delta experiment:
- Load Transformer weights from an LTX `.safetensors` checkpoint
- Optionally load per-layer R matrices (autocorrelation) collected via hooks
- Produce A/B/Q factors and write them to disk for later patching
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from lite_linear.ffn_delta import (
    decompose_weight,
    iter_ltx_ffn_linears,
    load_r_matrices,
    quantize_fp8_per_tensor,
    save_lowrank_factors,
)  # noqa: E402
from ltx_video.models.transformers.transformer3d import Transformer3DModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to LTX .safetensors checkpoint")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument(
        "--r_matrices",
        default=None,
        help="Optional path to torch-saved dict name->R matrix (from capture script)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    ap.add_argument(
        "--which",
        default="w1,w2",
        help="Comma list of FFN parts to decompose: w1,w2",
    )
    ap.add_argument(
        "--format",
        choices=["safetensors", "pt"],
        default="safetensors",
        help="Serialization format for factors",
    )
    ap.add_argument(
        "--save_dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for stored factors (smaller is faster to load)",
    )
    ap.add_argument(
        "--no_quant",
        action="store_true",
        help="Store Q in BF16/FP16/FP32 instead of FP8",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    save_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.save_dtype]

    r_mats: Dict[str, torch.Tensor] | None = None
    if args.r_matrices:
        r_mats = {}
        r_paths = [p.strip() for p in args.r_matrices.split(",") if p.strip()]
        for r_path_str in r_paths:
            p = Path(r_path_str)
            if p.exists():
                print(f"[decompose] Loading R matrices from {p}")
                loaded = load_r_matrices(p)
                print(f"[decompose] Loaded {len(loaded)} matrices from {p}")
                r_mats.update(loaded)
            else:
                print(f"[decompose] WARNING: R path {p} not found. Skipping.")
        if not r_mats:
            r_mats = None

    transformer = Transformer3DModel.from_pretrained(Path(args.ckpt))
    transformer = transformer.to(device=args.device, dtype=dtype)
    transformer.eval()

    which = set([p.strip() for p in args.which.split(",") if p.strip()])

    factors: Dict[str, Dict[str, torch.Tensor]] = {}
    for ref, mod in iter_ltx_ffn_linears(transformer):
        if ref.part not in which:
            continue

        W = mod.weight.detach()
        b = mod.bias.detach() if getattr(mod, "bias", None) is not None else None

        r = r_mats.get(ref.name) if r_mats is not None else None
        if r is not None:
            r = r.to(device=args.device, dtype=torch.float32)
            # Use r_sqrt/inv inside decompose_weight via (sqrt,inv) computation there would be cleaner,
            # but we keep logic in one place (apply_lowrank_delta_to_transformer does sqrt+inv).
            # Here we compute sqrt+inv so we can store "with-R" decompositions.
            from lite_linear.ffn_delta import r_sqrt_and_inv

            r_sqrt, r_sqrt_inv = r_sqrt_and_inv(r)
        else:
            r_sqrt, r_sqrt_inv = None, None

        A, B, Q = decompose_weight(
            W, rank=args.rank, r_sqrt=r_sqrt, r_sqrt_inv=r_sqrt_inv
        )

        entry: Dict[str, torch.Tensor] = {
            "A": A.to(device="cpu", dtype=save_dtype),
            "B": B.to(device="cpu", dtype=save_dtype),
        }
        if args.no_quant:
            entry["Q"] = Q.to(device="cpu", dtype=save_dtype)
        else:
            q = quantize_fp8_per_tensor(Q.to(dtype=torch.float32), dtype=torch.float8_e4m3fn)
            entry["Q_fp8"] = q.data.detach().cpu().contiguous()
            entry["Q_scale_inv"] = q.scale.reciprocal().detach().cpu().contiguous()
        if b is not None:
            entry["bias"] = b.to(device="cpu", dtype=save_dtype)
        factors[ref.name] = entry

        print(
            f"[decompose] {ref.name} W={tuple(W.shape)} rank={args.rank} with_R={r is not None}"
        )

    ext = "safetensors" if args.format == "safetensors" else "pt"
    out_path = out_dir / f"ffn_lowrank_factors_r{args.rank}.{ext}"
    meta = {
        "rank": str(args.rank),
        "which": args.which,
        "with_r": str(r_mats is not None),
        "r_matrices": args.r_matrices or "",
        "save_dtype": args.save_dtype,
        "q_dtype": "float8_e4m3fn" if not args.no_quant else args.save_dtype,
    }
    save_lowrank_factors(out_path, factors, metadata=meta)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
