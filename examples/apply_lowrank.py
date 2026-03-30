#!/usr/bin/env python3
"""
Run LTX inference with FFN low-rank delta patch applied.

This recreates the "swap FFN weights and compare outputs" part of the experiment.

Usage (example):
  python ffn/apply_lowrank.py \\
    --pipeline_config configs/ltxv-13b-0.9.8-dev.yaml \\
    --prompt "..." \\
    --rank 128
"""

from __future__ import annotations

import argparse

import torch

from lite_linear.ffn_delta import (
    apply_lowrank_delta_to_transformer,
    apply_lowrank_factors_to_transformer,
    load_lowrank_factors,
    load_r_matrices,
)  # noqa: E402
from ltx_video.inference import (
    InferenceConfig,
    init_inference,
    run_inference,
)  # noqa: E402


def _meta_flag(meta: dict, key: str) -> bool | None:
    value = meta.get(key)
    if value is None:
        return None
    return str(value).lower() in ("1", "true", "yes", "y")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_config", default="configs/ltxv-13b-0.9.8-dev.yaml")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument(
        "--factors",
        default=None,
        help="Path to precomputed low-rank factors (safetensors or pt)",
    )
    ap.add_argument(
        "--r_matrices", default=None, help="Optional R-matrices file for with-R mode"
    )
    ap.add_argument("--which", default="w1,w2", help="Comma list: w1,w2")
    ap.add_argument(
        "--quantize_q",
        action="store_true",
        help="Quantize remainder Q to FP8 (if supported)",
    )
    ap.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile (recommended for patching)",
    )
    ap.add_argument("--output_path", default=None)
    ap.add_argument("--output_filename", default=None)
    args = ap.parse_args()

    # Build the pipeline without compilation so we can patch modules.
    pipeline = init_inference(config=args.pipeline_config, compile=not args.no_compile)

    which = [p.strip() for p in args.which.split(",") if p.strip()]

    # If transformer is torch.compile'd, patching is not reliable. Require no_compile for now.
    if not args.no_compile:
        raise SystemExit(
            "--no_compile is required for now (patching compiled modules is unsafe)"
        )

    if args.factors:
        factors, metadata = load_lowrank_factors(args.factors)
        print(f"[patch] Loaded factors ({len(factors)} layers), meta={metadata}")
        if _meta_flag(metadata, "with_r"):
            print("[patch] Factors were built with calibration R (baked into B).")
        else:
            print("[patch] Factors were built without calibration R.")
        apply_lowrank_factors_to_transformer(
            pipeline.transformer,
            factors=factors,
            which=which,  # type: ignore[arg-type]
            quantize_q=args.quantize_q,
        )
    else:
        r_mats = load_r_matrices(args.r_matrices) if args.r_matrices else None
        apply_lowrank_delta_to_transformer(
            pipeline.transformer,
            rank=args.rank,
            which=which,  # type: ignore[arg-type]
            r_matrices=r_mats,
            quantize_q=args.quantize_q,
        )

    # Run inference via the normal entrypoint.
    cfg = InferenceConfig(prompt=args.prompt)
    if args.output_path is not None:
        cfg.output_path = args.output_path
    if args.output_filename is not None:
        cfg.output_filename = args.output_filename
    cfg.pipeline_config = args.pipeline_config

    out = run_inference(config=cfg, pipeline=pipeline)
    if out is None:
        print("No output produced (run_inference returned None)")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
