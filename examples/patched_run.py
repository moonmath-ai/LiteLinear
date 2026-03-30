#!/usr/bin/env python3
"""
Run LTX inference with FFN low-rank delta patch applied, using the same CLI surface
as `run_ltx_wrapper.py` (including legacy conditioning flags).

This is meant to make "baseline vs no-R vs with-R" comparisons apples-to-apples
for both T2V and I2V workflows.
"""

from __future__ import annotations

import argparse

from transformers import HfArgumentParser

import imageio  # noqa: E402

from lite_linear.ffn_delta import (
    apply_lowrank_delta_to_transformer,
    apply_lowrank_factors_to_transformer,
    load_lowrank_factors,
    load_r_matrices,
)  # noqa: E402
from ltx_video.inference import (  # noqa: E402
    InferenceConfig,
    enhance_video,
    init_inference,
    run_inference,
)
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItemRaw  # noqa: E402


def _get_transformer_from_pipeline(pipeline):
    # Supports both LTXVideoPipeline (has .transformer) and LTXMultiScalePipeline (has .video_pipeline.transformer).
    if hasattr(pipeline, "transformer"):
        return pipeline.transformer
    if hasattr(pipeline, "video_pipeline") and hasattr(
        pipeline.video_pipeline, "transformer"
    ):
        return pipeline.video_pipeline.transformer
    raise AttributeError(
        f"Cannot locate transformer on pipeline type {type(pipeline).__name__}. "
        "Expected .transformer or .video_pipeline.transformer"
    )


def _meta_flag(meta: dict, key: str) -> bool | None:
    value = meta.get(key)
    if value is None:
        return None
    return str(value).lower() in ("1", "true", "yes", "y")


def main() -> None:
    # 1) Parse patch args first, then pass remaining args to the existing wrapper-style parsing.
    patch_parser = argparse.ArgumentParser(add_help=True)
    patch_parser.add_argument("--rank", type=int, default=64)
    patch_parser.add_argument(
        "--factors",
        default=None,
        help="Path to precomputed low-rank factors (safetensors or pt)",
    )
    patch_parser.add_argument("--which", default="w1,w2", help="Comma list: w1,w2")
    patch_parser.add_argument(
        "--r_matrices", default=None, help="Path to saved R matrices for with-R mode"
    )
    patch_parser.add_argument(
        "--quantize_q",
        action="store_true",
        help="Quantize remainder Q to FP8 (if supported)",
    )
    patch_parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile (required for patching right now)",
    )

    patch_args, remaining_args = patch_parser.parse_known_args()

    # 2) Parse legacy conditioning args (same behavior as run_ltx_wrapper.py)
    legacy_parser = argparse.ArgumentParser(add_help=False)
    legacy_parser.add_argument("--conditioning_media_paths", nargs="+", default=None)
    legacy_parser.add_argument(
        "--conditioning_start_frames", nargs="+", type=int, default=None
    )
    legacy_parser.add_argument(
        "--conditioning_strengths", nargs="+", type=float, default=None
    )

    legacy_args, remaining_args = legacy_parser.parse_known_args(remaining_args)

    conditioning_images = None
    if legacy_args.conditioning_media_paths is not None:
        if legacy_args.conditioning_start_frames is None:
            raise ValueError(
                "conditioning_start_frames must be provided when using conditioning_media_paths"
            )
        if len(legacy_args.conditioning_media_paths) != len(
            legacy_args.conditioning_start_frames
        ):
            raise ValueError(
                f"conditioning_media_paths ({len(legacy_args.conditioning_media_paths)} items) and "
                f"conditioning_start_frames ({len(legacy_args.conditioning_start_frames)} items) must have the same length"
            )

        strengths = legacy_args.conditioning_strengths
        if strengths is None:
            strengths = [1.0] * len(legacy_args.conditioning_media_paths)
        elif len(strengths) != len(legacy_args.conditioning_media_paths):
            raise ValueError(
                f"conditioning_strengths ({len(strengths)} items) must have the same length as "
                f"conditioning_media_paths ({len(legacy_args.conditioning_media_paths)} items)"
            )

        conditioning_images = [
            ConditioningItemRaw(
                media_item=media_path,
                start_frame_number=start_frame,
                strength=strength,
            )
            for media_path, start_frame, strength in zip(
                legacy_args.conditioning_media_paths,
                legacy_args.conditioning_start_frames,
                strengths,
            )
        ]

    # 3) Parse the rest into InferenceConfig (same as run_ltx_wrapper.py)
    parser = HfArgumentParser(InferenceConfig)
    config = parser.parse_args_into_dataclasses(args=remaining_args)[0]
    if conditioning_images is not None:
        config.conditioning_images = conditioning_images

    # 4) Build pipeline (must be uncompiled for safe patching)
    if not patch_args.no_compile:
        raise SystemExit(
            "--no_compile is required for now (patching compiled modules is unsafe)"
        )

    pipeline = init_inference(config=config.pipeline_config, compile=False)

    which = [p.strip() for p in patch_args.which.split(",") if p.strip()]

    transformer = _get_transformer_from_pipeline(pipeline)
    if patch_args.factors:
        factors, metadata = load_lowrank_factors(patch_args.factors)
        print(f"[patch] Loaded factors ({len(factors)} layers), meta={metadata}")
        if _meta_flag(metadata, "with_r"):
            print("[patch] Factors were built with calibration R (baked into B).")
        else:
            print("[patch] Factors were built without calibration R.")
        apply_lowrank_factors_to_transformer(
            transformer,
            factors=factors,
            which=which,  # type: ignore[arg-type]
            quantize_q=patch_args.quantize_q,
        )
    else:
        r_mats = load_r_matrices(patch_args.r_matrices) if patch_args.r_matrices else None
        apply_lowrank_delta_to_transformer(
            transformer,
            rank=patch_args.rank,
            which=which,  # type: ignore[arg-type]
            r_matrices=r_mats,
            quantize_q=patch_args.quantize_q,
        )

    video_np = run_inference(config=config, pipeline=pipeline)
    if video_np is None:
        print("No output produced (run_inference returned None)")
        return

    # Match `run_ltx_wrapper.py`: apply enhancement and write mp4 to output_path/output_filename.
    enhanced = enhance_video(video_np)
    out_path = Path(config.output_path) / config.output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, enhanced, fps=config.frame_rate, format="mp4")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
