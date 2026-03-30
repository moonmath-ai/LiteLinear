#!/usr/bin/env python
"""
Decompose FFN layers in LTX-Video using Low-Rank + FP8 Quantization.
Can use captured R matrices for calibration (e.g. w1) and fall back to standard SVD for others (w2).
"""
import argparse
import copy
import gc
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn.functional as F
from ltx_video.inference import init_inference, InferenceConfig, run_inference
from lite_linear.ffn_delta import (
    apply_lowrank_delta_to_transformer,
    apply_lowrank_factors_to_transformer,
    load_r_matrices,
    load_lowrank_factors,
    iter_ltx_ffn_linears,
    _get_weight_bias,
    LowRankDeltaLinear,
    r_sqrt_and_inv,
    decompose_weight,
)
from ffn.ffn_patch import apply_te_linear_to_transformer
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItemRaw

DEFAULT_PROMPT = "Two anthropomorphic cats boxing in a well lit arena and throwing ferocious fast moving punches at each other."
DEFAULT_NUM_FRAMES = 121
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 704


def export_to_video(
    video_np: torch.Tensor | np.ndarray, output_path: str, fps: int = 25
):
    import imageio
    import numpy as np

    if isinstance(video_np, torch.Tensor):
        video_np = video_np.cpu().numpy()

    if video_np.dtype != np.uint8:
        video_np = (np.clip(video_np, 0, 1) * 255).astype(np.uint8)

    # video_np is (F, H, W, C)
    # Use imageio with a standard codec for maximum compatibility
    # video_np is (F, H, W, C)
    # Use imageio with a standard codec for maximum compatibility
    print(f"[export] Saving video to {output_path}...")
    try:
        imageio.mimwrite(output_path, video_np, fps=fps, quality=8, codec="libx264")
    except Exception as e:
        print(f"[export] Warning: imageio failed with libx264 ({e}), trying default...")
        imageio.mimwrite(output_path, video_np, fps=fps)

    print(f"[export] Video saved to {output_path}")


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


def _set_transformer_on_pipeline(pipeline, transformer) -> None:
    # Supports both LTXVideoPipeline (has .transformer) and LTXMultiScalePipeline (has .video_pipeline.transformer).
    if hasattr(pipeline, "transformer"):
        pipeline.transformer = transformer
        return
    if hasattr(pipeline, "video_pipeline") and hasattr(
        pipeline.video_pipeline, "transformer"
    ):
        pipeline.video_pipeline.transformer = transformer
        return
    raise AttributeError(
        f"Cannot set transformer on pipeline type {type(pipeline).__name__}. "
        "Expected .transformer or .video_pipeline.transformer"
    )


def _meta_flag(meta: dict, key: str) -> bool | None:
    value = meta.get(key)
    if value is None:
        return None
    return str(value).lower() in ("1", "true", "yes", "y")


def get_vram_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6


def benchmark_latency(
    pipeline,
    prompt=DEFAULT_PROMPT,
    runs=3,
    num_frames=DEFAULT_NUM_FRAMES,
    seed: int = 171198,
    pipeline_config=None,
    conditioning_image=None,
):
    """Measure latency of a short inference run."""
    # Avoid per-run cache eviction; it breaks/defeats CUDA graph capture and distorts benchmark numbers.
    prev_skip = os.environ.get("SKIP_EMPTY_CACHE")
    os.environ["SKIP_EMPTY_CACHE"] = "1"
    print(f"[benchmark] Warmup...")

    conditioning_images = None
    if conditioning_image:
        conditioning_images = [
            ConditioningItemRaw(
                media_item=conditioning_image,
                start_frame_number=0,
                strength=1.0,
            )
        ]

    cfg = InferenceConfig(prompt=prompt, conditioning_images=conditioning_images, seed=seed)
    if pipeline_config:
        cfg.pipeline_config = pipeline_config
    cfg.num_frames = num_frames

    try:
        # Warmup
        _ = run_inference(config=cfg, pipeline=pipeline)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
        print(f"[benchmark] Running {runs} iterations...")
        start = time.perf_counter()
        for _ in range(runs):
            _ = run_inference(config=cfg, pipeline=pipeline)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - start) / runs
    finally:
        if prev_skip is None:
            os.environ.pop("SKIP_EMPTY_CACHE", None)
        else:
            os.environ["SKIP_EMPTY_CACHE"] = prev_skip


class ProfiledLinear(torch.nn.Module):
    def __init__(self, original_linear, name):
        super().__init__()
        self.original = original_linear
        self.name = name

    def forward(self, x):
        with torch.profiler.record_function(self.name):
            with torch.cuda.nvtx.range(self.name):
                return self.original(x)


def wrap_baseline_for_profiling(transformer):
    """
    Wraps all FFN linear layers in the baseline model with markers
    so they appear clearly in traces (grouping).
    """
    print("[decompose] Wrapping Baseline FFN layers with Profiling Markers...")
    count = 0
    for ref, mod in iter_ltx_ffn_linears(transformer):
        # Create descriptive name: FFN_Block{i}_{part}
        # e.g., FFN_Block0_w1
        name = f"Baseline_FFN_Block{ref.block_idx}_{ref.part}"
        wrapper = ProfiledLinear(mod, name)

        # Replace in-place
        block = transformer.transformer_blocks[ref.block_idx]
        ff = block.ff
        if ref.part == "w2":
            ff.net[2] = wrapper
        else:  # w1
            act = ff.net[0]
            if hasattr(act, "proj"):
                act.proj = wrapper
            else:
                # Fallback: if we found it via scan, we might need manual surgery
                # But iter_ltx_ffn_linears guarantees finding it.
                # If standard structure:
                pass
        count += 1
    print(f"[decompose] Wrapped {count} baseline layers.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pipeline_config", default="configs/ltxv-13b-0.9.8-distilled.yaml"
    )
    ap.add_argument(
        "--r_path",
        default=None,
        help="Path to .pt file with R matrices (e.g. from capture_r.py)",
    )
    ap.add_argument(
        "--factors",
        default=None,
        help="Path to precomputed low-rank factors (safetensors or pt)",
    )
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument(
        "--blocks",
        default=None,
        help="Comma list/range of blocks to decompose (default: all blocks)",
    )
    ap.add_argument(
        "--no_quant",
        action="store_true",
        help="Disable FP8 quantization for Q (use BF16/FP16)",
    )
    ap.add_argument(
        "--pure_lowrank",
        action="store_true",
        help="Omit Q remainder (maximum speed, lower quality)",
    )
    ap.add_argument(
        "--benchmark", action="store_true", help="Run inference latency benchmark"
    )
    ap.add_argument(
        "--te_linear",
        action="store_true",
        help="Replace FFN linears with TransformerEngine Linear (BF16) for optional testing",
    )
    ap.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline benchmark to save memory/time",
    )
    ap.add_argument(
        "--baseline_latency_s",
        type=float,
        default=None,
        help="Use a provided baseline latency (seconds/run). Intended for rank_sweep when --skip_baseline is used.",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the benchmarked model",
    )
    ap.add_argument(
        "--compile_backend",
        type=str,
        default="inductor",
        help="torch.compile backend (default: inductor). Options: inductor, cudagraphs, aot_eager, eager",
    )
    ap.add_argument(
        "--inductor_cudagraphs",
        action="store_true",
        help="Enable Inductor CUDA graphs (triton.cudagraphs). Disabled by default for stability.",
    )
    ap.add_argument(
        "--inductor_cudagraph_trees",
        action="store_true",
        help="Enable Inductor cudagraph trees (triton.cudagraph_trees). Requires --inductor_cudagraphs.",
    )
    ap.add_argument(
        "--output_video", default=None, help="Path to save verification video"
    )
    ap.add_argument(
        "--force_output",
        action="store_true",
        help="Overwrite output_video if it already exists",
    )
    ap.add_argument(
        "--benchmark_runs",
        type=int,
        default=3,
        help="Number of iterations for benchmark",
    )
    ap.add_argument(
        "--num_frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of frames for benchmark/verification",
    )
    ap.add_argument(
        "--frame_rate",
        type=int,
        default=25,
        help="Frame rate for generation and video export",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=171198,
        help="Random seed for inference (used for benchmarks and exported videos)",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt for benchmark/verification",
    )
    ap.add_argument(
        "--conditioning_image",
        type=str,
        default=None,
        help="Path to reference image (I2V)",
    )
    ap.add_argument(
        "--static_scale",
        type=float,
        default=None,
        help="Static scale for input quantization (e.g. 1.0 = standard, 0.1 = high amp)",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Run a calibration pass to set per-layer static scales",
    )
    ap.add_argument(
        "--export_baseline",
        action="store_true",
        help="Export baseline video for comparison",
    )
    ap.add_argument(
        "--skip_decomposition",
        action="store_true",
        help="Skip decomposition/patching (useful for baseline-only measurement)",
    )
    ap.add_argument(
        "--use_lite_linear",
        action="store_true",
        help="Use LiteLinear FFNs (requires USE_LITE_LINEAR=1 in env); skip LowRankDelta patching",
    )
    ap.add_argument(
        "--compile_mode",
        type=str,
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode: default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs",
    )
    ap.add_argument(
        "--ablation_no_q",
        action="store_true",
        help="DIAGNOSTIC: Skip Q component (Run only LowRank)",
    )
    ap.add_argument(
        "--ablation_only_q",
        action="store_true",
        help="DIAGNOSTIC: Skip LowRank component (Run only Q)",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Run a single inference step with torch.profiler and export trace",
    )
    ap.add_argument(
        "--profile_baseline",
        action="store_true",
        help="Profile the BASELINE model instead of the decomposed one",
    )
    ap.add_argument(
        "--no_kernel",
        action="store_true",
        help="Disable Custom CUDA Kernels (Fallback to Torch FP8)",
    )
    ap.add_argument(
        "--capture_activations",
        action="store_true",
        help="Capture real activations and layers to ffn_delta_outputs/captured_activations.pt",
    )
    ap.add_argument(
        "--skip_benchmark",
        action="store_true",
        help="Skip benchmark warmup/runs, only do verification video if --output_video set",
    )
    args = ap.parse_args()

    if args.no_kernel:
        print("[decompose] Disabling Custom CUDA Kernels (Fallback to Torch FP8)...")
        import lite_linear.ffn_delta

        lite_linear.ffn_delta.HAS_CUDA_EXT = False

    def _compile_model(m: torch.nn.Module) -> torch.nn.Module:
        """
        torch.compile wrapper.
        Note: non-inductor backends (e.g. cudagraphs) do not accept the `mode=` kwarg.
        """
        # Make torch.compile more robust for this workload.
        # LowRankDeltaLinear has many module instances; we want to avoid hitting the default
        # recompilation limit due to per-module guards.
        try:
            dynamo_config = torch._dynamo.config  # type: ignore[attr-defined]
            if hasattr(dynamo_config, "recompile_limit"):
                dynamo_config.recompile_limit = 200  # type: ignore[assignment]
            if hasattr(dynamo_config, "capture_scalar_outputs"):
                dynamo_config.capture_scalar_outputs = True  # type: ignore[assignment]
        except Exception:
            pass

        if args.compile_backend == "inductor":
            # torch.compile forbids passing both (mode=...) and (options=...).
            # Build an explicit options dict from the mode preset, then apply overrides.
            from torch._inductor import list_mode_options

            options = dict(list_mode_options().get(args.compile_mode, {}))

            # Freeze weights into the graph for cudagraph friendliness and lower overhead.
            # Safe for inference (weights are not updated).
            options["freezing"] = True

            # Optional overrides (default OFF for stability).
            if args.inductor_cudagraphs:
                options["triton.cudagraphs"] = True
            if args.inductor_cudagraph_trees:
                options["triton.cudagraph_trees"] = True
                # Trees require cudagraphs to be enabled.
                options["triton.cudagraphs"] = True
            else:
                # Force-disable trees unless explicitly requested.
                options["triton.cudagraph_trees"] = False

            return torch.compile(m, backend=args.compile_backend, options=options)
        return torch.compile(m, backend=args.compile_backend)

    # Set up global compile mode
    global COMPILE_MODE
    COMPILE_MODE = args.compile_mode

    # 1. Load Model
    print(f"[decompose] Loading pipeline from {args.pipeline_config}")
    vram_start = get_vram_mb()
    pipeline = init_inference(
        config=args.pipeline_config, compile=False
    )  # Patching happens on uncompiled model
    transformer = _get_transformer_from_pipeline(pipeline)
    if args.te_linear:
        replaced = apply_te_linear_to_transformer(
            transformer, which=("w1", "w2"), dtype=torch.bfloat16, strict=True
        )
        print(f"[decompose] Patched {replaced} FFN layers with TE Linear (bf16).")
    vram_model = get_vram_mb()
    print(
        f"[decompose] Model loaded. VRAM: {vram_model:.0f} MB (Delta: {vram_model - vram_start:.0f} MB)"
    )

    # 2. Determine total number of blocks
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("Expected transformer to have .transformer_blocks")
    total_blocks = len(blocks)
    print(f"[decompose] Model has {total_blocks} transformer blocks (0-{total_blocks-1})")

    # 3. Load R Matrices (only needed for on-the-fly decomposition)
    r_matrices = {}
    if args.r_path and args.factors:
        print("[decompose] --factors set; skipping R matrices load.")
    elif args.r_path:
        r_paths = args.r_path.split(",")
        for r_path_str in r_paths:
            p = Path(r_path_str.strip())
            if p.exists():
                print(f"[decompose] Loading R matrices from {p}")
                loaded = load_r_matrices(p)
                print(f"[decompose] Loaded {len(loaded)} matrices from {p}")
                r_matrices.update(loaded)
            else:
                print(f"[decompose] WARNING: R path {p} not found. Skipping.")
        print(f"[decompose] Total R matrices loaded: {len(r_matrices)}")

    # 4. Parse blocks
    def parse_blocks(s):
        if not s or s.lower() == "empty" or s.lower() == "none":
            return set()
        ranges = s.split(",")
        idxs = []
        for r in ranges:
            if "-" in r:
                start, end = map(int, r.split("-"))
                idxs.extend(range(start, end + 1))
            else:
                idxs.append(int(r))
        return set(idxs)

    # If --blocks not specified, default to all blocks
    if args.blocks is None:
        args.blocks = f"0-{total_blocks-1}"
        print(f"[decompose] No --blocks specified, defaulting to all blocks: {args.blocks}")

    target_blocks = parse_blocks(args.blocks)
    if args.factors:
        print(
            f"[decompose] Targeting blocks: {sorted(list(target_blocks))} with Factors={args.factors}"
        )
    else:
        print(
            f"[decompose] Targeting blocks: {sorted(list(target_blocks))} with Rank={args.rank}"
        )

    # 4. Measure Baseline (Optional)
    lat_base = float(args.baseline_latency_s) if args.baseline_latency_s is not None else 0.0
    if args.benchmark and not args.skip_baseline:
        if args.compile:
            print("\n[bench] Compiling Baseline...")
            # IMPORTANT: Must assign back to pipeline to actually use it!
            t = _get_transformer_from_pipeline(pipeline)

            # Detection: If weights are FP8, nn.Linear (eager/fallback) will crash during tracing/warmup.
            # We cast to BF16 for baseline measurement to ensure it runs.
            is_fp8 = any(
                p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                for p in t.parameters()
            )
            if is_fp8:
                print(
                    "[bench] Detected FP8 weights. Casting to BF16 for Baseline latency measurement."
                )
                t.to(torch.bfloat16)

            t_compiled = _compile_model(t)

            _set_transformer_on_pipeline(pipeline, t_compiled)

            print("\n[bench] Measuring Baseline Latency (Compiled)...")
            lat_base = benchmark_latency(
                pipeline,
                prompt=args.prompt,
                num_frames=args.num_frames,
                seed=args.seed,
                runs=args.benchmark_runs,
                pipeline_config=args.pipeline_config,
                conditioning_image=args.conditioning_image,
            )
            print(f"[bench] Baseline: {lat_base:.3f} s/run")

            if args.output_video and args.export_baseline:
                print("[bench] Exporting baseline video for comparison...")
                baseline_out = args.output_video.replace(".mp4", "_baseline.mp4")
                conditioning_images = None
                if args.conditioning_image:
                    conditioning_images = [
                        ConditioningItemRaw(
                            media_item=args.conditioning_image,
                            start_frame_number=0,
                            strength=1.0,
                        )
                    ]

                cfg = InferenceConfig(
                    prompt=args.prompt,
                    num_frames=args.num_frames,
                    frame_rate=args.frame_rate,
                    seed=args.seed,
                    conditioning_images=conditioning_images,
                )
                cfg.pipeline_config = args.pipeline_config
                video_np = run_inference(config=cfg, pipeline=pipeline)
                if video_np is not None:
                    export_to_video(video_np, baseline_out, fps=args.frame_rate)

            # Baseline-only mode: do not attempt to reload/patch the model afterwards.
            # This is intended for external scripts (e.g. rank_sweep) that only need a baseline latency.
            if args.skip_decomposition:
                if args.output_video and not args.export_baseline:
                    if os.path.exists(args.output_video) and not args.force_output:
                        print(
                            f"[bench] Output exists, skipping export: {args.output_video}"
                        )
                    else:
                        print("[bench] Exporting baseline video...")
                        conditioning_images = None
                        if args.conditioning_image:
                            conditioning_images = [
                                ConditioningItemRaw(
                                    media_item=args.conditioning_image,
                                    start_frame_number=0,
                                    strength=1.0,
                                )
                            ]

                        cfg = InferenceConfig(
                            prompt=args.prompt,
                            num_frames=args.num_frames,
                            frame_rate=args.frame_rate,
                            seed=args.seed,
                            conditioning_images=conditioning_images,
                        )
                        cfg.pipeline_config = args.pipeline_config
                        video_np = run_inference(config=cfg, pipeline=pipeline)
                        if video_np is not None:
                            export_to_video(video_np, args.output_video, fps=args.frame_rate)
                print("[bench] --skip_decomposition set: exiting after baseline measurement.")
                return

            print(
                "[bench] Reloading model for patching (clearing compiled baseline)..."
            )
            # Aggressive cleanup
            del pipeline
            if "t" in locals():
                del t
            if "t_compiled" in locals():
                del t_compiled

            if "t_compiled" in locals():
                del t_compiled

            # gc is already imported globally
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
            torch.cuda.synchronize()

            pipeline = init_inference(config=args.pipeline_config, compile=False)
            transformer = _get_transformer_from_pipeline(pipeline)

            # Unblock non-FFN layers (like Attention to_q) by casting to BF16.
            # Our FFN patches will then re-introduce FP8 compute specifically for FFN remainders.
            print(
                "[decompose] Casting non-patched layers to BF16 for execution stability."
            )
            transformer.to(torch.bfloat16)

        else:
            print("\n[bench] Measuring Baseline Latency...")
            lat_base = benchmark_latency(
                pipeline,
                prompt=args.prompt,
                num_frames=args.num_frames,
                seed=args.seed,
                runs=args.benchmark_runs,
                pipeline_config=args.pipeline_config,
                conditioning_image=args.conditioning_image,
            )
            print(f"[bench] Baseline: {lat_base:.3f} s/run")

            if args.output_video and args.export_baseline:
                print("[bench] Exporting baseline video for comparison...")
                baseline_out = args.output_video.replace(".mp4", "_baseline.mp4")
                conditioning_images = None
                if args.conditioning_image:
                    conditioning_images = [
                        ConditioningItemRaw(
                            media_item=args.conditioning_image,
                            start_frame_number=0,
                            strength=1.0,
                        )
                    ]

                cfg = InferenceConfig(
                    prompt=args.prompt,
                    num_frames=args.num_frames,
                    frame_rate=args.frame_rate,
                    seed=args.seed,
                    conditioning_images=conditioning_images,
                )
                cfg.pipeline_config = args.pipeline_config
                video_np = run_inference(config=cfg, pipeline=pipeline)
                if video_np is not None:
                    export_to_video(video_np, baseline_out, fps=args.frame_rate)

            if args.skip_decomposition:
                print("[bench] --skip_decomposition set: exiting after baseline measurement.")
                return
    elif args.skip_baseline:
        if args.baseline_latency_s is not None:
            print(f"[bench] Skipping baseline, using provided value: {lat_base:.3f}s")
        else:
            print(f"[bench] Skipping baseline, using cached value: {lat_base:.3f}s")

    # Ensure model is in BF16 for stable Eager execution (Attention layers, etc.)
    # Our FFN patches will re-introduce FP8 where needed.
    print("[decompose] Casting remaining layers to BF16 for execution stability.")
    transformer.to(torch.bfloat16)

    # 5. Apply Decomposition
    count = 0
    vram_before_decomp = get_vram_mb()

    if args.use_lite_linear:
        if os.environ.get("USE_LITE_LINEAR", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            print(
                "[decompose] WARNING: --use_lite_linear set but USE_LITE_LINEAR env is not enabled. "
                "Set USE_LITE_LINEAR=1 before running to ensure FFNs are LiteLinear."
            )
        print("[decompose] --use_lite_linear set: skipping LowRankDelta patching.")
        # Best-effort count of LiteLinear modules for reporting.
        try:
            from lite_linear import LiteLinear

            for _, mod in iter_ltx_ffn_linears(transformer):
                if isinstance(mod, LiteLinear):
                    count += 1
        except Exception:
            pass
    elif not args.profile_baseline and not args.skip_decomposition:
        print("\n[decompose] Applying Low-Rank Decomposition...")
        if args.factors:
            factors, metadata = load_lowrank_factors(args.factors)
            print(f"[decompose] Loaded factors ({len(factors)} layers), meta={metadata}")
            if _meta_flag(metadata, "with_r"):
                print("[decompose] Factors were built with calibration R (baked into B).")
            else:
                print("[decompose] Factors were built without calibration R.")
            count = apply_lowrank_factors_to_transformer(
                transformer,
                factors=factors,
                which=("w1", "w2"),
                quantize_q=(not args.no_quant),
                static_input_scale=args.static_scale,
                ablation_no_q=args.ablation_no_q,
                ablation_only_q=args.ablation_only_q,
                pure_lowrank=args.pure_lowrank,
                block_filter=target_blocks,
                capture_inputs=args.capture_activations,
                strict=False,
            )
        else:
            for ref, mod in iter_ltx_ffn_linears(transformer):
                if ref.block_idx not in target_blocks:
                    continue

                W, b = _get_weight_bias(mod)
                device = W.device
                dtype = W.dtype

                # Determine R
                r = r_matrices.get(ref.name)
                if r is not None:
                    r = r.to(device=device, dtype=torch.float32)
                    r_sqrt, r_sqrt_inv = r_sqrt_and_inv(r)
                else:
                    r_sqrt, r_sqrt_inv = None, None

                if count == 0:
                    print(f"[decompose] Layer info: W={W.shape}, dtype={W.dtype}")

                # Decompose
                A, B, Q = decompose_weight(
                    W, rank=args.rank, r_sqrt=r_sqrt, r_sqrt_inv=r_sqrt_inv
                )

                # Cleanup R
                if r is not None:
                    del r, r_sqrt, r_sqrt_inv
                    r_matrices[ref.name] = None

                # Cast back
                A = A.to(device=device, dtype=dtype)
                B = B.to(device=device, dtype=dtype)
                Q = Q.to(device=device, dtype=dtype)
                bias = b.to(device=device, dtype=dtype) if b is not None else None

                if args.pure_lowrank:
                    Q = None

                if count == 0:
                    q_shape = Q.shape if Q is not None else "None (Pure LR)"
                    print(f"[decompose] Factors: A={A.shape}, B={B.shape}, Q={q_shape}")

                # Replace
                repl = LowRankDeltaLinear(
                    A=A,
                    B=B,
                    Q=Q,
                    bias=bias,
                    quantize_q=(not args.no_quant),
                    static_input_scale=args.static_scale,
                    ablation_no_q=args.ablation_no_q,
                    ablation_only_q=args.ablation_only_q,
                    layer_name=ref.name,
                )
                repl.to(device=device)

                if args.capture_activations:
                    repl.capture_inputs = True

                # Install replacement.
                block = transformer.transformer_blocks[ref.block_idx]
                ff = block.ff
                if ref.part == "w2":
                    ff.net[2] = repl
                else:
                    if hasattr(ff.net[0], "proj"):
                        ff.net[0].proj = repl

                # Explicitly release references
                del W, b, A, B, Q, bias, repl
                if "r" in locals() and r is not None:
                    del r
                if "r_sqrt" in locals() and r_sqrt is not None:
                    del r_sqrt

                count += 1
                if count % 5 == 0:
                    print(f"  ... decomposed {count} layers")
                    gc.collect()
                    torch.cuda.empty_cache()
    else:
        print("\n[decompose] --profile_baseline set. SKIP decomposition.")

    # 5. Prepare Inference Config
    conditioning_images = None
    if args.conditioning_image:
        conditioning_images = [
            ConditioningItemRaw(
                media_item=args.conditioning_image,
                start_frame_number=0,
                strength=1.0,
            )
        ]
    cfg = InferenceConfig(
        prompt=args.prompt,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        seed=args.seed,
        conditioning_images=conditioning_images,
    )
    cfg.pipeline_config = args.pipeline_config

    # 5.1 Auto-Calibration (Optional)
    if args.calibrate and not args.profile_baseline:
        print("\n[calibrate] Running calibration pass (Eager Mode)...")

        # Enable calibration mode
        for m in transformer.modules():
            if isinstance(m, LowRankDeltaLinear):
                m.calibration_mode = True

        cfg_calib = copy.copy(cfg)
        print(
            f"[calibrate] Pass: Resolution={cfg_calib.height}x{cfg_calib.width}, Frames={cfg_calib.num_frames}"
        )
        _ = run_inference(config=cfg_calib, pipeline=pipeline)

        # Apply scales and disable calibration
        fp8_max = 448.0  # For e4m3fn
        for m in transformer.modules():
            if isinstance(m, LowRankDeltaLinear):
                if m.use_fp8_q:
                    amax = m.max_amax.item()
                    safe_scale = fp8_max / max(amax, 1e-4)
                    # Cap scale to avoid extreme values. 12.0 is verified safe (48 frames) - Max Working Scale.
                    safe_scale = min(safe_scale, 12.0)
                    # Create tensor on same device as the model (CUDA)
                    device = m.A.device
                    scale_tensor = torch.tensor(
                        safe_scale, dtype=torch.float32, device=device
                    )
                    # Use setattr if buffer exists, register_buffer if new
                    if hasattr(m, "static_scale"):
                        m.static_scale = scale_tensor
                    else:
                        m.register_buffer("static_scale", scale_tensor)
                    print(
                        f"[calibrate] max_amax={amax:.4f} -> scale={safe_scale:.2f} (device={device})"
                    )
                m.calibration_mode = False
        print("[calibrate] Calibration done. Per-layer scales applied.")

    vram_after = get_vram_mb()
    if not args.profile_baseline:
        print(f"\n[decompose] Done. Replaced {count} layers.")
        print(f"[metrics] VRAM Before: {vram_before_decomp:.0f} MB")
        print(f"[metrics] VRAM After:  {vram_after:.0f} MB")
        print(f"[metrics] Savings:     {vram_before_decomp - vram_after:.0f} MB")

    # 5.2 Capture Activations (Optional)
    if args.capture_activations:
        print("\n[capture] Running capture pass...")
        _ = run_inference(config=cfg, pipeline=pipeline)

        captured = []
        for m in transformer.modules():
            if isinstance(m, LowRankDeltaLinear) and hasattr(m, "_captured_data"):
                captured.extend(m._captured_data)

        if captured:
            out_p = Path("ffn_delta_outputs/captured_activations.pt")
            out_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(captured, out_p)
            print(f"[capture] Saved {len(captured)} layer activation sets to {out_p}")
        else:
            print("[capture] No data captured!")

    # 6. Profiling / Benchmarking / Verification

    # 6a. Profiling (Exclusive Mode)
    if args.profile:
        if args.profile_baseline:
            wrap_baseline_for_profiling(transformer)

        if args.compile:
            if not args.profile_baseline:
                print("\n[profile] Compiling Decomposed Model (Full Transformer)...")
                transformer = _compile_model(transformer)
                _set_transformer_on_pipeline(pipeline, transformer)
            else:
                print("\n[profile] Compiling Baseline Model for Profiling...")
                transformer = _compile_model(transformer)
                _set_transformer_on_pipeline(pipeline, transformer)

        # Build hyper-descriptive filename
        flags = []
        if args.profile_baseline:
            flags.append("baseline")
        else:
            flags.append(f"r{args.rank}")

        flags.extend(
            [f"f{args.num_frames}", f"s{args.static_scale}", f"b{args.blocks}"]
        )

        if args.ablation_no_q:
            flags.append("noq")
        if args.ablation_only_q:
            flags.append("onlyq")
        if args.compile:
            flags.append("compile")

        trace_filename = f"./ffn_delta_outputs/profile_{'_'.join(flags)}.json"
        print(
            f"\n[decompose] Running with Profiler. Trace will be saved to {trace_filename}"
        )

        # 1. Warmup Pass (skip compilation overhead in trace)
        print("[decompose] Warmup pass (no profiling)...")
        cfg_warmup = copy.copy(cfg)
        run_inference(config=cfg_warmup, pipeline=pipeline)

        # 2. Profile Pass
        print(f"[decompose] Starting profile capture (Full Run: Inference Only)...")
        # Schedule: warmup=0, active=1
        profile_schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)

        video_np = None
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=profile_schedule,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as prof:

            with torch.profiler.record_function("FULL_INFERENCE_CAPTURE"):
                with torch.cuda.nvtx.range("FULL_INFERENCE_CAPTURE"):
                    cfg_prof = copy.copy(cfg)
                    video_np = run_inference(config=cfg_prof, pipeline=pipeline)

            # End Step 0 (Active) -> Trigger Save
            prof.step()

        # Export trace
        prof.export_chrome_trace(trace_filename)
        print(f"[decompose] Profile export complete: {trace_filename}")

        # 3. Save Video (Outside Profiler as requested)
        if video_np is not None and args.output_video:
            export_to_video(video_np, args.output_video, fps=args.frame_rate)

    # 6b. Benchmarking (Exclusive Mode, skipped if profiled)
    elif args.benchmark and not args.skip_benchmark:
        if args.compile:
            if not args.profile_baseline:
                print("\n[bench] Compiling Decomposed Model (Full Transformer)...")
                transformer = _compile_model(transformer)
                _set_transformer_on_pipeline(pipeline, transformer)
            else:
                print("\n[bench] Compiling Baseline Model for Profiling...")
                transformer = _compile_model(transformer)
                _set_transformer_on_pipeline(pipeline, transformer)

        print("\n[bench] Measuring Decomposed Latency...")
        lat_decomp = benchmark_latency(
            pipeline,
            prompt=args.prompt,
            num_frames=args.num_frames,
            seed=args.seed,
            runs=args.benchmark_runs,
            pipeline_config=args.pipeline_config,
            conditioning_image=args.conditioning_image,
        )
        print(f"[bench] Decomposed: {lat_decomp:.3f} s/run")
        if lat_base > 0:
            print(f"[bench] Speedup: {lat_base / lat_decomp:.2f}x")
        else:
            print("[bench] Speedup: N/A (baseline skipped)")

    # 7. Quality Verification (Regular video generation, not inside benchmark block)
    if args.output_video and not args.profile:
        if args.compile and args.skip_decomposition:
            print("\n[verify] Compiling Baseline Model (Verify)...")
            transformer = _compile_model(transformer)
            _set_transformer_on_pipeline(pipeline, transformer)
        print(f"\n[verify] Generating video to {args.output_video}")
        conditioning_images = None
        if args.conditioning_image:
            conditioning_images = [
                ConditioningItemRaw(
                    media_item=args.conditioning_image,
                    start_frame_number=0,
                    strength=1.0,
                )
            ]
        cfg_verify = InferenceConfig(
            prompt=args.prompt,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            seed=args.seed,
            conditioning_images=conditioning_images,
        )
        cfg_verify.pipeline_config = args.pipeline_config

        video_np = run_inference(config=cfg_verify, pipeline=pipeline)
        if video_np is not None:
            out_path = Path(args.output_video)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(video_np, str(out_path), fps=args.frame_rate)
            print(f"[verify] Generation complete: {out_path}")
        else:
            print("[verify] ERROR: run_inference returned None.")


if __name__ == "__main__":
    main()
