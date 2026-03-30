#!/usr/bin/env python3
"""
Capture FFN autocorrelation matrices R = X^T X / N for LTX transformer FFN linears.

This script is intentionally simple and meant for research runs.
It uses forward hooks on FFN linear layers and accumulates X^T X.

Important:
- This can be extremely heavy if you try to capture for many layers/dims.
- Start with a small subset of layers (e.g. one block, one of w1/w2).

Checkpointing / resume (spec §2.4):
- Use `--checkpoint_path` to save accumulator state periodically (every `--checkpoint_every` prompts).
- Use `--resume` to resume from an existing checkpoint (restores prompt index and accumulators).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math

import torch

from lite_linear.ffn_delta import (
    AutocorrAccumulator,
    iter_ltx_ffn_linears,
    save_r_matrices,
)  # noqa: E402
from ltx_video.inference import (
    InferenceConfig,
    init_inference,
    run_inference,
)  # noqa: E402


def _repo_root() -> Path:
    # ffn/capture_r.py -> repo root is 1 parent up
    return Path(__file__).resolve().parents[1]


DEFAULT_PROMPTS_PATH = (
    _repo_root() / "third_party" / "data" / "vidprom_filtered_extended.txt"
)


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


def _parse_layers(s: str) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def _iter_prompts_file(path: Path) -> Any:
    # Yields non-empty stripped lines (prompts) from a text file without loading it into RAM.
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.strip()
            if p:
                yield p


def _count_prompts(path: Path) -> int:
    """Count total number of non-empty prompts in file."""
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _get_rnd_index(seed: int, i: int, idx_from: int = 0, idx_to: int = 0) -> int:
    """
    Deterministic function that maps sequence index i to a unique prompt index.

    Uses a seeded random permutation to ensure:
    - Deterministic by seed
    - Unique indices for different i values (when i < idx_to - idx_from)
    - All indices in range [idx_from, idx_to)

    For efficiency with large datasets, uses a hash-based permutation that
    generates the i-th element of a deterministic shuffle without storing
    the full permutation.

    Args:
        seed: Random seed for determinism
        i: Sequence index (0, 1, 2, ...)
        idx_from: Start of index range (inclusive, default 0)
        idx_to: End of index range (exclusive)

    Returns:
        Unique prompt index in range [idx_from, idx_to)
    """
    if idx_to <= idx_from:
        raise ValueError(f"Invalid range: idx_from={idx_from}, idx_to={idx_to}")

    size = idx_to - idx_from

    # Use a modular linear permutation: idx = (a*i + b) mod size
    # Choose 'a' to be coprime with size for a full permutation.
    if size == 1:
        return idx_from

    a = (seed * 2 + 1) % size
    if a == 0:
        a = 1
    while math.gcd(a, size) != 1:
        a = (a + 2) % size
        if a == 0:
            a = 1

    b = seed % size
    return (a * (i % size) + b) % size + idx_from


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or ckpt.get("version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return ckpt


def _parse_accum_dtype(name: str) -> torch.dtype:
    norm = name.strip().lower()
    if norm in {"float64", "fp64", "f64"}:
        return torch.float64
    if norm in {"float32", "fp32", "f32"}:
        return torch.float32
    if norm in {"float16", "fp16", "f16"}:
        return torch.float16
    raise ValueError(f"Unsupported --accum_dtype: {name}")


def _resolve_accum_device(name: str) -> str:
    norm = name.strip().lower()
    if norm in {"cuda", "cuda:local", "local"}:
        return "cuda"
    if norm.startswith("cuda:"):
        return norm
    if norm == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported --accum_device: {name}")


def _raw_out_path(path: Path) -> Path:
    return path.with_suffix(".raw.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_config", default="configs/ltxv-13b-0.9.8-dev.yaml")
    ap.add_argument(
        "--prompts",
        default=str(DEFAULT_PROMPTS_PATH),
        help=(
            "Text file with one prompt per line. "
            f"Default: {DEFAULT_PROMPTS_PATH} (if present)"
        ),
    )
    ap.add_argument("--max_prompts", type=int, default=10)
    ap.add_argument(
        "--num_frames",
        type=int,
        default=17,
        help="Number of frames to generate per prompt (default: 17)",
    )
    ap.add_argument(
        "--gpu_accum",
        action="store_true",
        help="Accumulate X^T X on GPU (faster but uses more VRAM)",
    )
    ap.add_argument(
        "--accum_dtype",
        default="float64",
        help="Accumulator dtype: float64 (default), float32, or float16",
    )
    ap.add_argument(
        "--accum_device",
        default=None,
        help="Accumulator device: cpu, cuda, or cuda:N (default: cuda if --gpu_accum)",
    )
    ap.add_argument("--which", default="w1", help="Comma list: w1,w2 (default w1)")
    ap.add_argument(
        "--blocks",
        default=None,
        help="Comma list of block indices or ranges to capture (e.g. '0,1,2' or '0-3'). Default: all blocks",
    )
    ap.add_argument("--out", required=True, help="Output .pt path for name->R matrices")
    ap.add_argument(
        "--save_r",
        action="store_true",
        help="Write normalized R outputs (default: raw xtx + n_rows only).",
    )
    ap.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile (recommended)"
    )
    ap.add_argument(
        "--checkpoint_path",
        default=None,
        help="Optional path to save checkpoint state (accumulators + prompt index).",
    )
    ap.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Save checkpoint every N prompts (default: 10).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --checkpoint_path if it exists (restores prompt index + accumulators).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic prompt selection (default: 42)",
    )
    args = ap.parse_args()

    which = set([p.strip() for p in args.which.split(",") if p.strip()])

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise SystemExit(
            f"--prompts file not found: {prompts_path}\n"
            f"Tip: download prompts to {DEFAULT_PROMPTS_PATH} or pass --prompts explicitly."
        )

    # Count total prompts
    dataset_size = _count_prompts(prompts_path)
    print(f"[capture_r] Dataset has {dataset_size} prompts")
    print(f"[capture_r] Using seed={args.seed}, will process {args.max_prompts} prompts")

    pipeline = init_inference(config=args.pipeline_config, compile=not args.no_compile)
    if not args.no_compile:
        raise SystemExit(
            "--no_compile is required for now (hooking compiled modules is unsafe)"
        )

    transformer = _get_transformer_from_pipeline(pipeline)

    # Determine total number of blocks
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("Expected transformer to have .transformer_blocks")
    total_blocks = len(blocks)
    print(f"[capture_r] Model has {total_blocks} transformer blocks (0-{total_blocks-1})")

    # If --blocks not specified, default to all blocks
    if args.blocks is None:
        args.blocks = f"0-{total_blocks-1}"
        print(f"[capture_r] No --blocks specified, defaulting to all blocks: {args.blocks}")

    block_idxs = set(_parse_layers(args.blocks))

    accs: Dict[str, AutocorrAccumulator] = {}
    hooks = []
    start_idx = 0

    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    ckpt_accs: Dict[str, Any] = {}
    if args.resume:
        if ckpt_path is None:
            raise SystemExit("--resume requires --checkpoint_path")
        if ckpt_path.exists():
            ckpt = _load_checkpoint(ckpt_path)
            key = "seq_idx" if "seq_idx" in ckpt else "prompt_idx"
            start_idx = int(ckpt.get(key, 0))
            ckpt_seed = ckpt.get("seed")
            if ckpt_seed is not None and ckpt_seed != args.seed:
                print(
                    f"[ckpt] WARNING: checkpoint seed={ckpt_seed} differs from current seed={args.seed}"
                )
            ckpt_accs = ckpt.get("accs", {}) or {}
            print(f"[ckpt] loaded {ckpt_path} (resume from {key}={start_idx})")
        else:
            print(f"[ckpt] no checkpoint found at {ckpt_path}; starting from scratch")

    def make_hook(name: str):
        def hook(mod, inp, out):  # noqa: ANN001
            x = inp[0]
            accs[name].update(x)

        return hook

    if args.gpu_accum and torch.cuda.is_available():
        accum_device = "cuda"
        if args.accum_device:
            accum_device = _resolve_accum_device(args.accum_device)
    else:
        accum_device = "cpu"
    accum_dtype = _parse_accum_dtype(args.accum_dtype)
    print(f"[capture_r] accumulation device: {accum_device} dtype: {accum_dtype}")

    # Register hooks
    for ref, mod in iter_ltx_ffn_linears(transformer):
        if ref.block_idx not in block_idxs:
            continue
        if ref.part not in which:
            continue

        W = getattr(mod, "weight")
        d_in = W.shape[1]
        accs[ref.name] = AutocorrAccumulator(
            dim=d_in, device=accum_device, dtype=accum_dtype
        )
        if ref.name in ckpt_accs:
            accs[ref.name].load_state_dict(ckpt_accs[ref.name])
            # Ensure loaded state is on the correct device
            accs[ref.name].xtx = accs[ref.name].xtx.to(
                device=accum_device, dtype=accum_dtype
            )

        hooks.append(mod.register_forward_hook(make_hook(ref.name)))
        print(
            f"[hook] {ref.name} d_in={d_in} resume={'yes' if ref.name in ckpt_accs else 'no'}"
        )

    try:
        # Process prompts using on-demand deterministic indexing
        processed = 0
        for seq_idx in range(start_idx, args.max_prompts):
            # Get deterministic prompt index for this sequence position
            target_idx = _get_rnd_index(args.seed, seq_idx, idx_from=0, idx_to=dataset_size)

            # Seek to the target prompt index
            prompt = None
            i = 0
            for p in _iter_prompts_file(prompts_path):
                if i == target_idx:
                    prompt = p
                    break
                i += 1

            if prompt is None:
                raise RuntimeError(f"Failed to find prompt at index {target_idx}")

            cfg = InferenceConfig(prompt=prompt)
            cfg.pipeline_config = args.pipeline_config
            cfg.num_frames = args.num_frames
            # Keep runs small by default; user can override via pipeline config if desired.
            _ = run_inference(config=cfg, pipeline=pipeline)
            processed += 1
            print(f"[run] prompt_idx={target_idx} (seq={seq_idx+1}) processed={processed}/{args.max_prompts}")

            if ckpt_path is not None and args.checkpoint_every > 0:
                if processed % args.checkpoint_every == 0:
                    payload = {
                        "version": 1,
                        "seq_idx": seq_idx + 1,  # next sequence index to process
                        "pipeline_config": args.pipeline_config,
                        "prompts_path": str(args.prompts),
                        "max_prompts": int(args.max_prompts),
                        "seed": int(args.seed),
                        "which": sorted(which),
                        "blocks": sorted(block_idxs),
                        "accs": {k: v.state_dict() for k, v in accs.items()},
                    }
                    _atomic_torch_save(payload, ckpt_path)
                    print(
                        f"[ckpt] saved {ckpt_path} (next_seq_idx={seq_idx+1}, processed={processed})"
                    )
    finally:
        for h in hooks:
            h.remove()

    r_mats: Dict[str, torch.Tensor] = {}
    if args.save_r:
    for name, acc in accs.items():
        r_mats[name] = acc.finalize_r()
        print(f"[R] {name} n_rows={acc.n_rows}")
    else:
        for name, acc in accs.items():
            print(f"[raw] {name} n_rows={acc.n_rows}")

    raw_payload = {
        "version": 1,
        "xtx": {k: v.xtx.detach().cpu() for k, v in accs.items()},
        "n_rows": {k: int(v.n_rows) for k, v in accs.items()},
    }
    raw_path = _raw_out_path(Path(args.out))
    torch.save(raw_payload, raw_path)
    print(f"Wrote raw: {raw_path}")

    if args.save_r:
    save_r_matrices(args.out, r_mats)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
