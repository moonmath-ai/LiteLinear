#!/usr/bin/env python3
"""
Multi-GPU capture for FFN autocorrelation matrices R = X^T X / N.

Intended usage:
  torchrun --standalone --nproc_per_node=$NUM_GPUS ffn/capture_r_ddp.py ...

Notes:
- Prompts are partitioned round-robin by global index (i % world_size == rank).
- --max_prompts is a GLOBAL cap on prompt indices (not per-rank).
- Checkpoints are written per-rank with a `.rank{rank}` suffix.
"""

from __future__ import annotations

import argparse
import os
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from lite_linear.ffn_delta import (  # noqa: E402
    AutocorrAccumulator,
    FFNLinearRef,
    iter_ltx_ffn_linears,
    r_sqrt_and_inv,
    save_r_matrices,
)
from ltx_video.inference import (  # noqa: E402
    InferenceConfig,
    init_inference,
    run_inference,
)


def _repo_root() -> Path:
    # ffn/capture_r_ddp.py -> repo root is 1 parent up
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


def _iter_prompts_file(path: Path):
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


def _resolve_accum_device(name: str, *, local_rank: int) -> str:
    norm = name.strip().lower()
    if norm in {"cuda", "cuda:local", "local"}:
        return f"cuda:{local_rank}"
    if norm.startswith("cuda:"):
        return norm
    if norm == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported --accum_device: {name}")


def _resolve_rsqrt_device(name: str, *, local_rank: int) -> torch.device:
    norm = name.strip().lower()
    if norm in {"cpu"}:
        return torch.device("cpu")
    if norm in {"cuda", "cuda:0", "gpu0", "0", "cuda:local", "local"}:
        return torch.device(f"cuda:{local_rank}")
    if norm.startswith("cuda:"):
        return torch.device(norm)
    raise ValueError(f"Unsupported --rsqrt_device: {name}")


def _get_rank_info() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit(
            "Expected torchrun environment (RANK/WORLD_SIZE). "
            "Launch with torchrun --standalone --nproc_per_node=N."
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, local_rank, world_size


def _rank_print(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}")


def _format_block_path(base_dir: Path, part: str, block_idx: int, suffix: str) -> Path:
    return base_dir / f"r_{part}_b{block_idx}{suffix}.pt"


def _raw_out_path(path: Path) -> Path:
    return path.with_suffix(".raw.pt")


def _with_step_suffix(path: Path, step: int) -> Path:
    return path.with_suffix(f".step{step}{path.suffix}")


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
        help="Accumulator device: cpu, cuda, or cuda:N (default: cuda:{local_rank})",
    )
    ap.add_argument(
        "--accum_device_rank0_only",
        action="store_true",
        help="Use --accum_device only on rank 0; other ranks use their local GPU",
    )
    ap.add_argument(
        "--rank0_accum_only",
        action="store_true",
        help="Rank 0 skips inference and only aggregates/saves (inference starts at rank 1)",
    )
    ap.add_argument(
        "--skip_rsqrt",
        action="store_true",
        help="Skip computing/saving R^{1/2} and R^{-1/2} outputs",
    )
    ap.add_argument(
        "--rsqrt_device",
        default="cuda:0",
        help="Device for R^{1/2} and R^{-1/2} (cpu, cuda, cuda:N)",
    )
    ap.add_argument("--which", default="w1", help="Comma list: w1,w2 (default w1)")
    ap.add_argument(
        "--blocks",
        default=None,
        help="Comma list of block indices or ranges to capture (e.g. '0,1,2' or '0-3'). Default: all blocks",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output .pt path for combined name->R matrices (rank0 only)",
    )
    ap.add_argument(
        "--save_r",
        action="store_true",
        help="Write normalized R outputs (default: raw xtx + n_rows only).",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Directory to write per-block/per-part R matrices (rank0 only)",
    )
    ap.add_argument(
        "--out_rsqrt",
        default=None,
        help="Optional output .pt path for combined name->R^{1/2} matrices (rank0 only)",
    )
    ap.add_argument(
        "--out_rsqrt_inv",
        default=None,
        help="Optional output .pt path for combined name->R^{-1/2} matrices (rank0 only)",
    )
    ap.add_argument(
        "--out_rsqrt_dir",
        default=None,
        help="Directory to write per-block/per-part R^{1/2} matrices (rank0 only)",
    )
    ap.add_argument(
        "--out_rsqrt_inv_dir",
        default=None,
        help="Directory to write per-block/per-part R^{-1/2} matrices (rank0 only)",
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
        "--reduce_every",
        type=int,
        default=0,
        help="All-reduce + save partial outputs every N global prompts (0=disabled).",
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

    if args.out is None and args.out_dir is None:
        raise SystemExit("Provide --out and/or --out_dir to save results.")
    if args.reduce_every < 0:
        raise SystemExit("--reduce_every must be >= 0")

    out_dir = Path(args.out_dir) if args.out_dir else None
    out_rsqrt_dir = Path(args.out_rsqrt_dir) if args.out_rsqrt_dir else None
    out_rsqrt_inv_dir = Path(args.out_rsqrt_inv_dir) if args.out_rsqrt_inv_dir else None

    want_combined = args.out is not None
    want_per_block = out_dir is not None
    want_rsqrt = not args.skip_rsqrt and any(
        v is not None
        for v in (
            args.out_rsqrt,
            args.out_rsqrt_inv,
            out_rsqrt_dir,
            out_rsqrt_inv_dir,
        )
    )
    want_rsqrt_per_block = (not args.skip_rsqrt) and (
        out_rsqrt_dir is not None or out_rsqrt_inv_dir is not None
    )
    save_r = args.save_r

    # Ensure output dirs exist (rank 0 is the only writer).
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    if out_rsqrt_dir is not None:
        out_rsqrt_dir.mkdir(parents=True, exist_ok=True)
    if out_rsqrt_inv_dir is not None:
        out_rsqrt_inv_dir.mkdir(parents=True, exist_ok=True)

    rank, local_rank, world_size = _get_rank_info()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for DDP capture (torch.cuda.is_available() == False)")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    which = set([p.strip() for p in args.which.split(",") if p.strip()])

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise SystemExit(
            f"--prompts file not found: {prompts_path}\n"
            f"Tip: download prompts to {DEFAULT_PROMPTS_PATH} or pass --prompts explicitly."
        )

    device = torch.device("cuda", local_rank)

    if args.rank0_accum_only and world_size < 2:
        raise SystemExit("--rank0_accum_only requires world_size >= 2")

    inference_ranks = [
        r for r in range(world_size) if not (args.rank0_accum_only and r == 0)
    ]
    if not inference_ranks:
        raise SystemExit("No inference ranks available (check --rank0_accum_only)")
    infer_count = len(inference_ranks)
    if args.reduce_every > 0 and args.reduce_every % infer_count != 0:
        new_reduce = ((args.reduce_every + infer_count - 1) // infer_count) * infer_count
        if rank == 0:
            _rank_print(
                rank,
                f"[config] reduce_every adjusted from {args.reduce_every} to {new_reduce} "
                f"to align with inference_ranks={infer_count}",
            )
        args.reduce_every = new_reduce
    do_inference = rank in inference_ranks
    meta_rank = inference_ranks[0]

    # Count total prompts (on rank 0, broadcast to others)
    if rank == 0:
        dataset_size = _count_prompts(prompts_path)
        _rank_print(rank, f"Dataset has {dataset_size} prompts")
        _rank_print(rank, f"Using seed={args.seed}, will process {args.max_prompts} prompts")
    else:
        dataset_size = 0

    # Broadcast dataset_size to all ranks (NCCL requires CUDA tensors)
    dataset_size_tensor = torch.tensor(dataset_size, dtype=torch.int64, device=device)
    dist.broadcast(dataset_size_tensor, src=0)
    dataset_size = int(dataset_size_tensor.item())

    # Broadcast seed to all ranks (for consistency check)
    seed_tensor = torch.tensor(args.seed, dtype=torch.int64, device=device)
    dist.broadcast(seed_tensor, src=0)
    args.seed = int(seed_tensor.item())

    _rank_print(
        rank,
        f"ddp init rank={rank} local_rank={local_rank} world_size={world_size} seed={args.seed} "
        f"inference_rank={do_inference} rank0_accum_only={args.rank0_accum_only}",
    )

    if not args.no_compile:
        raise SystemExit(
            "--no_compile is required for now (hooking compiled modules is unsafe)"
        )

    pipeline = None
    transformer = None
    total_blocks = 0
    meta_tensor = None
    meta_count = 0

    if do_inference:
        pipeline = init_inference(config=args.pipeline_config, compile=False)
        transformer = _get_transformer_from_pipeline(pipeline)

    if rank == meta_rank:
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None:
            raise ValueError("Expected transformer to have .transformer_blocks")
        total_blocks = len(blocks)
        meta_rows = []
        for ref, mod in iter_ltx_ffn_linears(transformer):
            W = getattr(mod, "weight")
            d_in = int(W.shape[1])
            part_id = 0 if ref.part == "w1" else 1
            meta_rows.append((ref.block_idx, part_id, d_in))
        meta_tensor = torch.tensor(meta_rows, device=device, dtype=torch.int64)
        meta_count = meta_tensor.shape[0]

    total_blocks_tensor = torch.tensor(total_blocks, device=device, dtype=torch.int64)
    dist.broadcast(total_blocks_tensor, src=meta_rank)
    total_blocks = int(total_blocks_tensor.item())

    meta_count_tensor = torch.tensor(meta_count, device=device, dtype=torch.int64)
    dist.broadcast(meta_count_tensor, src=meta_rank)
    meta_count = int(meta_count_tensor.item())

    if rank != meta_rank:
        meta_tensor = torch.empty((meta_count, 3), device=device, dtype=torch.int64)
    if meta_tensor is None:
        raise RuntimeError("Internal error: meta_tensor not initialized")
    dist.broadcast(meta_tensor, src=meta_rank)

    _rank_print(
        rank,
        f"Model metadata: total_blocks={total_blocks} refs={meta_count}",
    )

    # If --blocks not specified, default to all blocks
    if args.blocks is None:
        args.blocks = f"0-{total_blocks-1}"
        _rank_print(rank, f"No --blocks specified, defaulting to all blocks: {args.blocks}")

    block_idxs = set(_parse_layers(args.blocks))

    accs: Dict[str, AutocorrAccumulator] = {}
    ref_by_name: Dict[str, Any] = {}
    hooks = []
    start_idx = 0

    ckpt_base = Path(args.checkpoint_path) if args.checkpoint_path else None
    ckpt_path = Path(f"{ckpt_base}.rank{rank}") if ckpt_base else None
    ckpt_accs: Dict[str, Any] = {}
    if args.resume:
        if ckpt_path is None:
            raise SystemExit("--resume requires --checkpoint_path")
        if ckpt_path.exists():
            ckpt = _load_checkpoint(ckpt_path)
            # Check if checkpoint uses old format (prompt_idx) or new format (sequence-based)
            if "prompt_idx" in ckpt:
                start_idx = int(ckpt.get("prompt_idx", 0))
            else:
                start_idx = int(ckpt.get("seq_idx", 0))
            # Verify seed matches if present in checkpoint
            ckpt_seed = ckpt.get("seed")
            if ckpt_seed is not None and ckpt_seed != args.seed:
                _rank_print(rank, f"[WARNING] Checkpoint seed={ckpt_seed} differs from current seed={args.seed}")
            ckpt_accs = ckpt.get("accs", {}) or {}
            _rank_print(rank, f"[ckpt] loaded {ckpt_path} (resume from seq_idx={start_idx})")
        else:
            _rank_print(rank, f"[ckpt] no checkpoint found at {ckpt_path}; starting from scratch")

    def make_hook(name: str):
        def hook(mod, inp, out):  # noqa: ANN001
            x = inp[0]
            accs[name].update(x)

        return hook

    if args.gpu_accum and torch.cuda.is_available():
        if args.accum_device:
            if args.accum_device_rank0_only and rank != 0:
                accum_device = f"cuda:{local_rank}"
            else:
                accum_device = _resolve_accum_device(
                    args.accum_device, local_rank=local_rank
                )
        else:
            accum_device = f"cuda:{local_rank}"
    else:
        accum_device = "cpu"
    accum_dtype = _parse_accum_dtype(args.accum_dtype)
    _rank_print(rank, f"accumulation device: {accum_device} dtype: {accum_dtype}")

    module_by_name: Dict[str, Any] = {}
    if do_inference:
        for ref, mod in iter_ltx_ffn_linears(transformer):
            module_by_name[ref.name] = mod

    meta_rows = meta_tensor.to(device="cpu").tolist()
    for block_idx, part_id, d_in in meta_rows:
        part = "w1" if part_id == 0 else "w2"
        if block_idx not in block_idxs or part not in which:
            continue
        name = f"transformer_blocks.{block_idx}.ff.{part}"
        accs[name] = AutocorrAccumulator(
            dim=int(d_in), device=accum_device, dtype=accum_dtype
        )
        ref_by_name[name] = FFNLinearRef(block_idx, part, name)
        if name in ckpt_accs:
            accs[name].load_state_dict(ckpt_accs[name])
            accs[name].xtx = accs[name].xtx.to(
                device=accum_device, dtype=accum_dtype
            )
        if do_inference:
            mod = module_by_name.get(name)
            if mod is None:
                raise RuntimeError(f"Missing module for {name} on rank {rank}")
            hooks.append(mod.register_forward_hook(make_hook(name)))
        tag = "hook" if do_inference else "accum"
        _rank_print(
            rank,
            f"[{tag}] {name} d_in={int(d_in)} resume={'yes' if name in ckpt_accs else 'no'}",
        )

    def reduce_and_save(*, step: Optional[int], log_layers: bool) -> None:
        is_partial = step is not None
        total_layers = len(accs)
        if rank == 0:
            if is_partial:
                _rank_print(rank, f"[reduce] begin snapshot step={step} layers={total_layers}")
            else:
                _rank_print(rank, f"[reduce] begin all-reduce for {total_layers} layers")

        r_mats: Dict[str, torch.Tensor] = {}
        r_sqrt_mats: Dict[str, torch.Tensor] = {}
        r_sqrt_inv_mats: Dict[str, torch.Tensor] = {}
        raw_xtx: Dict[str, torch.Tensor] = {}
        raw_n_rows: Dict[str, int] = {}
        rsqrt_device = torch.device("cpu")
        if want_rsqrt and rank == 0:
            rsqrt_device = _resolve_rsqrt_device(args.rsqrt_device, local_rank=local_rank)
            if not is_partial:
                _rank_print(rank, f"[rsqrt] device={rsqrt_device}")

        for idx, (name, acc) in enumerate(accs.items(), start=1):
            xtx = acc.xtx
            if xtx.device != device:
                xtx = xtx.to(device=device)
            else:
                xtx = xtx.clone()
            n_rows = torch.tensor(acc.n_rows, device=device, dtype=torch.int64)

            if rank == 0 and log_layers:
                _rank_print(rank, f"[reduce] ({idx}/{total_layers}) {name} all_reduce start")
            dist.all_reduce(xtx, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_rows, op=dist.ReduceOp.SUM)

            if rank == 0:
                total_rows = int(n_rows.item())
                if total_rows <= 0:
                    raise RuntimeError(f"No samples accumulated for {name}")
                xtx_cpu = xtx.cpu()
                r = None
                r_cpu = None
                if save_r or want_rsqrt:
                    r = (xtx / float(total_rows)).to(dtype=torch.float32)
                    r_cpu = r.cpu()

                if want_combined:
                    raw_xtx[name] = xtx_cpu
                    raw_n_rows[name] = total_rows
                    if save_r and r_cpu is not None:
                        r_mats[name] = r_cpu
                if log_layers:
                    _rank_print(
                        rank,
                        f"[reduce] ({idx}/{total_layers}) {name} all_reduce done n_rows={total_rows}",
                    )
                    _rank_print(rank, f"[R] {name} n_rows={total_rows}")

                ref = ref_by_name.get(name)
                if (want_per_block or want_rsqrt_per_block) and ref is None:
                    raise RuntimeError(f"Missing block ref for {name}")

                if want_per_block and out_dir is not None:
                    suffix = "" if not is_partial else f"_step{step}"
                    if save_r and r_cpu is not None:
                        save_r_matrices(
                            _format_block_path(out_dir, ref.part, ref.block_idx, suffix),
                            {name: r_cpu},
                        )
                    raw_suffix = "_raw" if not is_partial else f"_raw_step{step}"
                    raw_block_path = _format_block_path(
                        out_dir, ref.part, ref.block_idx, raw_suffix
                    )
                    torch.save(
                        {"version": 1, "xtx": {name: xtx_cpu}, "n_rows": {name: total_rows}},
                        raw_block_path,
                    )

                if want_rsqrt and r is not None:
                    if log_layers:
                        _rank_print(rank, f"[rsqrt] ({idx}/{total_layers}) {name} start")
                    r_for_sqrt = r
                    if r_for_sqrt.device != rsqrt_device:
                        r_for_sqrt = r_for_sqrt.to(device=rsqrt_device)
                    r_sqrt, r_sqrt_inv = r_sqrt_and_inv(r_for_sqrt)
                    r_sqrt = r_sqrt.cpu()
                    r_sqrt_inv = r_sqrt_inv.cpu()
                    if log_layers:
                        _rank_print(rank, f"[rsqrt] ({idx}/{total_layers}) {name} done")
                    if args.out_rsqrt:
                        r_sqrt_mats[name] = r_sqrt
                    if args.out_rsqrt_inv:
                        r_sqrt_inv_mats[name] = r_sqrt_inv
                    if out_rsqrt_dir is not None:
                        rsqrt_suffix = "_rsqrt" if not is_partial else f"_step{step}_rsqrt"
                        save_r_matrices(
                            _format_block_path(
                                out_rsqrt_dir, ref.part, ref.block_idx, rsqrt_suffix
                            ),
                            {name: r_sqrt},
                        )
                    if out_rsqrt_inv_dir is not None:
                        rsqrt_inv_suffix = (
                            "_rsqrt_inv" if not is_partial else f"_step{step}_rsqrt_inv"
                        )
                        save_r_matrices(
                            _format_block_path(
                                out_rsqrt_inv_dir,
                                ref.part,
                                ref.block_idx,
                                rsqrt_inv_suffix,
                            ),
                            {name: r_sqrt_inv},
                        )

        if rank == 0:
            if args.out:
                out_path = Path(args.out)
                if is_partial:
                    out_path = _with_step_suffix(out_path, int(step))
                if save_r:
                    save_r_matrices(out_path, r_mats)
                    _rank_print(rank, f"Wrote: {out_path}")
                raw_out_path = _raw_out_path(out_path)
                torch.save(
                    {"version": 1, "xtx": raw_xtx, "n_rows": raw_n_rows},
                    raw_out_path,
                )
                _rank_print(rank, f"Wrote raw: {raw_out_path}")
            if args.out_rsqrt and want_rsqrt:
                out_path = Path(args.out_rsqrt)
                if is_partial:
                    out_path = _with_step_suffix(out_path, int(step))
                save_r_matrices(out_path, r_sqrt_mats)
                _rank_print(rank, f"Wrote: {out_path}")
            if args.out_rsqrt_inv and want_rsqrt:
                out_path = Path(args.out_rsqrt_inv)
                if is_partial:
                    out_path = _with_step_suffix(out_path, int(step))
                save_r_matrices(out_path, r_sqrt_inv_mats)
                _rank_print(rank, f"Wrote: {out_path}")
            if not is_partial:
                if out_dir is not None:
                    if save_r:
                        _rank_print(rank, f"Wrote per-block R to: {out_dir}")
                    else:
                        _rank_print(rank, f"Wrote per-block raw xtx to: {out_dir}")
                if out_rsqrt_dir is not None and want_rsqrt:
                    _rank_print(rank, f"Wrote per-block R_sqrt to: {out_rsqrt_dir}")
                if out_rsqrt_inv_dir is not None and want_rsqrt:
                    _rank_print(rank, f"Wrote per-block R_sqrt_inv to: {out_rsqrt_inv_dir}")

    try:
        # Process prompts using on-demand deterministic indexing, round-robin by inference rank
        processed = 0
        for seq_idx in range(start_idx, args.max_prompts):
            reduce_step = args.reduce_every > 0 and (seq_idx + 1) % args.reduce_every == 0
            checkpoint_due = False
            target_rank = inference_ranks[seq_idx % len(inference_ranks)]
            if rank == target_rank:
                # Get deterministic prompt index for this sequence position
                target_idx = _get_rnd_index(
                    args.seed, seq_idx, idx_from=0, idx_to=dataset_size
                )

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

                if do_inference:
                    cfg = InferenceConfig(prompt=prompt)
                    cfg.pipeline_config = args.pipeline_config
                    cfg.num_frames = args.num_frames
                    _ = run_inference(config=cfg, pipeline=pipeline)
                    processed += 1
                    _rank_print(
                        rank,
                        f"[run] prompt_idx={target_idx} (seq={seq_idx+1}) processed_local={processed} global_max={args.max_prompts}",
                    )

                if ckpt_path is not None and args.checkpoint_every > 0:
                    if do_inference and processed % args.checkpoint_every == 0:
                        checkpoint_due = True
                        if not reduce_step:
                            payload = {
                                "version": 1,
                                "seq_idx": seq_idx + 1,  # next sequence index to process
                                "pipeline_config": args.pipeline_config,
                                "prompts_path": str(args.prompts),
                                "max_prompts": int(args.max_prompts),
                                "seed": int(args.seed),
                                "which": sorted(which),
                                "blocks": sorted(block_idxs),
                                "rank": rank,
                                "world_size": world_size,
                                "accs": {k: v.state_dict() for k, v in accs.items()},
                            }
                            _atomic_torch_save(payload, ckpt_path)
                            _rank_print(
                                rank,
                                f"[ckpt] saved {ckpt_path} (next_seq_idx={seq_idx+1}, processed_local={processed})",
                            )

            if reduce_step:
                if checkpoint_due:
                    payload = {
                        "version": 1,
                        "seq_idx": seq_idx + 1,  # next sequence index to process
                        "pipeline_config": args.pipeline_config,
                        "prompts_path": str(args.prompts),
                        "max_prompts": int(args.max_prompts),
                        "seed": int(args.seed),
                        "which": sorted(which),
                        "blocks": sorted(block_idxs),
                        "rank": rank,
                        "world_size": world_size,
                        "accs": {k: v.state_dict() for k, v in accs.items()},
                    }
                    _atomic_torch_save(payload, ckpt_path)
                    _rank_print(
                        rank,
                        f"[ckpt] saved {ckpt_path} before reduce (next_seq_idx={seq_idx+1}, processed_local={processed})",
                    )
                reduce_and_save(step=seq_idx + 1, log_layers=False)
    finally:
        for h in hooks:
            h.remove()

    if args.rank0_accum_only and args.reduce_every == 0:
        if rank == 0:
            for src in inference_ranks:
                if src == 0:
                    continue
                token = torch.zeros(1, device=device, dtype=torch.int64)
                dist.recv(token, src=src)
            _rank_print(rank, "[sync] received done signals from all inference ranks")
        elif do_inference:
            token = torch.ones(1, device=device, dtype=torch.int64)
            dist.send(token, dst=0)
    reduce_and_save(step=None, log_layers=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
