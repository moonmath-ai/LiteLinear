#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - optional
    cv2 = None

try:
    from pytorchvideo.models.hub import i3d_r50
except Exception:  # pragma: no cover - optional
    i3d_r50 = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute MSE/PSNR, CLIP similarity, and FVD on videos."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory with .mp4 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for metrics outputs (defaults to input dir).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for CLIP/FVD (cuda/cpu). Default: cuda if available.",
    )
    parser.add_argument(
        "--psnr-threshold",
        type=float,
        default=20.0,
        help="PSNR pass threshold in dB.",
    )
    parser.add_argument(
        "--fvd-degradation-threshold",
        type=float,
        default=10.0,
        help="FVD degradation threshold in percent.",
    )
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=16,
        help="Number of frames sampled for CLIP similarity.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Optional prompts JSON (seed/prompt entries or list of prompts).",
    )
    parser.add_argument(
        "--fvd-frames",
        type=int,
        default=16,
        help="Number of frames sampled for FVD features.",
    )
    parser.add_argument(
        "--fvd-backbone",
        choices=["r3d", "i3d"],
        default="i3d",
        help="Backbone used for FVD features.",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip CLIP similarity.",
    )
    parser.add_argument(
        "--skip-fvd",
        action="store_true",
        help="Skip FVD computation.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on number of test videos to process.",
    )
    return parser.parse_args()


def is_baseline(path: Path) -> bool:
    return "baseline" in path.name


def parse_video_key(name: str):
    seed = None
    frames = None
    seed_match = re.search(r"_s(\d+)_", name)
    if seed_match:
        seed = seed_match.group(1)
    frame_match = re.search(r"_f(\d+)_", name)
    if frame_match:
        frames = frame_match.group(1)
    parts = name.split("_")
    prompt_slug = parts[-1] if parts else name
    # Baseline duplicates often end with `_2`, `_3`, etc; keep the real prompt token.
    if "baseline" in name and prompt_slug.isdigit() and len(parts) >= 2:
        prompt_slug = parts[-2]
    prompt_slug = normalize_prompt_slug(prompt_slug)
    return seed, frames, prompt_slug


def parse_rank(name: str):
    match = re.search(r"_r(\d+)_", name)
    if match:
        return int(match.group(1))
    return None


def normalize_prompt_slug(prompt_slug: str):
    if not prompt_slug:
        return prompt_slug
    prompt_slug = re.sub(r"\.calib$", "", prompt_slug, flags=re.IGNORECASE)
    prompt_slug = re.sub(r"\s*\(\d+\)$", "", prompt_slug)
    prompt_slug = re.sub(
        r"\s+copy(?:\s*\(\d+\))?$", "", prompt_slug, flags=re.IGNORECASE
    )
    return prompt_slug


def prompt_id_from_text(prompt: str, index: int):
    prompt_id = re.sub(r"[^0-9A-Za-z_]", "", prompt[:50])[:30]
    if not prompt_id:
        prompt_id = f"prompt{index}"
    return prompt_id


def load_prompts(prompts_path: Path):
    if not prompts_path or not prompts_path.exists():
        return {}, {}
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_seed = {}
    by_id = {}
    if isinstance(data, list) and data:
        if isinstance(data[0], dict) and "prompt" in data[0]:
            for idx, entry in enumerate(data, start=1):
                prompt = entry.get("prompt", "")
                seed = str(entry.get("seed", "")) if entry.get("seed") else None
                pid = prompt_id_from_text(prompt, idx)
                if seed:
                    by_seed[seed] = prompt
                by_id[pid] = prompt
        elif isinstance(data[0], str):
            for idx, prompt in enumerate(data, start=1):
                pid = prompt_id_from_text(prompt, idx)
                by_id[pid] = prompt
    return by_seed, by_id


def choose_baseline(candidates):
    if not candidates:
        return None
    bf16 = [p for p in candidates if "bf16" in p.name]
    if bf16:
        return sorted(bf16)[0]
    return sorted(candidates)[0]


def compute_mse_psnr(path_a: Path, path_b: Path):
    sse = 0.0
    count = 0
    frames = 0
    with av.open(str(path_a)) as ca, av.open(str(path_b)) as cb:
        it_a = ca.decode(video=0)
        it_b = cb.decode(video=0)
        for frame_a, frame_b in zip(it_a, it_b):
            arr_a = frame_a.to_rgb().to_ndarray()
            arr_b = frame_b.to_rgb().to_ndarray()
            if arr_a.shape != arr_b.shape:
                if cv2 is None:
                    raise RuntimeError(
                        f"Frame size mismatch and cv2 unavailable: {path_a} vs {path_b}"
                    )
                arr_b = cv2.resize(
                    arr_b, (arr_a.shape[1], arr_a.shape[0]), interpolation=cv2.INTER_AREA
                )
            diff = arr_a.astype(np.float32) - arr_b.astype(np.float32)
            sse += float(np.sum(diff * diff))
            count += diff.size
            frames += 1
    if count == 0:
        return None, None, 0
    mse = sse / count
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10((255.0 * 255.0) / mse)
    return mse, psnr, frames


def count_frames(path: Path):
    frames = 0
    with av.open(str(path)) as container:
        for _ in container.decode(video=0):
            frames += 1
    return frames


def sample_indices(total_frames: int, num_frames: int):
    if total_frames <= 0:
        return []
    if num_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))
    indices = np.linspace(0, total_frames - 1, num_frames)
    return [int(round(x)) for x in indices]


def load_frames_for_indices(path: Path, indices):
    indices = list(indices)
    if not indices:
        return []
    index_set = set(indices)
    frames = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in index_set:
                frames[i] = frame.to_rgb().to_ndarray()
                if len(frames) >= len(index_set):
                    break
    return [frames[i] for i in indices if i in frames]


def compute_clip_image_similarity(
    path_a: Path,
    path_b: Path,
    total_frames: int,
    num_frames: int,
    device: torch.device,
    clip_model,
    clip_processor,
):
    indices = sample_indices(total_frames, num_frames)
    frames_a = load_frames_for_indices(path_a, indices)
    frames_b = load_frames_for_indices(path_b, indices)
    if not frames_a or not frames_b:
        return None
    pair_count = min(len(frames_a), len(frames_b))
    frames_a = frames_a[:pair_count]
    frames_b = frames_b[:pair_count]
    images_a = [Image.fromarray(f) for f in frames_a]
    images_b = [Image.fromarray(f) for f in frames_b]
    with torch.no_grad():
        inputs_a = clip_processor(images=images_a, return_tensors="pt")
        inputs_b = clip_processor(images=images_b, return_tensors="pt")
        inputs_a = {k: v.to(device) for k, v in inputs_a.items()}
        inputs_b = {k: v.to(device) for k, v in inputs_b.items()}
        feats_a = clip_model.get_image_features(**inputs_a)
        feats_b = clip_model.get_image_features(**inputs_b)
        feats_a = feats_a / feats_a.norm(dim=-1, keepdim=True)
        feats_b = feats_b / feats_b.norm(dim=-1, keepdim=True)
        sims = (feats_a * feats_b).sum(dim=-1)
    return float(sims.mean().item())


def compute_clip_text_similarity(
    path: Path,
    prompt: str,
    total_frames: int,
    num_frames: int,
    device: torch.device,
    clip_model,
    clip_processor,
    text_feature_cache,
):
    indices = sample_indices(total_frames, num_frames)
    frames = load_frames_for_indices(path, indices)
    if not frames:
        return None
    images = [Image.fromarray(f) for f in frames]
    if prompt not in text_feature_cache:
        with torch.no_grad():
            text_inputs = clip_processor(
                text=[prompt], return_tensors="pt", padding=True, truncation=True
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_feat = clip_model.get_text_features(**text_inputs)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_feature_cache[prompt] = text_feat
    text_feat = text_feature_cache[prompt]
    with torch.no_grad():
        image_inputs = clip_processor(images=images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        image_feats = clip_model.get_image_features(**image_inputs)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        sims = (image_feats @ text_feat.T).squeeze(-1)
    return float(sims.mean().item())


def load_r3d_model(device: torch.device):
    weights = torchvision.models.video.R3D_18_Weights.KINETICS400_V1
    model = torchvision.models.video.r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    preprocess = weights.transforms()
    return model, preprocess


def extract_r3d_feature(path: Path, total_frames: int, num_frames: int, model, preprocess, device):
    indices = sample_indices(total_frames, num_frames)
    frames = load_frames_for_indices(path, indices)
    if not frames:
        return None
    video = torch.from_numpy(np.stack(frames, axis=0))
    # Expected shape for torchvision video presets: (T, C, H, W)
    if video.ndim == 4 and video.shape[-1] == 3:
        video = video.permute(0, 3, 1, 2)
    video = preprocess(video)
    video = video.unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(video).squeeze(0).cpu()
    return feat


def load_i3d_model(device: torch.device):
    if i3d_r50 is None:
        raise RuntimeError(
            "I3D backbone requested but pytorchvideo is unavailable. "
            "Install it with `pip install pytorchvideo`."
        )
    model = i3d_r50(pretrained=True)
    # Use penultimate embedding for Fréchet distance.
    model.blocks[-1].proj = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def extract_i3d_feature(path: Path, total_frames: int, num_frames: int, model, device):
    indices = sample_indices(total_frames, num_frames)
    frames = load_frames_for_indices(path, indices)
    if not frames:
        return None
    video = torch.from_numpy(np.stack(frames, axis=0)).float() / 255.0
    if video.ndim == 4 and video.shape[-1] == 3:
        video = video.permute(0, 3, 1, 2)  # T, C, H, W
    video = torch.nn.functional.interpolate(
        video, size=(224, 224), mode="bilinear", align_corners=False
    )
    mean = torch.tensor([0.45, 0.45, 0.45], dtype=video.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], dtype=video.dtype).view(1, 3, 1, 1)
    video = (video - mean) / std
    video = video.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # B, C, T, H, W
    with torch.no_grad():
        feat = model(video).squeeze(0).cpu()
    return feat


def extract_fvd_feature(
    path: Path,
    total_frames: int,
    num_frames: int,
    backbone: str,
    model,
    preprocess,
    device,
):
    if backbone == "i3d":
        return extract_i3d_feature(path, total_frames, num_frames, model, device)
    return extract_r3d_feature(path, total_frames, num_frames, model, preprocess, device)


def compute_stats(features):
    feats = torch.stack(features).double()
    mu = feats.mean(dim=0)
    diff = feats - mu
    cov = diff.T @ diff / max(feats.shape[0] - 1, 1)
    return mu, cov


def sqrtm_psd(mat: torch.Tensor):
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=0)
    return (eigvecs * torch.sqrt(eigvals).unsqueeze(0)) @ eigvecs.T


def frechet_distance(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    sqrt_cov1 = sqrtm_psd(cov1)
    cov_prod = sqrt_cov1 @ cov2 @ sqrt_cov1
    sqrt_cov_prod = sqrtm_psd(cov_prod)
    trace = torch.trace(cov1 + cov2 - 2.0 * sqrt_cov_prod)
    value = diff.dot(diff) + trace
    return float(max(value.item(), 0.0))


def format_float(value, digits=3):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def build_file_aliases(baseline_files, per_video_metrics):
    alias_map = {}
    ordered_aliases = []

    for idx, filename in enumerate(sorted(set(baseline_files)), start=1):
        alias = f"baseline{idx}"
        alias_map[filename] = alias
        ordered_aliases.append((alias, filename))

    lr_idx = 0
    test_idx = 0
    test_files = sorted(
        {Path(row["test_path"]).name for row in per_video_metrics},
        key=lambda name: (
            parse_rank(Path(name).stem) if parse_rank(Path(name).stem) is not None else 10**9,
            name,
        ),
    )
    for filename in test_files:
        lower = filename.lower()
        if "lr" in lower:
            lr_idx += 1
            alias = f"lr{lr_idx}"
        else:
            test_idx += 1
            alias = f"test{test_idx}"
        alias_map[filename] = alias
        ordered_aliases.append((alias, filename))

    return alias_map, ordered_aliases


def write_summary_md(path, summary, per_video_metrics, fvd_results, pairwise_psnr):
    baseline_files = [Path(p).name for p in summary.get("baseline_videos", [])]
    alias_map, ordered_aliases = build_file_aliases(baseline_files, per_video_metrics)
    baseline_file_order = sorted(set(baseline_files))
    baseline_alias_order = [alias_map.get(name, name) for name in baseline_file_order]
    psnr_threshold = summary.get("psnr_threshold_db", 20.0)
    fvd_threshold = summary.get("fvd_degradation_threshold_pct", 10.0)
    fvd_backbone = summary.get("fvd_backbone", "i3d")
    total = len(per_video_metrics)
    psnr_pass = sum(1 for r in per_video_metrics if r.get("psnr_pass"))
    prompts_with_baseline_split = {
        normalize_prompt_slug(row.get("prompt_slug"))
        for row in fvd_results
        if row.get("baseline_split_fvd") is not None
    }

    def fmt_pct(value):
        if value is None:
            return "N/A"
        return f"{value:.2f}%"

    lines = []
    lines.append("# Metrics Summary (q_sample)")
    lines.append("")
    lines.append("## Explanations")
    lines.append("- **MSE**: Mean Squared Error between baseline and test frames (lower is better).")
    lines.append(
        f"- **PSNR**: Peak Signal-to-Noise Ratio (dB) computed from MSE (higher is better). "
        f"Pass threshold: **> {psnr_threshold:.1f} dB**."
    )
    lines.append(
        "- **CLIP image similarity**: Cosine similarity between CLIP image embeddings of baseline vs test frames (higher is better)."
    )
    lines.append(
        "- **CLIP text similarity**: Cosine similarity between CLIP text embedding (prompt) and CLIP image embeddings of test frames (higher is better)."
    )
    lines.append(
        f"- **FVD** ({fvd_backbone}): Fréchet Video Distance between feature distributions of baseline vs test sets (lower is better). "
        f"Grouped by prompt; degradation threshold: **< {fvd_threshold:.1f}%**."
    )
    lines.append("")
    lines.append("## Summary")
    if baseline_files:
        aliases = [alias_map.get(name, name) for name in sorted(set(baseline_files))]
        lines.append(
            f"- **Baselines ({len(baseline_files)})**: "
            + ", ".join(f"`{alias}`" for alias in aliases)
        )
    else:
        lines.append("- **Baselines**: N/A")
    lines.append(f"- **PSNR pass rate**: **{psnr_pass} / {total}**")
    if not prompts_with_baseline_split:
        lines.append("- **FVD degradation**: N/A (need >=2 baseline videos per prompt)")
    else:
        lines.append(
            f"- **FVD degradation**: available for {len(prompts_with_baseline_split)} prompt(s)"
        )
    lines.append("")
    lines.append("## File Legend")
    lines.append("")
    lines.append("| Alias | File |")
    lines.append("| --- | --- |")
    for alias, filename in ordered_aliases:
        lines.append(f"| `{alias}` | `{filename}` |")
    lines.append("")
    lines.append("## Per-Video Metrics")
    lines.append("")
    lines.append("| Video | Baseline | Rank | MSE | PSNR (dB) | PSNR Pass | CLIP Img | CLIP Text |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in sorted(
        per_video_metrics, key=lambda r: (r.get("prompt_slug") or "", r.get("rank") or 0)
    ):
        filename = Path(row["test_path"]).name
        baseline_name = Path(row["baseline_path"]).name
        video_alias = alias_map.get(filename, filename)
        baseline_alias = alias_map.get(baseline_name, baseline_name)
        rank = row.get("rank", "N/A")
        mse = format_float(row.get("mse"), 3)
        psnr = format_float(row.get("psnr"), 3)
        psnr_pass = "✅" if row.get("psnr_pass") else "❌"
        clip_img = format_float(row.get("clip_image_similarity"), 4)
        clip_text = format_float(row.get("clip_text_similarity"), 4)
        lines.append(
            f"| `{video_alias}` | `{baseline_alias}` | {rank} | {mse} | {psnr} | {psnr_pass} | {clip_img} | {clip_text} |"
        )
    lines.append("")
    lines.append("## Per-Video PSNR vs Each Baseline")
    lines.append("")
    per_target = defaultdict(dict)
    for row in pairwise_psnr:
        if row.get("target_is_baseline"):
            continue
        target = row.get("target_path")
        baseline = row.get("baseline_path")
        if target and baseline:
            per_target[target][baseline] = row

    header_cols = ["Video", "Rank"] + [f"{alias} PSNR (dB)" for alias in baseline_alias_order]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")
    ordered_targets = sorted(
        {Path(row["test_path"]).name for row in per_video_metrics},
        key=lambda name: (
            parse_rank(Path(name).stem) if parse_rank(Path(name).stem) is not None else 10**9,
            name,
        ),
    )
    for target_name in ordered_targets:
        video_alias = alias_map.get(target_name, target_name)
        rank = parse_rank(Path(target_name).stem)
        rank_str = rank if rank is not None else "N/A"
        row_vals = [f"`{video_alias}`", str(rank_str)]
        for baseline_name in baseline_file_order:
            psnr = None
            if target_name in per_target and baseline_name in per_target[target_name]:
                psnr = per_target[target_name][baseline_name].get("psnr")
            row_vals.append(format_float(psnr, 3))
        lines.append("| " + " | ".join(row_vals) + " |")
    lines.append("")
    lines.append("## Pairwise PSNR (each baseline vs all other files)")
    lines.append("")
    lines.append("| Prompt | Baseline | Target | Target Type | Rank | MSE | PSNR (dB) | PSNR Pass |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in sorted(
        pairwise_psnr,
        key=lambda r: (
            r.get("prompt_slug") or "",
            r.get("baseline_path") or "",
            r.get("target_path") or "",
        ),
    ):
        target_type = "baseline" if row.get("target_is_baseline") else "test"
        rank = row.get("target_rank")
        rank_str = rank if rank is not None else "N/A"
        baseline_name = row.get("baseline_path", "N/A")
        target_name = row.get("target_path", "N/A")
        baseline_alias = alias_map.get(baseline_name, baseline_name)
        target_alias = alias_map.get(target_name, target_name)
        mse = format_float(row.get("mse"), 3)
        psnr = format_float(row.get("psnr"), 3)
        psnr_pass = "✅" if row.get("psnr_pass") else "❌"
        lines.append(
            f"| {row.get('prompt_slug','N/A')} | `{baseline_alias}` | "
            f"`{target_alias}` | {target_type} | {rank_str} | {mse} | {psnr} | {psnr_pass} |"
        )
    lines.append("")
    lines.append("## Prompt PSNR Summary")
    lines.append("")
    lines.append("| Prompt | PSNR Pass | Pass/Total |")
    lines.append("| --- | --- | --- |")
    prompt_groups = defaultdict(list)
    for row in per_video_metrics:
        prompt_slug = normalize_prompt_slug(row.get("prompt_slug")) or "unknown"
        prompt_groups[prompt_slug].append(row)
    for prompt_slug in sorted(prompt_groups.keys()):
        rows = prompt_groups[prompt_slug]
        pass_count = sum(1 for r in rows if r.get("psnr_pass"))
        total_count = len(rows)
        all_fail = pass_count == 0
        status = "❌" if all_fail else "✅"
        lines.append(
            f"| {prompt_slug} | {status} | {pass_count}/{total_count} |"
        )
    lines.append("")
    lines.append("## FVD Results (per prompt + rank)")
    lines.append("")
    lines.append("| Prompt | Rank | FVD | Degradation | Pass | Videos | Baselines |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in sorted(
        fvd_results, key=lambda r: (r.get("prompt_slug") or "", r.get("group") or "")
    ):
        fvd = format_float(row.get("fvd"), 4)
        degradation = fmt_pct(row.get("degradation_pct"))
        status = row.get("pass")
        pass_str = "N/A" if status is None else ("✅" if status else "❌")
        lines.append(
            f"| {row.get('prompt_slug','N/A')} | {row.get('group','N/A')} | {fvd} | {degradation} | {pass_str} | {row.get('video_count', 0)} | {row.get('baseline_count', 0)} |"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        repo_root = Path(__file__).resolve().parent
        input_dir = repo_root / "ffn_delta_outputs" / "q_sample"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    videos = sorted(input_dir.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files found under {input_dir}")

    prompts_file = None
    if args.prompts_file:
        prompts_file = Path(args.prompts_file)
    else:
        candidate = Path("test_prompts.json")
        if candidate.exists():
            prompts_file = candidate
        else:
            candidate = Path(__file__).resolve().parent / "test_prompts.json"
            if candidate.exists():
                prompts_file = candidate
    prompts_by_seed, prompts_by_id = load_prompts(prompts_file) if prompts_file else ({}, {})

    baseline_candidates = defaultdict(list)
    test_videos = []
    for path in videos:
        key = parse_video_key(path.stem)
        if is_baseline(path):
            baseline_candidates[key].append(path)
        else:
            test_videos.append(path)

    baseline_map = {}
    for key, candidates in baseline_candidates.items():
        baseline = choose_baseline(candidates)
        if baseline is not None:
            baseline_map[key] = baseline

    if not baseline_map:
        raise SystemExit("No baseline videos found (filenames containing 'baseline').")

    baseline_all = sorted(
        {p for candidates in baseline_candidates.values() for p in candidates}
    )
    baseline_by_prompt = defaultdict(list)
    for key, candidates in baseline_candidates.items():
        prompt_slug = normalize_prompt_slug(key[2])
        for path in candidates:
            baseline_by_prompt[prompt_slug].append(path)
    test_by_prompt = defaultdict(list)
    for path in test_videos:
        key = parse_video_key(path.stem)
        prompt_slug = normalize_prompt_slug(key[2])
        test_by_prompt[prompt_slug].append(path)

    if args.max_videos:
        test_videos = test_videos[: args.max_videos]

    clip_model = None
    clip_processor = None
    text_feature_cache = {}
    if not args.skip_clip:
        from transformers import CLIPModel, CLIPProcessor

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval().to(device)
        clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", use_fast=False
        )

    per_video_metrics = []
    for test_path in test_videos:
        key = parse_video_key(test_path.stem)
        baseline_path = baseline_map.get(key)
        if baseline_path is None:
            continue
        mse, psnr, frames = compute_mse_psnr(baseline_path, test_path)
        prompt_text = None
        if key[0] and key[0] in prompts_by_seed:
            prompt_text = prompts_by_seed[key[0]]
        elif key[2] and key[2] in prompts_by_id:
            prompt_text = prompts_by_id[key[2]]
        clip_sim = None
        clip_text_sim = None
        if not args.skip_clip and frames > 0:
            clip_sim = compute_clip_image_similarity(
                baseline_path,
                test_path,
                frames,
                args.clip_frames,
                device,
                clip_model,
                clip_processor,
            )
            if prompt_text:
                clip_text_sim = compute_clip_text_similarity(
                    test_path,
                    prompt_text,
                    frames,
                    args.clip_frames,
                    device,
                    clip_model,
                    clip_processor,
                    text_feature_cache,
                )
        psnr_pass = psnr is not None and psnr > args.psnr_threshold
        per_video_metrics.append(
            {
                "test_path": test_path.name,
                "baseline_path": baseline_path.name,
                "rank": parse_rank(test_path.stem),
                "seed": key[0],
                "frames": key[1],
                "prompt_slug": key[2],
                "prompt_text": prompt_text,
                "frame_count": frames,
                "mse": mse,
                "psnr": psnr,
                "clip_image_similarity": clip_sim,
                "clip_text_similarity": clip_text_sim,
                "psnr_pass": psnr_pass,
            }
        )

    pairwise_psnr = []
    for prompt_slug, baseline_paths in baseline_by_prompt.items():
        unique_baselines = sorted(set(baseline_paths))
        targets = sorted(set(unique_baselines + test_by_prompt.get(prompt_slug, [])))
        for baseline_path in unique_baselines:
            for target_path in targets:
                if target_path == baseline_path:
                    continue
                mse, psnr, frames = compute_mse_psnr(baseline_path, target_path)
                pairwise_psnr.append(
                    {
                        "prompt_slug": prompt_slug,
                        "baseline_path": baseline_path.name,
                        "target_path": target_path.name,
                        "target_is_baseline": is_baseline(target_path),
                        "target_rank": parse_rank(target_path.stem),
                        "frame_count": frames,
                        "mse": mse,
                        "psnr": psnr,
                        "psnr_pass": psnr is not None and psnr > args.psnr_threshold,
                    }
                )

    fvd_results = []
    baseline_split_fvd_by_prompt = {}
    if not args.skip_fvd:
        if args.fvd_backbone == "i3d":
            model = load_i3d_model(device)
            preprocess = None
        else:
            model, preprocess = load_r3d_model(device)
        baseline_features_by_prompt = {}
        for prompt_slug, baseline_paths in baseline_by_prompt.items():
            unique_paths = sorted(set(baseline_paths))
            features = []
            for baseline_path in unique_paths:
                frames = count_frames(baseline_path)
                feat = extract_fvd_feature(
                    baseline_path,
                    frames,
                    args.fvd_frames,
                    args.fvd_backbone,
                    model,
                    preprocess,
                    device,
                )
                if feat is not None:
                    features.append(feat)
            baseline_features_by_prompt[prompt_slug] = features
            if len(features) >= 2:
                indices = list(range(len(features)))
                random.Random(0).shuffle(indices)
                mid = len(indices) // 2
                left = [features[i] for i in indices[:mid]]
                right = [features[i] for i in indices[mid:]]
                if left and right:
                    mu_l, cov_l = compute_stats(left)
                    mu_r, cov_r = compute_stats(right)
                    baseline_split_fvd_by_prompt[prompt_slug] = frechet_distance(
                        mu_l, cov_l, mu_r, cov_r
                    )

        test_groups = defaultdict(list)
        frame_count_map = {
            item["test_path"]: item["frame_count"] for item in per_video_metrics
        }
        for item in per_video_metrics:
            test_path_name = str(item["test_path"])
            rank = item["rank"]
            prompt_slug = normalize_prompt_slug(item["prompt_slug"])
            key = (prompt_slug, f"r{rank}" if rank is not None else "rank_unknown")
            test_groups[key].append(test_path_name)

        if baseline_features_by_prompt and test_groups:
            for (prompt_slug, group_name), paths in sorted(test_groups.items()):
                baseline_features = baseline_features_by_prompt.get(prompt_slug, [])
                if not baseline_features:
                    continue
                group_features = []
                for path_name in paths:
                    path = input_dir / path_name
                    key = parse_video_key(Path(path_name).stem)
                    baseline_path = baseline_map.get(key)
                    if baseline_path is None:
                        continue
                    frames = frame_count_map.get(path_name)
                    if frames is None:
                        frames = count_frames(path)
                    feat = extract_fvd_feature(
                        path,
                        frames,
                        args.fvd_frames,
                        args.fvd_backbone,
                        model,
                        preprocess,
                        device,
                    )
                    if feat is not None:
                        group_features.append(feat)
                if not group_features:
                    continue
                mu_base, cov_base = compute_stats(baseline_features)
                mu_group, cov_group = compute_stats(group_features)
                fvd_value = frechet_distance(mu_base, cov_base, mu_group, cov_group)
                baseline_split_fvd = baseline_split_fvd_by_prompt.get(prompt_slug)
                degradation_pct = None
                fvd_pass = None
                if baseline_split_fvd is not None and baseline_split_fvd > 0:
                    degradation_pct = (
                        100.0 * (fvd_value - baseline_split_fvd) / baseline_split_fvd
                    )
                    fvd_pass = degradation_pct < args.fvd_degradation_threshold
                fvd_results.append(
                    {
                        "prompt_slug": prompt_slug,
                        "group": group_name,
                        "fvd": fvd_value,
                        "baseline_split_fvd": baseline_split_fvd,
                        "degradation_pct": degradation_pct,
                        "pass": fvd_pass,
                        "video_count": len(group_features),
                        "baseline_count": len(baseline_features),
                    }
                )

    summary = {
        "input_dir": input_dir.name,
        "baseline_videos": [p.name for p in baseline_all],
        "fvd_backbone": args.fvd_backbone,
        "psnr_threshold_db": args.psnr_threshold,
        "fvd_degradation_threshold_pct": args.fvd_degradation_threshold,
        "per_video_metrics": per_video_metrics,
        "pairwise_psnr": pairwise_psnr,
        "fvd_results": fvd_results,
    }

    json_path = output_dir / "metrics_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = output_dir / "metrics_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_path",
                "baseline_path",
                "rank",
                "seed",
                "frames",
                "prompt_slug",
                "prompt_text",
                "frame_count",
                "mse",
                "psnr",
                "clip_image_similarity",
                "clip_text_similarity",
                "psnr_pass",
            ],
        )
        writer.writeheader()
        for row in per_video_metrics:
            writer.writerow(row)

    pairwise_csv_path = output_dir / "metrics_pairwise_psnr.csv"
    with open(pairwise_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_slug",
                "baseline_path",
                "target_path",
                "target_is_baseline",
                "target_rank",
                "frame_count",
                "mse",
                "psnr",
                "psnr_pass",
            ],
        )
        writer.writeheader()
        for row in pairwise_psnr:
            writer.writerow(row)

    fvd_csv_path = output_dir / "metrics_results_fvd.csv"
    with open(fvd_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_slug",
                "group",
                "fvd",
                "baseline_split_fvd",
                "degradation_pct",
                "pass",
                "video_count",
                "baseline_count",
            ],
        )
        writer.writeheader()
        for row in fvd_results:
            writer.writerow(row)

    summary_md_path = output_dir / "metrics_summary.md"
    write_summary_md(
        summary_md_path,
        summary,
        per_video_metrics,
        fvd_results,
        pairwise_psnr,
    )

    total = len(per_video_metrics)
    psnr_pass = sum(1 for r in per_video_metrics if r["psnr_pass"])
    pairwise_total = len(pairwise_psnr)
    pairwise_pass = sum(1 for r in pairwise_psnr if r["psnr_pass"])
    print(f"Videos evaluated: {total}")
    print(f"PSNR pass: {psnr_pass}/{total} (threshold {args.psnr_threshold} dB)")
    print(
        "Pairwise baseline-vs-other PSNR pass: "
        f"{pairwise_pass}/{pairwise_total} (threshold {args.psnr_threshold} dB)"
    )
    if args.skip_fvd:
        print("FVD: skipped")
    else:
        print(f"FVD backbone: {args.fvd_backbone}")
        if not fvd_results:
            print("FVD: no results (check baseline/test sets)")
        for result in fvd_results:
            if result["degradation_pct"] is None:
                status = "N/A"
                deg_str = "N/A"
            else:
                status = "PASS" if result["pass"] else "FAIL"
                deg_str = f"{result['degradation_pct']:.2f}%"
            print(
                f"FVD {result.get('prompt_slug','N/A')} {result['group']}: "
                f"{result['fvd']:.4f} (degradation {deg_str}, {status})"
            )
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {pairwise_csv_path}")
    print(f"Saved: {fvd_csv_path}")
    print(f"Saved: {summary_md_path}")


if __name__ == "__main__":
    main()
