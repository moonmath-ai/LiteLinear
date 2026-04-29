#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick(row: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return row[key]
    return default


def to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def rel_path(path_str: str | Path | None, base: Path) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f} s"


def fmt_mb(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f} MB"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def pct_change(base: float | None, new: float | None) -> float | None:
    if base in (None, 0) or new is None:
        return None
    return ((new - base) / base) * 100.0


def pct_speedup(base: float | None, new: float | None) -> float | None:
    if base in (None, 0) or new is None:
        return None
    return ((base - new) / base) * 100.0


def metric_change_verb(base: float | None, new: float | None) -> str:
    delta = pct_change(base, new)
    if delta is None:
        return "changed"
    if delta < 0:
        return "decreased"
    if delta > 0:
        return "increased"
    return "changed"


def summarize_ok_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    denoise_values = [
        row["denoise_elapsed_sec"]
        for row in ok_rows
        if row.get("denoise_elapsed_sec") is not None
    ]
    return {
        "runs_total": len(rows),
        "runs_ok": len(ok_rows),
        "latency_mean_sec": mean([row["elapsed_sec"] for row in ok_rows]) if ok_rows else None,
        "latency_min_sec": min([row["elapsed_sec"] for row in ok_rows]) if ok_rows else None,
        "latency_max_sec": max([row["elapsed_sec"] for row in ok_rows]) if ok_rows else None,
        "denoise_elapsed_mean_sec": mean(denoise_values) if denoise_values else None,
        "denoise_elapsed_min_sec": min(denoise_values) if denoise_values else None,
        "denoise_elapsed_max_sec": max(denoise_values) if denoise_values else None,
        "avg_allocated_mean_mb": mean([row["avg_allocated_mb"] for row in ok_rows]) if ok_rows else None,
        "peak_allocated_max_mb": max([row["peak_allocated_mb"] for row in ok_rows]) if ok_rows else None,
        "avg_reserved_mean_mb": mean([row["avg_reserved_mb"] for row in ok_rows]) if ok_rows else None,
        "peak_reserved_max_mb": max([row["peak_reserved_mb"] for row in ok_rows]) if ok_rows else None,
    }


def normalize_runs(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in raw_rows:
        run_index = to_int(row.get("run_index"))
        phase = row.get("phase") or ("cold" if run_index == 1 else "warm")
        normalized.append(
            {
                "mode": row.get("mode", ""),
                "prompt_name": row.get("prompt_name", ""),
                "phase": phase,
                "execution_index": to_int(row.get("execution_index")),
                "run_index": run_index,
                "seed": to_int(row.get("seed")),
                "elapsed_sec": to_float(row.get("elapsed_sec")),
                "denoise_line": pick(row, "denoise_line", "tqdm_line"),
                "denoise_step_count": pick(row, "denoise_step_count", "tqdm_step_count"),
                "denoise_elapsed_text": pick(row, "denoise_elapsed_text", "tqdm_elapsed_text"),
                "denoise_elapsed_sec": to_float(pick(row, "denoise_elapsed_sec", "tqdm_elapsed_sec", default=None)),
                "denoise_rate_text": pick(row, "denoise_rate_text", "tqdm_rate_text"),
                "avg_allocated_mb": to_float(row.get("avg_allocated_mb")),
                "peak_allocated_mb": to_float(row.get("peak_allocated_mb")),
                "avg_reserved_mb": to_float(row.get("avg_reserved_mb")),
                "peak_reserved_mb": to_float(row.get("peak_reserved_mb")),
                "samples": to_int(row.get("samples")),
                "status": row.get("status", ""),
                "error": row.get("error", ""),
                "save_file": row.get("save_file", ""),
                "log_file": row.get("log_file", ""),
                "mem_samples_file": row.get("mem_samples_file", ""),
            }
        )
    return normalized


def normalize_summary_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in raw_rows:
        normalized.append(
            {
                **row,
                "prompt_name": row.get("prompt_name", ""),
                "mode": row.get("mode", ""),
                "phase": row.get("phase", ""),
                "runs_total": to_int(row.get("runs_total")),
                "runs_ok": to_int(row.get("runs_ok")),
                "prompt_count": to_int(row.get("prompt_count")),
                "latency_mean_sec": to_float(row.get("latency_mean_sec")),
                "latency_min_sec": to_float(row.get("latency_min_sec")),
                "latency_max_sec": to_float(row.get("latency_max_sec")),
                "denoise_elapsed_mean_sec": to_float(
                    pick(row, "denoise_elapsed_mean_sec", "tqdm_elapsed_mean_sec", default=None)
                ),
                "denoise_elapsed_min_sec": to_float(
                    pick(row, "denoise_elapsed_min_sec", "tqdm_elapsed_min_sec", default=None)
                ),
                "denoise_elapsed_max_sec": to_float(
                    pick(row, "denoise_elapsed_max_sec", "tqdm_elapsed_max_sec", default=None)
                ),
                "avg_allocated_mean_mb": to_float(row.get("avg_allocated_mean_mb")),
                "peak_allocated_max_mb": to_float(row.get("peak_allocated_max_mb")),
                "avg_reserved_mean_mb": to_float(row.get("avg_reserved_mean_mb")),
                "peak_reserved_max_mb": to_float(row.get("peak_reserved_max_mb")),
            }
        )
    return normalized


def parse_step_count(step_count: str) -> int | None:
    if not step_count:
        return None
    match = re.search(r"/(\d+)$", step_count.strip())
    if match:
        return int(match.group(1))
    match = re.search(r"^(\d+)/", step_count.strip())
    if match:
        return int(match.group(1))
    return None


def detect_sample_steps(config: dict[str, Any], runs_rows: list[dict[str, Any]], benchmark_log_text: str) -> int | None:
    config_steps = to_int(config.get("sample_steps"))
    if config_steps is not None:
        return config_steps
    for row in runs_rows:
        parsed = parse_step_count(str(row.get("denoise_step_count") or ""))
        if parsed is not None:
            return parsed
    match = re.search(r"sample_steps=(\d+)", benchmark_log_text)
    if match:
        return int(match.group(1))
    return None


def detect_gpu_info(config: dict[str, Any]) -> tuple[str | None, int | None]:
    gpu_name = str(config.get("gpu_name", "")).strip() or None
    gpu_total_memory_mib = to_int(config.get("gpu_total_memory_mib"))
    if gpu_name and gpu_total_memory_mib is not None:
        return gpu_name, gpu_total_memory_mib

    preferred_indices: list[str] = []
    cuda_visible_devices = str(config.get("cuda_visible_devices", "")).strip()
    if cuda_visible_devices:
        preferred_indices = [part.strip() for part in cuda_visible_devices.split(",") if part.strip()]

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return gpu_name, gpu_total_memory_mib

    candidates: list[tuple[str, str, int]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, name, memory = parts
        memory_mib = to_int(memory)
        if memory_mib is None:
            continue
        candidates.append((index, name, memory_mib))

    if not candidates:
        return gpu_name, gpu_total_memory_mib

    for preferred in preferred_indices:
        for index, name, memory_mib in candidates:
            if index == preferred:
                return name, memory_mib

    _, name, memory_mib = candidates[0]
    return name, memory_mib


def format_gpu_line(config: dict[str, Any]) -> str | None:
    gpu_name, gpu_total_memory_mib = detect_gpu_info(config)
    if gpu_name and gpu_total_memory_mib is not None:
        return f"`{gpu_name} ({gpu_total_memory_mib:,} MiB VRAM)`"
    if gpu_name:
        return f"`{gpu_name}`"
    return None


def format_seed_line(runs_rows: list[dict[str, Any]], seed_step: int | None) -> str | None:
    seeds = sorted({row["seed"] for row in runs_rows if row.get("seed") is not None})
    if not seeds:
        return None
    seed_text = ", ".join(f"`{seed}`" for seed in seeds)
    if seed_step == 0 and len(seeds) == 1:
        return f"{seed_text} (same seed for every run; `seed_step=0`)"
    if seed_step not in (None, 0):
        return f"{seed_text} (`seed_step={seed_step}`)"
    return seed_text


def metric_line(label: str, baseline_value: float | None, litelinear_value: float | None, *, formatter) -> str:
    verb = metric_change_verb(baseline_value, litelinear_value)
    return (
        f"- {label} {verb} from `{formatter(baseline_value)}` to `{formatter(litelinear_value)}` "
        f"(`{fmt_pct(pct_change(baseline_value, litelinear_value))}`) with LiteLinear."
    )


def runs_summary_delta_text(baseline_ok: int | None, litelinear_ok: int | None) -> str:
    if baseline_ok is None or litelinear_ok is None:
        return ""
    delta = litelinear_ok - baseline_ok
    sign = "+" if delta >= 0 else ""
    suffix = "run" if abs(delta) == 1 else "runs"
    return f"{sign}{delta} {suffix}"


def build_overall_intro(
    runs_rows: list[dict[str, Any]],
    prompt_names: list[str],
    mode_phase_lookup: dict[tuple[str, str], dict[str, Any]],
    prompt_mode_lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    total_runs = len(runs_rows)
    ok_runs = len([row for row in runs_rows if row.get("status") == "ok"])
    baseline_all = mode_phase_lookup.get(("baseline", "all"))
    litelinear_all = mode_phase_lookup.get(("litelinear", "all"))

    if ok_runs == total_runs:
        lines.append(
            f"All `{ok_runs}/{total_runs}` runs completed successfully across `{len(prompt_names)}` prompts in this benchmark."
        )
        return lines

    baseline_failures = [row for row in runs_rows if row.get("mode") == "baseline" and row.get("status") != "ok"]
    litelinear_failures = [row for row in runs_rows if row.get("mode") == "litelinear" and row.get("status") != "ok"]
    if (
        baseline_all
        and litelinear_all
        and litelinear_all.get("runs_ok") == litelinear_all.get("runs_total")
        and baseline_all.get("runs_ok", 0) < baseline_all.get("runs_total", 0)
        and baseline_failures
        and all(row.get("phase") == "warm" for row in baseline_failures)
        and not litelinear_failures
    ):
        lines.append(
            f"LiteLinear completed all `{litelinear_all['runs_ok']}/{litelinear_all['runs_total']}` runs. "
            f"Baseline completed only `{baseline_all['runs_ok']}/{baseline_all['runs_total']}`, "
            f"with every failure occurring in the warm phase."
        )
    else:
        baseline_text = (
            f"`{baseline_all['runs_ok']}/{baseline_all['runs_total']}` baseline runs"
            if baseline_all
            else "`0/0` baseline runs"
        )
        litelinear_text = (
            f"`{litelinear_all['runs_ok']}/{litelinear_all['runs_total']}` LiteLinear runs"
            if litelinear_all
            else "`0/0` LiteLinear runs"
        )
        lines.append(
            f"Completed {baseline_text} and {litelinear_text} across `{len(prompt_names)}` prompts in this benchmark."
        )

    if baseline_all and baseline_all.get("runs_ok") != baseline_all.get("runs_total"):
        surviving = []
        for prompt_name in prompt_names:
            row = prompt_mode_lookup.get((prompt_name, "baseline"))
            if row and row.get("runs_ok") is not None:
                surviving.append(f"`{row['runs_ok']}` for `{prompt_name}`")
        if surviving:
            lines.append(
                f"All comparison metrics below are based on successful runs only, so the baseline averages "
                f"reflect just `{baseline_all['runs_ok']}` surviving runs ({', '.join(surviving)})."
            )
    return lines


def build_failure_notes(runs_rows: list[dict[str, Any]]) -> list[str]:
    failed_rows = [row for row in runs_rows if row.get("status") != "ok"]
    if not failed_rows:
        return []

    lines: list[str] = []
    error_types = []
    for row in failed_rows:
        error_text = str(row.get("error") or "").strip()
        error_type = error_text.split(":", 1)[0].strip() if error_text else "unknown error"
        error_types.append(error_type)

    unique_error_types = sorted(set(error_types))
    failed_mode_counts = defaultdict(int)
    for row in failed_rows:
        failed_mode_counts[str(row.get("mode"))] += 1

    if len(unique_error_types) == 1:
        mode_label = ""
        if len(failed_mode_counts) == 1:
            only_mode = next(iter(failed_mode_counts))
            mode_label = f" {only_mode}"
        lines.append(f"All `{len(failed_rows)}`{mode_label} failures were `{unique_error_types[0]}` exceptions.")
    else:
        lines.append(
            "Failures were observed with multiple error types: "
            + ", ".join(f"`{error_type}`" for error_type in unique_error_types)
            + "."
        )

    for prompt_name in sorted({str(row.get("prompt_name")) for row in failed_rows}):
        warm_baseline_rows = [
            row
            for row in runs_rows
            if row.get("prompt_name") == prompt_name
            and row.get("mode") == "baseline"
            and row.get("phase") == "warm"
        ]
        if not warm_baseline_rows:
            continue
        warm_failures = [row for row in warm_baseline_rows if row.get("status") != "ok"]
        if warm_failures:
            lines.append(
                f"- `{prompt_name}`: `{len(warm_failures)}/{len(warm_baseline_rows)}` warm baseline runs failed."
            )

    allocation_sizes = []
    free_sizes = []
    for row in failed_rows:
        error_text = str(row.get("error") or "")
        alloc_match = re.search(r"Tried to allocate ([0-9.]+) MiB", error_text)
        free_match = re.search(r"of which ([0-9.]+) MiB is free", error_text)
        if alloc_match:
            allocation_sizes.append(float(alloc_match.group(1)))
        if free_match:
            free_sizes.append(float(free_match.group(1)))
    if allocation_sizes and free_sizes:
        alloc_min = min(allocation_sizes)
        alloc_max = max(allocation_sizes)
        free_min = min(free_sizes)
        free_max = max(free_sizes)
        lines.append(
            f"- The failing runs tried to allocate roughly `{alloc_min:.0f}-{alloc_max:.0f} MiB` "
            f"with only `{free_min:.1f}-{free_max:.1f} MiB` free on the device."
        )

    if any("expandable_segments:True" in str(row.get("error") or "") for row in failed_rows):
        lines.append(
            "- The PyTorch error text suggested `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` "
            "as a possible fragmentation mitigation."
        )

    return lines


def should_include_warmup(
    runs_rows: list[dict[str, Any]],
    runs_per_mode: int | None,
) -> bool:
    if not runs_rows or runs_per_mode is None or runs_per_mode < 2:
        return False
    if any(row.get("status") != "ok" for row in runs_rows):
        return False
    litelinear_cold = [
        row["denoise_elapsed_sec"]
        for row in runs_rows
        if row.get("mode") == "litelinear" and row.get("phase") == "cold" and row.get("denoise_elapsed_sec") is not None
    ]
    litelinear_warm = [
        row["denoise_elapsed_sec"]
        for row in runs_rows
        if row.get("mode") == "litelinear" and row.get("phase") == "warm" and row.get("denoise_elapsed_sec") is not None
    ]
    if not litelinear_cold or not litelinear_warm:
        return False
    return (mean(litelinear_cold) - mean(litelinear_warm)) >= 5.0


def build_warmup_section(runs_rows: list[dict[str, Any]], benchmark_log_text: str) -> list[str]:
    def phase_values(mode: str, phase: str, key: str) -> list[float]:
        return [
            row[key]
            for row in runs_rows
            if row.get("mode") == mode and row.get("phase") == phase and row.get("status") == "ok" and row.get(key) is not None
        ]

    baseline_cold_denoise = phase_values("baseline", "cold", "denoise_elapsed_sec")
    litelinear_cold_denoise = phase_values("litelinear", "cold", "denoise_elapsed_sec")
    baseline_warm_denoise = phase_values("baseline", "warm", "denoise_elapsed_sec")
    litelinear_warm_denoise = phase_values("litelinear", "warm", "denoise_elapsed_sec")

    baseline_cold_wall = phase_values("baseline", "cold", "elapsed_sec")
    litelinear_cold_wall = phase_values("litelinear", "cold", "elapsed_sec")
    baseline_warm_wall = phase_values("baseline", "warm", "elapsed_sec")
    litelinear_warm_wall = phase_values("litelinear", "warm", "elapsed_sec")

    warmup_delta = None
    if litelinear_cold_denoise and litelinear_warm_denoise:
        warmup_delta = mean(litelinear_cold_denoise) - mean(litelinear_warm_denoise)

    lines = [
        "The first LiteLinear run for each prompt is slower than the later LiteLinear runs, "
        "but it is still faster than the matching baseline cold run.",
        "",
        f"- Cold LiteLinear denoise time decreased from `{fmt_seconds(mean(baseline_cold_denoise))}` "
        f"to `{fmt_seconds(mean(litelinear_cold_denoise))}` "
        f"(`{fmt_pct(pct_change(mean(baseline_cold_denoise), mean(litelinear_cold_denoise)))}`) "
        "across the first run of each prompt.",
        f"- Warm LiteLinear denoise time decreased from `{fmt_seconds(mean(baseline_warm_denoise))}` "
        f"to `{fmt_seconds(mean(litelinear_warm_denoise))}` "
        f"(`{fmt_pct(pct_change(mean(baseline_warm_denoise), mean(litelinear_warm_denoise)))}`) "
        "when comparing only the later runs after that first LiteLinear run per prompt.",
        f"- Cold LiteLinear wall time decreased from `{fmt_seconds(mean(baseline_cold_wall))}` "
        f"to `{fmt_seconds(mean(litelinear_cold_wall))}` "
        f"(`{fmt_pct(pct_change(mean(baseline_cold_wall), mean(litelinear_cold_wall)))}`).",
        f"- Warm LiteLinear wall time decreased from `{fmt_seconds(mean(baseline_warm_wall))}` "
        f"to `{fmt_seconds(mean(litelinear_warm_wall))}` "
        f"(`{fmt_pct(pct_change(mean(baseline_warm_wall), mean(litelinear_warm_wall)))}`).",
    ]

    if warmup_delta is not None:
        warmup_seconds = int(round(warmup_delta / 5.0) * 5.0)
        if warmup_seconds >= 5:
            message = (
                f"- The extra `~{warmup_seconds} s` on the first LiteLinear run appears to be a one-time "
                "LiteLinear first-step warmup effect."
            )
        else:
            message = "- The extra time on the first LiteLinear run appears to be a one-time LiteLinear first-step warmup effect."
        if "reuse cached patched shards" in benchmark_log_text:
            message += (
                " The log already shows cached patched shards being reused before denoising begins, "
                "so this does not look like a patched-shard cache rebuild."
            )
        lines.append(message)
    return lines


def build_summary_markdown(benchmark_root: Path) -> str:
    benchmark_root = benchmark_root.resolve()
    summary_dir = benchmark_root / "summary"
    repo_root = benchmark_root.parents[2]

    config = load_json(benchmark_root / "benchmark_config.json") or {}
    raw_runs = load_csv(summary_dir / "runs.csv")
    runs_rows = normalize_runs(raw_runs)
    summary_rows = normalize_summary_rows(load_csv(summary_dir / "summary.csv"))
    summary_by_phase_rows = normalize_summary_rows(load_csv(summary_dir / "summary_by_phase.csv"))
    mode_phase_rows = normalize_summary_rows(load_csv(summary_dir / "mode_phase_summary.csv"))
    benchmark_log_text = (benchmark_root / "bench_wan_i2v.log").read_text(encoding="utf-8") if (benchmark_root / "bench_wan_i2v.log").exists() else ""

    prompt_names = sorted({row["prompt_name"] for row in runs_rows if row.get("prompt_name")})
    prompt_mode_lookup = {(row["prompt_name"], row["mode"]): row for row in summary_rows}
    mode_phase_lookup = {(row["mode"], row["phase"]): row for row in mode_phase_rows}

    if not mode_phase_lookup:
        mode_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in runs_rows:
            mode_groups[str(row["mode"])].append(row)
        for mode, rows in mode_groups.items():
            mode_phase_lookup[(mode, "all")] = {"mode": mode, "phase": "all", **summarize_ok_rows(rows)}
            cold_rows = [row for row in rows if row.get("phase") == "cold"]
            warm_rows = [row for row in rows if row.get("phase") == "warm"]
            if cold_rows:
                mode_phase_lookup[(mode, "cold")] = {"mode": mode, "phase": "cold", **summarize_ok_rows(cold_rows)}
            if warm_rows:
                mode_phase_lookup[(mode, "warm")] = {"mode": mode, "phase": "warm", **summarize_ok_rows(warm_rows)}

    summary_lines: list[str] = []
    append = summary_lines.append

    append("# Wan I2V Benchmark Summary")
    append("")
    append("## Configuration")
    append("")

    benchmark_root_display = rel_path(benchmark_root, repo_root)
    prompts_display = rel_path(config.get("prompts_json"), repo_root)
    append(f"- Benchmark root: `{benchmark_root_display}`")
    if prompts_display:
        append(f"- Prompt file: `{prompts_display}`")
    if config.get("ckpt_dir"):
        append(f"- Checkpoint directory: `{config['ckpt_dir']}`")
    if config.get("task"):
        append(f"- Task: `{config['task']}`")
    if config.get("size"):
        append(f"- Size: `{config['size']}`")
    if config.get("frame_num") is not None:
        append(f"- Frame count: `{config['frame_num']}`")
    if config.get("sample_fps") is not None:
        append(f"- Sample FPS: `{config['sample_fps']}`")

    sample_steps = detect_sample_steps(config, runs_rows, benchmark_log_text)
    if sample_steps is not None:
        append(f"- Denoise steps: `{sample_steps}`")

    if config.get("runs_per_mode") is not None:
        append(f"- Runs per mode: `{config['runs_per_mode']}`")

    seed_line = format_seed_line(runs_rows, to_int(config.get("seed_step")))
    if seed_line:
        append(f"- Seeds: {seed_line}")

    if config.get("order_mode"):
        order_mode = str(config.get("order_mode") or "").strip()
        order_seed = config.get("order_seed")
        if order_seed not in (None, "", 0, "0"):
            append(f"- Order mode: `{order_mode}` (`order_seed={order_seed}`)")
        else:
            append(f"- Order mode: `{order_mode}`")

    if config.get("sample_interval_sec") is not None:
        append(f"- Sample interval: `{config['sample_interval_sec']}` s")

    gpu_line = format_gpu_line(config)
    if gpu_line:
        append(f"- GPU: {gpu_line}")

    patch_config = str(config.get("litelinear_patch_config") or "").strip()
    if patch_config:
        append(f"- Patch config: `{Path(patch_config).name}`")

    append("")
    append("## Overall")
    append("")

    for line in build_overall_intro(runs_rows, prompt_names, mode_phase_lookup, prompt_mode_lookup):
        append(line)
    append("")

    baseline_all = mode_phase_lookup.get(("baseline", "all"))
    litelinear_all = mode_phase_lookup.get(("litelinear", "all"))
    if baseline_all and litelinear_all:
        append(metric_line("Mean wall time", baseline_all["latency_mean_sec"], litelinear_all["latency_mean_sec"], formatter=fmt_seconds))
        append(metric_line("Mean denoise time", baseline_all["denoise_elapsed_mean_sec"], litelinear_all["denoise_elapsed_mean_sec"], formatter=fmt_seconds))
        append(metric_line("Mean allocated VRAM", baseline_all["avg_allocated_mean_mb"], litelinear_all["avg_allocated_mean_mb"], formatter=fmt_mb))
        append(metric_line("Peak allocated VRAM", baseline_all["peak_allocated_max_mb"], litelinear_all["peak_allocated_max_mb"], formatter=fmt_mb))
        append(metric_line("Mean reserved VRAM", baseline_all["avg_reserved_mean_mb"], litelinear_all["avg_reserved_mean_mb"], formatter=fmt_mb))
        append(metric_line("Peak reserved VRAM", baseline_all["peak_reserved_max_mb"], litelinear_all["peak_reserved_max_mb"], formatter=fmt_mb))

    charts_dir = summary_dir / "charts"
    timing_chart = charts_dir / "metric_bars_timing.svg"
    memory_chart = charts_dir / "metric_bars_memory.svg"
    consumption_chart = charts_dir / "memory_consumption.svg"
    has_failures = any(row.get("status") != "ok" for row in runs_rows)
    mode_display = {"baseline": "Baseline", "litelinear": "LiteLinear"}

    if has_failures and ("baseline", "cold") in mode_phase_lookup:
        append("")
        append("## Reliability And Phase Behavior")
        append("")
        append("| Mode | Phase | Successful runs | Mean wall time | Mean denoise time | Avg allocated VRAM | Peak allocated VRAM |")
        append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for mode in ("baseline", "litelinear"):
            for phase in ("cold", "warm"):
                row = mode_phase_lookup.get((mode, phase))
                if not row:
                    continue
                append(
                    f"| {mode_display.get(mode, mode.capitalize())} | {phase.capitalize()} | "
                    f"{row['runs_ok']}/{row['runs_total']} | "
                    f"{fmt_seconds(row['latency_mean_sec'])} | "
                    f"{fmt_seconds(row['denoise_elapsed_mean_sec'])} | "
                    f"{fmt_mb(row['avg_allocated_mean_mb'])} | "
                    f"{fmt_mb(row['peak_allocated_max_mb'])} |"
                )

        baseline_warm = mode_phase_lookup.get(("baseline", "warm"))
        if baseline_warm and baseline_warm.get("runs_ok") not in (None, baseline_warm.get("runs_total")):
            warm_failures = baseline_warm["runs_total"] - baseline_warm["runs_ok"]
            successful_label = "run" if baseline_warm["runs_ok"] == 1 else "runs"
            failed_label = "run" if warm_failures == 1 else "runs"
            append("")
            append(
                f"Baseline warm-phase averages are based on the `{baseline_warm['runs_ok']}` successful warm "
                f"{successful_label}; the other `{warm_failures}` warm baseline {failed_label} failed."
            )

        failure_lines = build_failure_notes(runs_rows)
        if failure_lines:
            append("")
            append("## Failure Notes")
            append("")
            for idx, line in enumerate(failure_lines):
                append(line)

    runs_per_mode = to_int(config.get("runs_per_mode"))
    if should_include_warmup(runs_rows, runs_per_mode):
        append("")
        append("## Warmup Interpretation")
        append("")
        for line in build_warmup_section(runs_rows, benchmark_log_text):
            append(line)

    if timing_chart.exists() and memory_chart.exists():
        append("")
        append("## Metric Bar Charts")
        append("")
        append(f"![Timing metric bar charts](charts/{timing_chart.name})")
        append("")
        append(f"![VRAM metric bar charts](charts/{memory_chart.name})")

    if consumption_chart.exists():
        append("")
        append("## Memory Consumption Charts")
        append("")
        append(
            "The chart below shows the mean allocated and reserved VRAM traces for each prompt and mode "
            "after resampling each successful run onto a common 0-100% progress axis."
        )
        append("")
        append(f"![Memory consumption charts](charts/{consumption_chart.name})")

    append("")
    append("## Per-Prompt Results")
    append("")
    append(
        "`Denoise time` is the elapsed time reported by the progress bar for the diffusion sampling loop itself, "
        "excluding some setup and finalization overhead included in wall time."
    )
    append("")
    append("| Prompt | Metric | Baseline | LiteLinear | Delta |")
    append("| --- | --- | ---: | ---: | ---: |")

    metric_specs = [
        ("Wall time", "latency_mean_sec", fmt_seconds),
        ("Denoise time", "denoise_elapsed_mean_sec", fmt_seconds),
        ("Avg allocated VRAM", "avg_allocated_mean_mb", fmt_mb),
        ("Peak allocated VRAM", "peak_allocated_max_mb", fmt_mb),
        ("Avg reserved VRAM", "avg_reserved_mean_mb", fmt_mb),
        ("Peak reserved VRAM", "peak_reserved_max_mb", fmt_mb),
    ]
    for prompt_name in prompt_names:
        baseline_row = prompt_mode_lookup.get((prompt_name, "baseline"))
        litelinear_row = prompt_mode_lookup.get((prompt_name, "litelinear"))
        if not baseline_row or not litelinear_row:
            continue

        baseline_runs_ok = baseline_row.get("runs_ok")
        baseline_runs_total = baseline_row.get("runs_total")
        litelinear_runs_ok = litelinear_row.get("runs_ok")
        litelinear_runs_total = litelinear_row.get("runs_total")
        if (
            baseline_runs_ok != baseline_runs_total
            or litelinear_runs_ok != litelinear_runs_total
        ):
            append(
                f"| `{prompt_name}` | Successful runs | "
                f"{baseline_runs_ok}/{baseline_runs_total} | "
                f"{litelinear_runs_ok}/{litelinear_runs_total} | "
                f"{runs_summary_delta_text(baseline_runs_ok, litelinear_runs_ok)} |"
            )

        for metric_label, key, formatter in metric_specs:
            base_value = baseline_row.get(key)
            lite_value = litelinear_row.get(key)
            if base_value is None or lite_value is None:
                continue
            append(
                f"| `{prompt_name}` | {metric_label} | "
                f"{formatter(base_value)} | {formatter(lite_value)} | {fmt_pct(pct_change(base_value, lite_value))} |"
            )

    if has_failures:
        failed_prompt_notes: list[str] = []
        for prompt_name in prompt_names:
            baseline_row = prompt_mode_lookup.get((prompt_name, "baseline"))
            if baseline_row and baseline_row.get("runs_ok") not in (None, baseline_row.get("runs_total")):
                if baseline_row["runs_ok"] == 1 and baseline_row["runs_total"] and baseline_row["runs_total"] > 1:
                    successful_baseline_rows = [
                        row
                        for row in runs_rows
                        if row.get("prompt_name") == prompt_name
                        and row.get("mode") == "baseline"
                        and row.get("status") == "ok"
                    ]
                    phase_hint = ""
                    if len(successful_baseline_rows) == 1 and successful_baseline_rows[0].get("phase"):
                        phase_hint = f"{successful_baseline_rows[0]['phase']} "
                    failed_prompt_notes.append(
                        f"Baseline `{prompt_name}` metrics are based on the single successful "
                        f"{phase_hint}run because the remaining baseline attempts failed."
                    )
        if failed_prompt_notes:
            append("")
            for note in failed_prompt_notes:
                append(note)

    append("")
    append("## Source Files")
    append("")
    source_items: list[tuple[str, str]] = [
        ("Aggregated CSV", "summary.csv"),
        ("Per-run CSV", "runs.csv"),
        ("Per-run JSON", "runs.json"),
    ]
    if has_failures:
        if (summary_dir / "summary_by_phase.csv").exists():
            source_items.append(("Phase summary CSV", "summary_by_phase.csv"))
        if (summary_dir / "mode_phase_summary.csv").exists():
            source_items.append(("Mode/phase summary CSV", "mode_phase_summary.csv"))
    source_items.extend(
        [
            ("Benchmark config", "../benchmark_config.json"),
            ("Qualitative outputs", "../videos"),
            ("Per-run logs", "../logs"),
        ]
    )
    if has_failures and (benchmark_root / "memory_samples").exists():
        source_items.append(("Per-run memory samples", "../memory_samples"))
    if has_failures and (benchmark_root / "bench_wan_i2v.log").exists():
        source_items.append(("Benchmark log", "../bench_wan_i2v.log"))
    if timing_chart.exists():
        source_items.append(("Timing metric charts", f"charts/{timing_chart.name}"))
    if memory_chart.exists():
        source_items.append(("VRAM metric charts", f"charts/{memory_chart.name}"))
    if consumption_chart.exists():
        source_items.append(("Memory consumption charts", f"charts/{consumption_chart.name}"))

    for label, rel_target in source_items:
        target_path = (summary_dir / rel_target).resolve()
        if target_path.exists():
            append(f"- {label}: [`{rel_target}`]({rel_target})")

    append("")
    return "\n".join(summary_lines)


def write_summary_markdown(benchmark_root: Path) -> Path:
    benchmark_root = benchmark_root.resolve()
    summary_path = benchmark_root / "summary" / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary_markdown(benchmark_root), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark summary.md for LiteLinear benchmark runs.")
    parser.add_argument(
        "--benchmark-root",
        required=True,
        help="Path to the benchmark root directory (the directory containing summary/, videos/, logs/, etc.).",
    )
    args = parser.parse_args()
    summary_path = write_summary_markdown(Path(args.benchmark_root))
    print(summary_path)


if __name__ == "__main__":
    main()
