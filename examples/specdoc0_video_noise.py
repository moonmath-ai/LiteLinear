#!/usr/bin/env python3
"""
SpecDoc0 helper: compare baseline (nn.Linear) vs LiteLinear rendered videos.

Computes frame-wise noise metrics and writes:
  - per-frame CSV
  - summary text
  - growth chart (SVG)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


def _slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=np.float64)
    x0 = x - x.mean()
    y0 = y.astype(np.float64) - y.mean()
    den = np.dot(x0, x0)
    if den <= 0:
        return 0.0
    return float(np.dot(x0, y0) / den)


def _luma(x: np.ndarray) -> np.ndarray:
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def compute_metrics(base_video: Path, lite_video: Path) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
    base = iio.imread(base_video).astype(np.float32)
    lite = iio.imread(lite_video).astype(np.float32)
    if base.ndim != 4 or lite.ndim != 4:
        raise ValueError("Expected 4D arrays [T,H,W,C] for both videos.")
    t = min(base.shape[0], lite.shape[0])
    base = base[:t]
    lite = lite[:t]

    d = lite - base
    mse = np.mean(d**2, axis=(1, 2, 3))
    mae = np.mean(np.abs(d), axis=(1, 2, 3))
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(255.0) - 10 * np.log10(np.maximum(mse, 1e-12))

    signal_rms = np.sqrt(np.mean(base**2, axis=(1, 2, 3)))
    err_to_signal = rmse / np.maximum(signal_rms, 1e-6)

    la = _luma(base)
    lb = _luma(lite)
    mean_a = la.mean(axis=(1, 2))
    mean_b = lb.mean(axis=(1, 2))
    std_a = la.std(axis=(1, 2))
    std_b = lb.std(axis=(1, 2))

    summary = {
        "frames": float(t),
        "mse_mean": float(np.mean(mse)),
        "mse_min": float(np.min(mse)),
        "mse_max": float(np.max(mse)),
        "mae_mean": float(np.mean(mae)),
        "mae_min": float(np.min(mae)),
        "mae_max": float(np.max(mae)),
        "psnr_mean_db": float(np.mean(psnr)),
        "psnr_min_db": float(np.min(psnr)),
        "psnr_max_db": float(np.max(psnr)),
        "err_to_signal_mean": float(np.mean(err_to_signal)),
        "err_to_signal_max": float(np.max(err_to_signal)),
        "mse_slope_per_frame": _slope(mse),
        "mae_slope_per_frame": _slope(mae),
        "err_to_signal_slope_per_frame": _slope(err_to_signal),
        "global_luma_mean_baseline": float(la.mean()),
        "global_luma_mean_lite": float(lb.mean()),
        "global_luma_mean_abs_delta": float(abs(la.mean() - lb.mean())),
        "global_luma_std_baseline": float(la.std()),
        "global_luma_std_lite": float(lb.std()),
        "global_luma_std_abs_delta": float(abs(la.std() - lb.std())),
        "frame_mean_luma_mae": float(np.mean(np.abs(mean_a - mean_b))),
        "frame_std_luma_mae": float(np.mean(np.abs(std_a - std_b))),
        "black_frames_baseline_mean_lt_5": float(np.sum(mean_a < 5.0)),
        "black_frames_lite_mean_lt_5": float(np.sum(mean_b < 5.0)),
    }

    series = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "err_to_signal": err_to_signal,
    }
    return summary, series


def write_outputs(
    out_dir: Path,
    stem: str,
    summary: Dict[str, float],
    series: Dict[str, np.ndarray],
    title: str,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    txt_path = out_dir / f"{stem}_summary.txt"
    svg_path = out_dir / f"{stem}_growth.svg"

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "mse", "rmse", "mae", "psnr_db", "err_to_signal"])
        for i in range(len(series["mse"])):
            w.writerow(
                [
                    i,
                    float(series["mse"][i]),
                    float(series["rmse"][i]),
                    float(series["mae"][i]),
                    float(series["psnr"][i]),
                    float(series["err_to_signal"][i]),
                ]
            )

    with txt_path.open("w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    x = np.arange(len(series["mse"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.6), dpi=140, sharex=True)
    ax1.plot(x, series["mse"], label="MSE", linewidth=1.7)
    ax1.plot(x, series["mae"], label="MAE", linewidth=1.7)
    ax1.set_ylabel("Pixel error (0..255)")
    ax1.set_title("Frame-wise output difference")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right")

    ax2.plot(x, series["psnr"], label="PSNR (dB)", linewidth=1.7)
    ax2.plot(x, series["err_to_signal"], label="Error/Signal RMS", linewidth=1.7)
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Quality metrics")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(svg_path)

    return csv_path, txt_path, svg_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare baseline vs LiteLinear videos.")
    ap.add_argument(
        "--baseline-video",
        type=Path,
        default=Path("/root/users/vh/LTX-2/outputs/2026-02-15/boxing-cats_78_0_72.mp4"),
    )
    ap.add_argument(
        "--lite-video",
        type=Path,
        default=Path("/root/users/vh/LTX-2/outputs/2026-02-15/boxing-cats_78_1_72.mp4"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/assets"),
        help="Output directory for CSV/TXT/SVG artifacts.",
    )
    ap.add_argument(
        "--stem",
        type=str,
        default="litelinear_video_noise_2026-02-15",
        help="Output stem for generated files.",
    )
    args = ap.parse_args()

    summary, series = compute_metrics(args.baseline_video, args.lite_video)
    title = f"Video pair: {args.baseline_video.name} vs {args.lite_video.name}"
    csv_path, txt_path, svg_path = write_outputs(args.out_dir, args.stem, summary, series, title)

    print(f"[noise] wrote csv: {csv_path}")
    print(f"[noise] wrote summary: {txt_path}")
    print(f"[noise] wrote chart: {svg_path}")
    print("[noise] key metrics:")
    for k in (
        "frames",
        "mse_mean",
        "mae_mean",
        "psnr_mean_db",
        "err_to_signal_mean",
        "mse_slope_per_frame",
        "mae_slope_per_frame",
        "err_to_signal_slope_per_frame",
        "black_frames_baseline_mean_lt_5",
        "black_frames_lite_mean_lt_5",
    ):
        print(f"  {k}: {summary[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
