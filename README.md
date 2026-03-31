# LiteLinear

**LiteLinear** is a specialized PyTorch module designed to replace standard `nn.Linear` layers in Feed-Forward Networks (FFN), specifically targeting LTX-Video contexts. It implements a decomposition strategy:

$$
W \approx A \cdot B + Q, \quad Q \to \text{FP8}
$$

Where the computation is performed using a custom fused CUDA kernel:

$$
y = (x B^T) A^T + \text{scale} \cdot (x Q_{\text{fp8}}^T) + \text{bias}
$$

This approach allows for significant memory savings and potential speedups by utilizing low-rank approximations and FP8 arithmetic for the residual.


## LTX2 LiteLinear vs Baseline (FA3 Self-Attn, No-Calib)

### Timing Overview

<div align="center">
<table><tr>
<td align="center"><img src="docs/assets/ltx2_transformer_compact.svg" alt="LTX2 Transformer(Audio+Video) compact bar" height="320" style="vertical-align:middle"/></td>
<td align="center"><img src="docs/assets/ltx2_e2e_stacked_compact.svg" alt="LTX2 E2E stacked compact bar" height="320" style="vertical-align:middle"/></td>
</tr></table>

| Group | Transformer Mean, s | Min, s | Max, s | Std, s | Transformer % Faster | Decode Mean, s | Save, s | E2E Total, s | E2E % Faster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 4.520 | 4.460 | 4.650 | 0.070 | 0.00% | 3.710 | 5.100 | 13.330 | 0.00% |
| litelinear | 3.500 | 3.490 | 3.520 | 0.010 | 22.57% | 3.710 | 5.100 | 12.310 | 7.65% |

</div>

### First-Run Compile Effect

<div align="center" style="max-width: 420px; margin-left: auto; margin-right: auto">

<img src="docs/assets/ltx2_coldstart_compile_compact.svg" alt="LTX2 cold-start compile compact bar" style="max-width: 100%; height: auto"/>

| Group | First-run transformer, s | % Lower vs Baseline |
| --- | ---: | ---: |
| baseline | 335.430 | 0.00% |
| litelinear (mean r32/r64/r512) | 38.433 | 88.54% |

</div>

### Memory comparison

It focuses on `warmup+bench` memory behavior, where LiteLinear shows lower allocated VRAM:

- Peak allocated: `65,633.66 MB` (baseline) vs `58,058.06 MB` (LiteLinear)
- Average allocated: `58,858.35 MB` (baseline) vs `43,476.85 MB` (LiteLinear)

![LiteLinear memory comparison](docs/assets/litelinear_memory_comparison_warmup_bench.svg)

## Required Metrics

- **MSE**: Mean Squared Error between baseline and test frames (lower is better).
- **PSNR**: Peak Signal-to-Noise Ratio (dB) from MSE (higher is better). Pass: **> 20.0 dB**.
- **CLIP image similarity**: Cosine similarity, baseline vs test frames (higher is better).
- **CLIP text similarity**: Cosine similarity, prompt vs test frames (higher is better).
- **FVD** (i3d): Fréchet Video Distance, baseline vs test sets (lower is better). Per prompt; degradation threshold: **< 10.0%**.


### LTX2 metrics (q_sample, i3d)


#### PSNR, Per Prompt, Group Means, dB

<div align="center">

| Prompt | baseline1 vs baseline2..10 | baseline1 vs r32 group | baseline1 vs r64 group | baseline1 vs r512 group |
| --- | ---: | ---: | ---: | ---: |
| a-dramatic-underwater-scene-featuring-a-person-s | 41.747 | 19.823 | 19.822 | 20.413 |
| a-man-in-a-sleek-modern-jetpack-flying-upwards-t | 43.306 | 20.872 | 21.517 | 21.163 |
| a-serene-view-of-the-banks-of-the-rhine-river-sh | 38.664 | 20.277 | 20.208 | 19.735 |
| a-single-water-droplet-falls-from-a-height-movin | 28.724 | 27.827 | 27.375 | 30.680 |
| two-anthropomorphic-cats-boxing-in-a-well-lit-ar | ~identical (MSE=0) | 18.646 | 20.924 | 21.188 |

</div>

**Prompt PSNR summary**

| Prompt | PSNR Pass | Pass/Total |
| --- | --- | --- |
| a-dramatic-underwater-scene-featuring-a-person-s | ✅ | 10/30 |
| a-man-in-a-sleek-modern-jetpack-flying-upwards-t | ✅ | 30/30 |
| a-serene-view-of-the-banks-of-the-rhine-river-sh | ✅ | 20/30 |
| a-single-water-droplet-falls-from-a-height-movin | ✅ | 30/30 |
| two-anthropomorphic-cats-boxing-in-a-well-lit-ar | ✅ | 20/30 |

#### CLIP Similarity (Per Prompt, Rank Group Means)

| Prompt | CLIP Image r32 | CLIP Text r32 | CLIP Image r64 | CLIP Text r64 | CLIP Image r512 | CLIP Text r512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a-dramatic-underwater-scene-featuring-a-person-s | 0.9601 | 0.3058 | 0.9593 | 0.3037 | 0.9590 | 0.3116 |
| a-man-in-a-sleek-modern-jetpack-flying-upwards-t | 0.9636 | 0.3365 | 0.9634 | 0.3222 | 0.9620 | 0.3199 |
| a-serene-view-of-the-banks-of-the-rhine-river-sh | 0.9736 | 0.2656 | 0.9726 | 0.2739 | 0.9633 | 0.2703 |
| a-single-water-droplet-falls-from-a-height-movin | 0.9629 | 0.3597 | 0.9635 | 0.3544 | 0.9780 | 0.3683 |
| two-anthropomorphic-cats-boxing-in-a-well-lit-ar | 0.9739 | 0.3634 | 0.9738 | 0.3669 | 0.9798 | 0.3662 |


**FVD (per prompt + rank)**

| Prompt | Rank | FVD | Degradation | Pass | Videos | Baselines |
| --- | --- | --- | --- | --- | --- | --- |
| a-dramatic-underwater-scene-featuring-a-person-s | r32 | 8.8313 | 2922.27% | ❌ | 10 | 10 |
| a-dramatic-underwater-scene-featuring-a-person-s | r512 | 8.1199 | 2678.84% | ❌ | 10 | 10 |
| a-dramatic-underwater-scene-featuring-a-person-s | r64 | 18.2055 | 6130.36% | ❌ | 10 | 10 |
| a-man-in-a-sleek-modern-jetpack-flying-upwards-t | r32 | 16.8284 | 9496.61% | ❌ | 10 | 10 |
| a-man-in-a-sleek-modern-jetpack-flying-upwards-t | r512 | 21.2910 | 12041.49% | ❌ | 10 | 10 |
| a-man-in-a-sleek-modern-jetpack-flying-upwards-t | r64 | 21.7506 | 12303.55% | ❌ | 10 | 10 |
| a-serene-view-of-the-banks-of-the-rhine-river-sh | r32 | 1.2744 | 1459.17% | ❌ | 10 | 10 |
| a-serene-view-of-the-banks-of-the-rhine-river-sh | r512 | 2.5507 | 3020.73% | ❌ | 10 | 10 |
| a-serene-view-of-the-banks-of-the-rhine-river-sh | r64 | 1.9026 | 2227.73% | ❌ | 10 | 10 |
| a-single-water-droplet-falls-from-a-height-movin | r32 | 20.4085 | 588.22% | ❌ | 10 | 10 |
| a-single-water-droplet-falls-from-a-height-movin | r512 | 24.1211 | 713.41% | ❌ | 10 | 10 |
| a-single-water-droplet-falls-from-a-height-movin | r64 | 30.4487 | 926.79% | ❌ | 10 | 10 |
| two-anthropomorphic-cats-boxing-in-a-well-lit-ar | r32 | 9.7384 | N/A | N/A | 10 | 10 |
| two-anthropomorphic-cats-boxing-in-a-well-lit-ar | r512 | 18.6098 | N/A | N/A | 10 | 10 |
| two-anthropomorphic-cats-boxing-in-a-well-lit-ar | r64 | 15.0996 | N/A | N/A | 10 | 10 |

#### Video samples (baseline vs LiteLinear r32 / r64 / r512)

Click a thumbnail to play the video.

| Baseline | LiteLinear r32 | LiteLinear r64 | LiteLinear r512 |
| --- | --- | --- | --- |
| [![p1](docs/assets/thumbs/ltx-2-19b-distilled_p001_h6482cab9_s486307_f72_baseline_i01_a-single-water-droplet-falls-from-a-height-movin.jpg)](docs/assets/ltx-2-19b-distilled_p001_h6482cab9_s486307_f72_baseline_i01_a-single-water-droplet-falls-from-a-height-movin.mp4) | [![p1](docs/assets/thumbs/ltx-2-19b-distilled_p001_h6482cab9_s486307_f72_liteffn_r32_nocalib_i01_a-single-water-droplet-falls-from-a-height-movin.jpg)](docs/assets/ltx-2-19b-distilled_p001_h6482cab9_s486307_f72_liteffn_r32_nocalib_i01_a-single-water-droplet-falls-from-a-height-movin.mp4) | [![p1](docs/assets/thumbs/ltx-2-19b-distilled_p001_h6482cab9_s486307_f72_liteffn_r64_nocalib_i01_a-single-water-droplet-falls-from-a-height-movin.jpg)](docs/assets/ltx-2-19b-distilled_p001_h6482cab9_s486307_f72_liteffn_r64_nocalib_i01_a-single-water-droplet-falls-from-a-height-movin.mp4) | [![p1](docs/assets/thumbs/ltx-2-19b-distilled_p001_h6482cab9_s486307_f72_liteffn_r512_nocalib_i01_a-single-water-droplet-falls-from-a-height-movin.jpg)](docs/assets/ltx-2-19b-distilled_p001_h6482cab9_s486307_f72_liteffn_r512_nocalib_i01_a-single-water-droplet-falls-from-a-height-movin.mp4) |
| [![p2](docs/assets/thumbs/ltx-2-19b-distilled_p002_h757f9ac3_s789012_f72_baseline_i01_a-man-in-a-sleek-modern-jetpack-flying-upwards-t.jpg)](docs/assets/ltx-2-19b-distilled_p002_h757f9ac3_s789012_f72_baseline_i01_a-man-in-a-sleek-modern-jetpack-flying-upwards-t.mp4) | [![p2](docs/assets/thumbs/ltx-2-19b-distilled_p002_h757f9ac3_s789012_f72_liteffn_r32_nocalib_i01_a-man-in-a-sleek-modern-jetpack-flying-upwards-t.jpg)](docs/assets/ltx-2-19b-distilled_p002_h757f9ac3_s789012_f72_liteffn_r32_nocalib_i01_a-man-in-a-sleek-modern-jetpack-flying-upwards-t.mp4) | [![p2](docs/assets/thumbs/ltx-2-19b-distilled_p002_h757f9ac3_s789012_f72_liteffn_r64_nocalib_i01_a-man-in-a-sleek-modern-jetpack-flying-upwards-t.jpg)](docs/assets/ltx-2-19b-distilled_p002_h757f9ac3_s789012_f72_liteffn_r64_nocalib_i01_a-man-in-a-sleek-modern-jetpack-flying-upwards-t.mp4) | [![p2](docs/assets/thumbs/ltx-2-19b-distilled_p002_h757f9ac3_s789012_f72_liteffn_r512_nocalib_i01_a-man-in-a-sleek-modern-jetpack-flying-upwards-t.jpg)](docs/assets/ltx-2-19b-distilled_p002_h757f9ac3_s789012_f72_liteffn_r512_nocalib_i01_a-man-in-a-sleek-modern-jetpack-flying-upwards-t.mp4) |
| [![p3](docs/assets/thumbs/ltx-2-19b-distilled_p003_h0c667058_s650048_f72_baseline_i01_two-anthropomorphic-cats-boxing-in-a-well-lit-ar.jpg)](docs/assets/ltx-2-19b-distilled_p003_h0c667058_s650048_f72_baseline_i01_two-anthropomorphic-cats-boxing-in-a-well-lit-ar.mp4) | [![p3](docs/assets/thumbs/ltx-2-19b-distilled_p003_h0c667058_s650048_f72_liteffn_r32_nocalib_i01_two-anthropomorphic-cats-boxing-in-a-well-lit-ar.jpg)](docs/assets/ltx-2-19b-distilled_p003_h0c667058_s650048_f72_liteffn_r32_nocalib_i01_two-anthropomorphic-cats-boxing-in-a-well-lit-ar.mp4) | [![p3](docs/assets/thumbs/ltx-2-19b-distilled_p003_h0c667058_s650048_f72_liteffn_r64_nocalib_i01_two-anthropomorphic-cats-boxing-in-a-well-lit-ar.jpg)](docs/assets/ltx-2-19b-distilled_p003_h0c667058_s650048_f72_liteffn_r64_nocalib_i01_two-anthropomorphic-cats-boxing-in-a-well-lit-ar.mp4) | [![p3](docs/assets/thumbs/ltx-2-19b-distilled_p003_h0c667058_s650048_f72_liteffn_r512_nocalib_i01_two-anthropomorphic-cats-boxing-in-a-well-lit-ar.jpg)](docs/assets/ltx-2-19b-distilled_p003_h0c667058_s650048_f72_liteffn_r512_nocalib_i01_two-anthropomorphic-cats-boxing-in-a-well-lit-ar.mp4) |
| [![p4](docs/assets/thumbs/ltx-2-19b-distilled_p004_h97e8e2d0_s960015_f72_baseline_i01_a-serene-view-of-the-banks-of-the-rhine-river-sh.jpg)](docs/assets/ltx-2-19b-distilled_p004_h97e8e2d0_s960015_f72_baseline_i01_a-serene-view-of-the-banks-of-the-rhine-river-sh.mp4) | [![p4](docs/assets/thumbs/ltx-2-19b-distilled_p004_h97e8e2d0_s960015_f72_liteffn_r32_nocalib_i01_a-serene-view-of-the-banks-of-the-rhine-river-sh.jpg)](docs/assets/ltx-2-19b-distilled_p004_h97e8e2d0_s960015_f72_liteffn_r32_nocalib_i01_a-serene-view-of-the-banks-of-the-rhine-river-sh.mp4) | [![p4](docs/assets/thumbs/ltx-2-19b-distilled_p004_h97e8e2d0_s960015_f72_liteffn_r64_nocalib_i01_a-serene-view-of-the-banks-of-the-rhine-river-sh.jpg)](docs/assets/ltx-2-19b-distilled_p004_h97e8e2d0_s960015_f72_liteffn_r64_nocalib_i01_a-serene-view-of-the-banks-of-the-rhine-river-sh.mp4) | [![p4](docs/assets/thumbs/ltx-2-19b-distilled_p004_h97e8e2d0_s960015_f72_liteffn_r512_nocalib_i01_a-serene-view-of-the-banks-of-the-rhine-river-sh.jpg)](docs/assets/ltx-2-19b-distilled_p004_h97e8e2d0_s960015_f72_liteffn_r512_nocalib_i01_a-serene-view-of-the-banks-of-the-rhine-river-sh.mp4) |
| [![p5](docs/assets/thumbs/ltx-2-19b-distilled_p005_h4e2b1dcb_s536857_f72_baseline_i01_a-dramatic-underwater-scene-featuring-a-person-s.jpg)](docs/assets/ltx-2-19b-distilled_p005_h4e2b1dcb_s536857_f72_baseline_i01_a-dramatic-underwater-scene-featuring-a-person-s.mp4) | [![p5](docs/assets/thumbs/ltx-2-19b-distilled_p005_h4e2b1dcb_s536857_f72_liteffn_r32_nocalib_i01_a-dramatic-underwater-scene-featuring-a-person-s.jpg)](docs/assets/ltx-2-19b-distilled_p005_h4e2b1dcb_s536857_f72_liteffn_r32_nocalib_i01_a-dramatic-underwater-scene-featuring-a-person-s.mp4) | [![p5](docs/assets/thumbs/ltx-2-19b-distilled_p005_h4e2b1dcb_s536857_f72_liteffn_r64_nocalib_i01_a-dramatic-underwater-scene-featuring-a-person-s.jpg)](docs/assets/ltx-2-19b-distilled_p005_h4e2b1dcb_s536857_f72_liteffn_r64_nocalib_i01_a-dramatic-underwater-scene-featuring-a-person-s.mp4) | [![p5](docs/assets/thumbs/ltx-2-19b-distilled_p005_h4e2b1dcb_s536857_f72_liteffn_r512_nocalib_i01_a-dramatic-underwater-scene-featuring-a-person-s.jpg)](docs/assets/ltx-2-19b-distilled_p005_h4e2b1dcb_s536857_f72_liteffn_r512_nocalib_i01_a-dramatic-underwater-scene-featuring-a-person-s.mp4) |

Full LTX2 summary (incl. video samples, detailed tables): [metrics_summary.md](docs/ltx2_metrics_2026-02-20_recalc_i3d/metrics_summary.md) · [metrics_detailed.md](docs/ltx2_metrics_2026-02-20_recalc_i3d/metrics_detailed.md)

### LTX-Video 0.9.8 LiteLinear vs Baseline (LiteAttention self, Calibrated)

#### Timing (Video only, compile mode - default, fullgraphs - false)

| Mode | Timesteps (s) | Run inference (s) | Enhance (s) | % faster (timesteps) | % faster (total) |
| --- | --- | --- | --- | --- | --- |
| LiteLinear | 6.80 | 8.61 | 0.72 | **7.86%** | **7.52%** |
| Baseline | 7.38 | 9.31 | 0.72 | — | — |

*Timesteps* = time inside the diffusion denoising loop (per-step transformer forward + scheduler, etc.). So it includes **non-FFN** work: attention, layernorms, embeddings, and scheduler/noise handling — only part of the step is FFN; the reported % faster is for the whole step.

- **Baselines (2)**: `baseline1`, `baseline2`
- **calibrated sample**: `lr1calib` - 25000 random prompts from [vidprom_filtered_extended.txt](https://huggingface.co/gdhe17/Self-Forcing/blob/5c6e33328c3f025cbc9c60ab31f377ef6a65cda1/vidprom_filtered_extended.txt)

#### Per-video metrics

| Video | Baseline | Rank | MSE | PSNR (dB) | PSNR Pass | CLIP Img | CLIP Text |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lr1calib` | `baseline1` | 64 | 428.790 | 21.808 | ✅ | 0.9930 | N/A |
| `lr2` | `baseline1` | 64 | 664.877 | 19.903 | ❌ | 0.9845 | N/A |
| `lr3` | `baseline1` | 64 | 516.288 | 21.002 | ✅ | 0.9861 | N/A |

#### Prompt PSNR summary

| Prompt | PSNR Pass | Pass/Total |
| --- | --- | --- |
| Acharminganimatedsceneofafluff | ✅ | 2/3 |

#### FVD (per prompt + rank)

| Prompt | Rank | FVD | Degradation | Pass | Videos | Baselines |
| --- | --- | --- | --- | --- | --- | --- |
| Acharminganimatedsceneofafluff | r64 | 50.9825 | 554.10% | ❌ | 3 | 2 |

#### Video samples

Click a thumbnail to play the video.

<table>
<tr>
<th align="center" width="20%">baseline1</th>
<th align="center" width="20%">baseline2</th>
<th align="center" width="20%">lr1calib</th>
<th align="center" width="20%">lr2</th>
<th align="center" width="20%">lr3</th>
</tr>
<tr>
<td align="center" width="20%"><a href="docs/ltx0.9.8_metrics/validate_s171198_f121_default_bf16_baseline_Acharminganimatedsceneofafluff.mp4"><img src="docs/ltx0.9.8_metrics/thumbs/validate_s171198_f121_default_bf16_baseline_Acharminganimatedsceneofafluff.jpg" width="150" alt="baseline1"></a></td>
<td align="center" width="20%"><a href="docs/ltx0.9.8_metrics/validate_s171198_f121_default_bf16_baseline_Acharminganimatedsceneofafluff_2.mp4"><img src="docs/ltx0.9.8_metrics/thumbs/validate_s171198_f121_default_bf16_baseline_Acharminganimatedsceneofafluff_2.jpg" width="150" alt="baseline2"></a></td>
<td align="center" width="20%"><a href="docs/ltx0.9.8_metrics/validate_r64_s171198_f121_default_cxfp8_lr-cu_Acharminganimatedsceneofafluff.calib.mp4"><img src="docs/ltx0.9.8_metrics/thumbs/validate_r64_s171198_f121_default_cxfp8_lr-cu_Acharminganimatedsceneofafluff.calib.jpg" width="150" alt="lr1calib"></a></td>
<td align="center" width="20%"><a href="docs/ltx0.9.8_metrics/validate_r64_s171198_f121_default_cxfp8_lr-cu_Acharminganimatedsceneofafluff.mp4"><img src="docs/ltx0.9.8_metrics/thumbs/validate_r64_s171198_f121_default_cxfp8_lr-cu_Acharminganimatedsceneofafluff.jpg" width="150" alt="lr2"></a></td>
<td align="center" width="20%"><a href="docs/ltx0.9.8_metrics/validate_r64_s171198_f121_default_cxfp8_lr-cu_noR_Acharminganimatedsceneofafluff.mp4"><img src="docs/ltx0.9.8_metrics/thumbs/validate_r64_s171198_f121_default_cxfp8_lr-cu_noR_Acharminganimatedsceneofafluff.jpg" width="150" alt="lr3"></a></td>
</tr>
</table>

Full LTX 0.9.8 summary: [metrics_summary.md](docs/ltx0.9.8_metrics/metrics_summary.md).

## Installation

Ensure you have a CUDA-compatible environment (CUDA 12.x recommended) and PyTorch installed.

Install the Python package (from this tree or a wheel; PyPI name is `lite-linear`):

```bash
python -m pip install -v . --no-build-isolation --no-deps --force-reinstall
```

Notes:

- Set `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` as needed for your CUDA version / GPU arch before building.
- If you hit `Error: setup script specifies an absolute path`, regenerate the manifest and retry:

```bash
rm -rf lite_linear.egg-info && python setup.py egg_info
```

## Usage

```python
import torch
from lite_linear import LiteLinear

# Standalone usage (manual materialization)
# - Requires the `lite_linear._cuda` extension (see Installation)
# - Requires CUDA weights/inputs
# - Note: `materialize_from_weight()` is silent; 
# logs only appear for auto-materialization (triggered by `.eval()` or first forward). 
# Cache files go to ffn_delta_outputs/lr_data/lite_linear_auto/.
linear = LiteLinear(in_features=1024, out_features=4096, rank=64, device="cuda", dtype=torch.bfloat16)

# Load/set weights BEFORE materialization
with torch.no_grad():
    linear.weight.copy_(torch.randn(4096, 1024, dtype=torch.bfloat16, device="cuda"))
    if linear.bias is not None:
        linear.bias.zero_()

# Decompose once (installs FP8+LR factors), then run
linear.materialize_from_weight()

# Forward pass (uses fused kernel if available)
x = torch.randn(16, 1024, dtype=torch.bfloat16, device="cuda")
y = linear(x)
```

## Features

- **Low-Rank Decomposition**: Decomposes weights into low-rank factors $A$ and $B$.
- **FP8 Quantization**: Quantizes the residual $Q$ to FP8 (E4M3FN) for efficiency.
- **Fused CUDA Kernel**: Optimized custom CUDA kernel for the fused operation.
- **Drop-in Replacement**: Can often replace `nn.Linear` in existing Transformers with minimal code changes.

## Tested hardware (for metrics below)

- GPU model: `NVIDIA H200`
- VRAM: `143771 MiB`
- Driver / CUDA: `590.48.01` / `13.1`
- Note: benchmark and noise snapshots in this README were collected on this device.

## Performance Benchmark on shapes captured from LTX-Video

Units:

- per-shape rows: `us`
- `TOTAL` row: `ms`

Column glossary:

- `Cfg`: FFN projection shape (`w1` = up-proj, `w2` = down-proj).
- `M`: flattened activation rows for that GEMM shape.
- `Count`: number of calls for that shape in the captured workload.
- `Lin`: baseline `nn.Linear` latency.
- `TE`: Transformer Engine linear latency.
- `PT`: LiteLinear PyTorch path latency.
- `CUDA`: LiteLinear CUDA path latency.
- `TE%` / `PT%` / `CUDA%`: relative improvement vs baseline linear (`+` is faster, `-` is slower).
- `TOTAL`: count-weighted aggregate across listed shapes.

```text
Cfg      M  Count |  Lin    TE    PT  CUDA |    TE%    PT%  CUDA%
------------------------------------------------------------------
w2    1400    336 |  392   260   298   186 | +33.7% +23.8% +52.5%
w1    1400    336 |  378   273   301   182 | +27.8% +20.3% +51.7%
w2    2450    336 |  597   437   501   317 | +26.8% +15.9% +46.9%
w1    2450    336 |  562   455   511   302 | +19.0%  +9.1% +46.2%
w2    5600    480 | 1213   994  1215   762 | +18.1%  -0.1% +37.2%
w1    5600    480 | 1141  1097  1198   712 |  +3.8%  -5.0% +37.6%
w2    9800    144 | 2008  1763  2074  1261 | +12.2%  -3.3% +37.2%
w1    9800    144 | 1946  1868  2042  1202 |  +4.0%  -4.9% +38.3%
w2   10850    336 | 2277  2110  2281  1395 |  +7.4%  -0.2% +38.8%
w1   10850    336 | 2035  2156  2215  1324 |  -5.9%  -8.9% +34.9%
w2   22400    144 | 4889  4506  4596  3018 |  +7.8%  +6.0% +38.3%
w1   22400    144 | 4714  4581  4471  2805 |  +2.8%  +5.1% +40.5%
w2   43400    144 | 9275  8285  8845  5772 | +10.7%  +4.6% +37.8%
w1   43400    144 | 8847  8460  8669  5328 |  +4.4%  +2.0% +39.8%
------------------------------------------------------------------
TOTAL        3840 | 7788  7158  7631  4744 |  +8.1%  +2.0% +39.1%
```

## Integration example: from [LTX-Video integration](https://github.com/moonmath-ai/LTX-Video/pull/14)

This repo integrates `LiteLinear` into LTX-Video transformer FFNs (w1 + w2), behind an env var:

```bash
# Enable (default)
export USE_LITE_LINEAR=1

# Disable (use standard nn.Linear)
export USE_LITE_LINEAR=0
```

Implementation lives in `../ltx_video/models/transformers/attention.py` (see also `../LiteLinear_integration.md`).

Example (from `FeedForward.__init__` in
[`attention.py#L1294-L1318`](../ltx_video/models/transformers/attention.py#L1294-L1318)):

```python
linear_cls = nn.Linear

# act_fn = GELU/GEGLU/ApproximateGELU(...)

if USE_LITE_LINEAR:
    # FFN w1: diffusers activations create an internal `nn.Linear` at `act_fn.proj`.
    # Replace it with LiteLinear (centralized helper; keeps state_dict keys stable).
    (linear_cls := LiteLinear).replace_activation_proj_(act_fn)

# FFN w2: it will use LiteLinear for the output projection too.
self.net.append(linear_cls(inner_dim, dim_out, bias=bias))
```

Runtime behavior:

- `model.eval()` triggers a one-time decomposition+cache/load pass for all `LiteLinear` instances (after weights are loaded and moved to CUDA).
- Factors are cached under `${LITELINEAR_CACHE}/lr_data/` (defaults to `HF_CACHE`).
- If neither env var is set, cache falls back to `<script_dir>/.cache/litelinear/lr_data/`.
- Cache filename defaults to `lite_linear_<fingerprint>_r<rank>_<calib|nocalib>.safetensors`.
- The calib tag is derived from safetensors metadata (`with_r=1` when R is baked into B/Q). If both caches exist, calibrated is preferred.
- A warning is emitted if filename tag disagrees with metadata (`with_r`).

Troubleshooting:

- `ImportError: lite_linear._cuda ...`: build/install the extension, or set `USE_LITE_LINEAR=0`.
- `LiteLinear requires CUDA weights`: move the model to CUDA before calling `model.eval()`.
- If cached factors mismatch a new checkpoint: delete the cache file under the path above to regenerate.

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support) (CUDA build; see `pyproject.toml` for the minimum version used here)
- NVIDIA GPU with Compute Capability 8.9+ (Ada Lovelace) or 9.0+ (Hopper) recommended for best FP8 performance, though the kernel is compiled for arch 9.0a by default (set `TORCH_CUDA_ARCH_LIST` to your GPU arch when building `lite_linear._cuda`)

## Additional docs

- `docs/extras.md`: R-calibration online/offline flow, checkpointing, resume, and merge fallback.
- `docs/kernel.md`: `kernel_v13.cu` behavior notes with cast/satfinite + autotune details.
- `extras/specdoc0_video_noise.py`: reproducible video noise-growth metrics and chart generation.
- `extras/specdoc0_castfinite_check.py`: castfinite behavior and fused-kernel NaN/Inf stress check.

## Obfuscated wheel workflow

For the wheel-only obfuscated distribution pipeline (Cython + PyArmor + CUDA binary-only wheel) and benchmark validation flow, see:

- `docs/obfuscated_build.md`
