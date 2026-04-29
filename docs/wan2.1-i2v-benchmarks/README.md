# Wan2.1 I2V Benchmarks

This page keeps the checked-in benchmark summary for the Wan2.1 image-to-video
LiteLinear integration. Raw benchmark runs are generated under
`LiteLinear/benchmarks/` and are intentionally local-only artifacts.

## Benchmark Sources

This page combines two local benchmark runs with different purposes:

- Runtime/reliability source: `LiteLinear/benchmarks/wan_i2v_20260424_080809`
  - Used for the timing, memory, and OOM reliability summary below.
  - Configuration: `49` frames, `20` denoise steps, `6` runs per mode.
- WorldJen quality source: `LiteLinear/benchmarks/wan_i2v_20260424_025712`
  - Used for the WorldJen video uploads and quality scores below.
  - Configuration: `81` frames, `40` denoise steps, `3` runs per mode.
  - All `12/12` generated videos completed locally; WorldJen scored `10/12`
    uploads in the captured UI snapshot, with `2` still evaluating/no-score.

Both runs used the same `surf-cat` and `angelic-clock` prompts with seed
`868276`, but their runtime and quality numbers should be read as separate
benchmark artifacts.

## Runtime And OOM Reference Run

- Source run: `LiteLinear/benchmarks/wan_i2v_20260424_080809`
- GPU: `NVIDIA H200 (143,771 MiB VRAM)`
- Task: `i2v-14B`
- Size: `1280*720`
- Frame count: `49`
- Denoise steps: `20`
- Prompts: `angelic-clock`, `surf-cat`
- Runs per mode: `6`
- Seed: `868276`

## Result Summary

LiteLinear completed all `12/12` runs. Baseline completed only `3/12` runs,
with every failure occurring during warm baseline runs.

All comparison metrics below use successful runs only, so the baseline averages
reflect the `3` surviving baseline runs.

| Metric | Baseline | LiteLinear | Delta |
| --- | ---: | ---: | ---: |
| Successful runs | 3/12 | 12/12 | +9 runs |
| Mean wall time | 311.90 s | 312.73 s | +0.26% |
| Mean denoise time | 229.00 s | 224.83 s | -1.82% |
| Mean allocated VRAM | 69,095.78 MB | 50,091.79 MB | -27.50% |
| Peak allocated VRAM | 100,901.70 MB | 79,297.12 MB | -21.41% |
| Mean reserved VRAM | 73,938.35 MB | 55,353.36 MB | -25.14% |
| Peak reserved VRAM | 110,228.00 MB | 88,448.00 MB | -19.76% |

## WorldJen Quality Evaluation

The WorldJen results for the same `surf-cat` and `angelic-clock` prompts are
checked in under `LiteLinear/docs/worldjen_results/`. The WorldJen UI did not expose original
filenames, so local filenames were recovered for the report. The report includes
LL-vs-baseline metric charts and per-video metric charts.

- Source run: `LiteLinear/benchmarks/wan_i2v_20260424_025712`
- Full report: [LiteLinear/docs/worldjen_results/worldjen_results.md](../worldjen_results/worldjen_results.md)

Average WorldJen scores by prompt:

| Prompt | Metric | Baseline | LiteLinear | Delta |
| --- | --- | ---: | ---: | ---: |
| `surf-cat` | subject consistency | 93.33% | 97.50% | +4.17 |
| `surf-cat` | scene consistency | 96.67% | 98.75% | +2.08 |
| `surf-cat` | motion smoothness | 75.00% | 98.75% | +23.75 |
| `surf-cat` | temporal flickering | 86.67% | 98.75% | +12.08 |
| `surf-cat` | physical mechanics | 84.17% | 96.25% | +12.08 |
| `surf-cat` | object permanence | 94.17% | 98.75% | +4.58 |
| `surf-cat` | human fidelity | 91.67% | 98.75% | +7.08 |
| `surf-cat` | dynamic degree | 56.67% | 87.50% | +30.83 |
| `surf-cat` | semantic adherence | 100.00% | 100.00% | +0.00 |
| `surf-cat` | spatial relationship | 94.17% | 100.00% | +5.83 |
| `surf-cat` | semantic drift | 100.00% | 100.00% | +0.00 |
| `surf-cat` | total | 88.41% | 97.73% | +9.32 |
| `angelic-clock` | subject consistency | 87.50% | 86.25% | -1.25 |
| `angelic-clock` | scene consistency | 88.33% | 83.75% | -4.58 |
| `angelic-clock` | motion smoothness | 86.67% | 67.50% | -19.17 |
| `angelic-clock` | temporal flickering | 90.00% | 70.00% | -20.00 |
| `angelic-clock` | physical mechanics | 85.83% | 65.00% | -20.83 |
| `angelic-clock` | object permanence | 88.33% | 80.00% | -8.33 |
| `angelic-clock` | human fidelity | 90.00% | 82.50% | -7.50 |
| `angelic-clock` | dynamic degree | 7.50% | 26.25% | +18.75 |
| `angelic-clock` | semantic adherence | 99.17% | 97.50% | -1.67 |
| `angelic-clock` | spatial relationship | 91.67% | 85.00% | -6.67 |
| `angelic-clock` | semantic drift | 97.50% | 92.50% | -5.00 |
| `angelic-clock` | total | 82.95% | 76.02% | -6.93 |

## Reliability Notes

All `9` baseline failures were `torch.OutOfMemoryError` exceptions.

- `angelic-clock`: `4/5` warm baseline runs failed.
- `surf-cat`: `5/5` warm baseline runs failed.
- The failing runs tried to allocate roughly `1008-1018 MiB` with only
  `16.5-238.5 MiB` free on the device.

## Reproducing

Run the benchmark harness from the Wan2.1 repository root:

```bash
WAN_BENCH_CKPT_DIR=/path/to/Wan2.1-I2V-14B-720P \
  ./LiteLinear/extras/bench_wan_i2v.sh
```

The runner writes generated outputs under `LiteLinear/benchmarks/wan_i2v_*`,
including videos, logs, memory samples, CSV/JSON summaries, and a generated
`summary/summary.md`.

To regenerate a markdown summary for a local run:

```bash
python LiteLinear/extras/generate_benchmark_summary.py \
  --benchmark-root LiteLinear/benchmarks/wan_i2v_<timestamp>
```

Only curated markdown from benchmark runs should be checked in. Raw generated
CSVs, JSON files, logs, memory samples, videos, and timestamped run directories
should stay local.
