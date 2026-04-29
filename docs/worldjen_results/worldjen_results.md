# WorldJen Results

Extracted from the provided WorldJen playground HTML snapshot.

- Evaluated videos: 10
- Skipped/no-score videos: 2 (527370 · litelinear__run_01-03_seed_868276.mp4, 298207 · litelinear__run_01-03_seed_868276.mp4)
- Each prompt chart has 5 scored videos because 1 upload per prompt was still evaluating/no-score in the HTML snapshot.
- Scores are percentages. Average is the unweighted mean across the 11 metrics.
- Baseline/LiteLinear labels are verified by SHA-256 hash matching against `LiteLinear/benchmarks/wan_i2v_20260424_025712/videos_flattened`; no-score rows are excluded from averages.

## LL vs Baseline

| Prompt Filename | Prompt Group | Variant | Total | Evaluated Videos | Videos |
| --- | --- | --- | ---: | ---: | --- |
| surf-cat | cat surfboard | baseline | 88.41% | 3 | 834323 · baseline__run_01-03_seed_868276.mp4<br>519042 · baseline__run_01-03_seed_868276.mp4<br>395822 · baseline__run_01-03_seed_868276.mp4 |
| surf-cat | cat surfboard | LL | 97.73% | 2 | 167501 · litelinear__run_01-03_seed_868276.mp4<br>896203 · litelinear__run_01-03_seed_868276.mp4 |
| angelic-clock | angels clock | baseline | 82.95% | 3 | 779763 · baseline__run_01-03_seed_868276.mp4<br>873195 · baseline__run_01-03_seed_868276.mp4<br>925612 · baseline__run_01-03_seed_868276.mp4 |
| angelic-clock | angels clock | LL | 76.02% | 2 | 672401 · litelinear__run_01-03_seed_868276.mp4<br>474878 · litelinear__run_01-03_seed_868276.mp4 |

## No-Score Videos

| WorldJen ID | Prompt Filename | Variant | Original Filename | Reason |
| --- | --- | --- | --- | --- |
| 527370 | surf-cat | LL | `litelinear__run_01-03_seed_868276.mp4` | Still evaluating / no scores in provided HTML |
| 298207 | angelic-clock | LL | `litelinear__run_01-03_seed_868276.mp4` | Still evaluating / no scores in provided HTML |

## Grouped Prompt Charts

### surf-cat

- Prompt group: cat surfboard
- Hash-mapped local file groups: `surf-cat__baseline__run_01-03_seed_868276.mp4`, `surf-cat__litelinear__run_01-03_seed_868276.mp4`

![](charts/surf-cat_ll_vs_baseline_metrics_total.png)

### angelic-clock

- Prompt group: angels clock
- Hash-mapped local file groups: `angelic-clock__baseline__run_01-03_seed_868276.mp4`, `angelic-clock__litelinear__run_01-03_seed_868276.mp4`

![](charts/angelic-clock_ll_vs_baseline_metrics_total.png)

## Per-Video Metric Charts

### surf-cat

![](charts/surf-cat_videos_by_metric.png)

### angelic-clock

![](charts/angelic-clock_videos_by_metric.png)

## Summary

| Video | Prompt Filename | Variant | Original Filename | Average | LL vs Baseline Chart | Per-Video Chart | Thumbnail |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| 167501 · litelinear__run_01-03_seed_868276.mp4 | surf-cat | LL | `litelinear__run_01-03_seed_868276.mp4` | 100.00% | [chart](charts/surf-cat_ll_vs_baseline_metrics_total.png) | [chart](charts/surf-cat_videos_by_metric.png) | thumbnails/167501.jpg |
| 896203 · litelinear__run_01-03_seed_868276.mp4 | surf-cat | LL | `litelinear__run_01-03_seed_868276.mp4` | 95.45% | [chart](charts/surf-cat_ll_vs_baseline_metrics_total.png) | [chart](charts/surf-cat_videos_by_metric.png) | thumbnails/896203.jpg |
| 519042 · baseline__run_01-03_seed_868276.mp4 | surf-cat | baseline | `baseline__run_01-03_seed_868276.mp4` | 91.36% | [chart](charts/surf-cat_ll_vs_baseline_metrics_total.png) | [chart](charts/surf-cat_videos_by_metric.png) | thumbnails/519042.jpg |
| 779763 · baseline__run_01-03_seed_868276.mp4 | angelic-clock | baseline | `baseline__run_01-03_seed_868276.mp4` | 90.68% | [chart](charts/angelic-clock_ll_vs_baseline_metrics_total.png) | [chart](charts/angelic-clock_videos_by_metric.png) | thumbnails/779763.jpg |
| 873195 · baseline__run_01-03_seed_868276.mp4 | angelic-clock | baseline | `baseline__run_01-03_seed_868276.mp4` | 90.68% | [chart](charts/angelic-clock_ll_vs_baseline_metrics_total.png) | [chart](charts/angelic-clock_videos_by_metric.png) | thumbnails/873195.jpg |
| 395822 · baseline__run_01-03_seed_868276.mp4 | surf-cat | baseline | `baseline__run_01-03_seed_868276.mp4` | 87.27% | [chart](charts/surf-cat_ll_vs_baseline_metrics_total.png) | [chart](charts/surf-cat_videos_by_metric.png) | thumbnails/395822.jpg |
| 834323 · baseline__run_01-03_seed_868276.mp4 | surf-cat | baseline | `baseline__run_01-03_seed_868276.mp4` | 86.59% | [chart](charts/surf-cat_ll_vs_baseline_metrics_total.png) | [chart](charts/surf-cat_videos_by_metric.png) | thumbnails/834323.jpg |
| 474878 · litelinear__run_01-03_seed_868276.mp4 | angelic-clock | LL | `litelinear__run_01-03_seed_868276.mp4` | 83.86% | [chart](charts/angelic-clock_ll_vs_baseline_metrics_total.png) | [chart](charts/angelic-clock_videos_by_metric.png) | thumbnails/474878.jpg |
| 672401 · litelinear__run_01-03_seed_868276.mp4 | angelic-clock | LL | `litelinear__run_01-03_seed_868276.mp4` | 68.18% | [chart](charts/angelic-clock_ll_vs_baseline_metrics_total.png) | [chart](charts/angelic-clock_videos_by_metric.png) | thumbnails/672401.jpg |
| 925612 · baseline__run_01-03_seed_868276.mp4 | angelic-clock | baseline | `baseline__run_01-03_seed_868276.mp4` | 67.50% | [chart](charts/angelic-clock_ll_vs_baseline_metrics_total.png) | [chart](charts/angelic-clock_videos_by_metric.png) | thumbnails/925612.jpg |

## Per-Video Metrics

### 167501 · litelinear__run_01-03_seed_868276.mp4

- Prompt filename: surf-cat
- Prompt group: cat surfboard
- Variant: LL
- Original filename: `litelinear__run_01-03_seed_868276.mp4`
- Hash-mapped local file group: `surf-cat__litelinear__run_01-03_seed_868276.mp4`
- Average: 100.00%
- Video URL: https://dev-app.worldjen.com/files/25b472e5c5b4d45da0c38b53336ff87ae7c5d1754975d269c0840e76a89c9d4c.mp4
- Grouped chart: [charts/surf-cat_ll_vs_baseline_metrics_total.png](charts/surf-cat_ll_vs_baseline_metrics_total.png)
- Per-video chart: [charts/surf-cat_videos_by_metric.png](charts/surf-cat_videos_by_metric.png)
- Thumbnail: ![](thumbnails/167501.jpg)

| Metric | Score |
| --- | ---: |
| subject consistency | 100.00% |
| scene consistency | 100.00% |
| motion smoothness | 100.00% |
| temporal flickering | 100.00% |
| physical mechanics | 100.00% |
| object permanence | 100.00% |
| human fidelity | 100.00% |
| dynamic degree | 100.00% |
| semantic adherence | 100.00% |
| spatial relationship | 100.00% |
| semantic drift | 100.00% |

### 834323 · baseline__run_01-03_seed_868276.mp4

- Prompt filename: surf-cat
- Prompt group: cat surfboard
- Variant: baseline
- Original filename: `baseline__run_01-03_seed_868276.mp4`
- Hash-mapped local file group: `surf-cat__baseline__run_01-03_seed_868276.mp4`
- Average: 86.59%
- Video URL: https://dev-app.worldjen.com/files/28301c75fabe1be00ebbccfdd9c9b587194bf4d1c47f1a47433c76199afdc425.mp4
- Grouped chart: [charts/surf-cat_ll_vs_baseline_metrics_total.png](charts/surf-cat_ll_vs_baseline_metrics_total.png)
- Per-video chart: [charts/surf-cat_videos_by_metric.png](charts/surf-cat_videos_by_metric.png)
- Thumbnail: ![](thumbnails/834323.jpg)

| Metric | Score |
| --- | ---: |
| subject consistency | 92.50% |
| scene consistency | 95.00% |
| motion smoothness | 75.00% |
| temporal flickering | 85.00% |
| physical mechanics | 85.00% |
| object permanence | 92.50% |
| human fidelity | 90.00% |
| dynamic degree | 47.50% |
| semantic adherence | 100.00% |
| spatial relationship | 90.00% |
| semantic drift | 100.00% |

### 896203 · litelinear__run_01-03_seed_868276.mp4

- Prompt filename: surf-cat
- Prompt group: cat surfboard
- Variant: LL
- Original filename: `litelinear__run_01-03_seed_868276.mp4`
- Hash-mapped local file group: `surf-cat__litelinear__run_01-03_seed_868276.mp4`
- Average: 95.45%
- Video URL: https://dev-app.worldjen.com/files/bcbd68c8b1f8c090bc1c9c6370a44f7254288c4b750fa64e86a51a971f2fbaa8.mp4
- Grouped chart: [charts/surf-cat_ll_vs_baseline_metrics_total.png](charts/surf-cat_ll_vs_baseline_metrics_total.png)
- Per-video chart: [charts/surf-cat_videos_by_metric.png](charts/surf-cat_videos_by_metric.png)
- Thumbnail: ![](thumbnails/896203.jpg)

| Metric | Score |
| --- | ---: |
| subject consistency | 95.00% |
| scene consistency | 97.50% |
| motion smoothness | 97.50% |
| temporal flickering | 97.50% |
| physical mechanics | 92.50% |
| object permanence | 97.50% |
| human fidelity | 97.50% |
| dynamic degree | 75.00% |
| semantic adherence | 100.00% |
| spatial relationship | 100.00% |
| semantic drift | 100.00% |

### 519042 · baseline__run_01-03_seed_868276.mp4

- Prompt filename: surf-cat
- Prompt group: cat surfboard
- Variant: baseline
- Original filename: `baseline__run_01-03_seed_868276.mp4`
- Hash-mapped local file group: `surf-cat__baseline__run_01-03_seed_868276.mp4`
- Average: 91.36%
- Video URL: https://dev-app.worldjen.com/files/4062e229caa9f268a8de98b2d7ff4512e663f2ff2ee332ac5b3b9fbb374fcafa.mp4
- Grouped chart: [charts/surf-cat_ll_vs_baseline_metrics_total.png](charts/surf-cat_ll_vs_baseline_metrics_total.png)
- Per-video chart: [charts/surf-cat_videos_by_metric.png](charts/surf-cat_videos_by_metric.png)
- Thumbnail: ![](thumbnails/519042.jpg)

| Metric | Score |
| --- | ---: |
| subject consistency | 95.00% |
| scene consistency | 100.00% |
| motion smoothness | 75.00% |
| temporal flickering | 82.50% |
| physical mechanics | 82.50% |
| object permanence | 100.00% |
| human fidelity | 95.00% |
| dynamic degree | 75.00% |
| semantic adherence | 100.00% |
| spatial relationship | 100.00% |
| semantic drift | 100.00% |

### 395822 · baseline__run_01-03_seed_868276.mp4

- Prompt filename: surf-cat
- Prompt group: cat surfboard
- Variant: baseline
- Original filename: `baseline__run_01-03_seed_868276.mp4`
- Hash-mapped local file group: `surf-cat__baseline__run_01-03_seed_868276.mp4`
- Average: 87.27%
- Video URL: https://dev-app.worldjen.com/files/ceb118037cabc98cb8a4bd729fd217fd8c1aadeb64999b575980dc5baf2d5671.mp4
- Grouped chart: [charts/surf-cat_ll_vs_baseline_metrics_total.png](charts/surf-cat_ll_vs_baseline_metrics_total.png)
- Per-video chart: [charts/surf-cat_videos_by_metric.png](charts/surf-cat_videos_by_metric.png)
- Thumbnail: ![](thumbnails/395822.jpg)

| Metric | Score |
| --- | ---: |
| subject consistency | 92.50% |
| scene consistency | 95.00% |
| motion smoothness | 75.00% |
| temporal flickering | 92.50% |
| physical mechanics | 85.00% |
| object permanence | 90.00% |
| human fidelity | 90.00% |
| dynamic degree | 47.50% |
| semantic adherence | 100.00% |
| spatial relationship | 92.50% |
| semantic drift | 100.00% |

### 779763 · baseline__run_01-03_seed_868276.mp4

- Prompt filename: angelic-clock
- Prompt group: angels clock
- Variant: baseline
- Original filename: `baseline__run_01-03_seed_868276.mp4`
- Hash-mapped local file group: `angelic-clock__baseline__run_01-03_seed_868276.mp4`
- Average: 90.68%
- Video URL: https://dev-app.worldjen.com/files/f8b5f62039fb23e4183da1f60a913b08d25e41dc4a1e17e23231cf4bf61a5b6a.mp4
- Grouped chart: [charts/angelic-clock_ll_vs_baseline_metrics_total.png](charts/angelic-clock_ll_vs_baseline_metrics_total.png)
- Per-video chart: [charts/angelic-clock_videos_by_metric.png](charts/angelic-clock_videos_by_metric.png)
- Thumbnail: ![](thumbnails/779763.jpg)

| Metric | Score |
| --- | ---: |
| subject consistency | 97.50% |
| scene consistency | 100.00% |
| motion smoothness | 100.00% |
| temporal flickering | 100.00% |
| physical mechanics | 100.00% |
| object permanence | 100.00% |
| human fidelity | 100.00% |
| dynamic degree | 0.00% |
| semantic adherence | 100.00% |
| spatial relationship | 100.00% |
| semantic drift | 100.00% |

### 873195 · baseline__run_01-03_seed_868276.mp4

- Prompt filename: angelic-clock
- Prompt group: angels clock
- Variant: baseline
- Original filename: `baseline__run_01-03_seed_868276.mp4`
- Hash-mapped local file group: `angelic-clock__baseline__run_01-03_seed_868276.mp4`
- Average: 90.68%
- Video URL: https://dev-app.worldjen.com/files/16f1203fbb1a1bdeee2ff1725eb49cf940fb8d3c1be9c80888a48a8bee36169b.mp4
- Grouped chart: [charts/angelic-clock_ll_vs_baseline_metrics_total.png](charts/angelic-clock_ll_vs_baseline_metrics_total.png)
- Per-video chart: [charts/angelic-clock_videos_by_metric.png](charts/angelic-clock_videos_by_metric.png)
- Thumbnail: ![](thumbnails/873195.jpg)

| Metric | Score |
| --- | ---: |
| subject consistency | 97.50% |
| scene consistency | 100.00% |
| motion smoothness | 100.00% |
| temporal flickering | 100.00% |
| physical mechanics | 100.00% |
| object permanence | 100.00% |
| human fidelity | 100.00% |
| dynamic degree | 0.00% |
| semantic adherence | 100.00% |
| spatial relationship | 100.00% |
| semantic drift | 100.00% |

### 672401 · litelinear__run_01-03_seed_868276.mp4

- Prompt filename: angelic-clock
- Prompt group: angels clock
- Variant: LL
- Original filename: `litelinear__run_01-03_seed_868276.mp4`
- Hash-mapped local file group: `angelic-clock__litelinear__run_01-03_seed_868276.mp4`
- Average: 68.18%
- Video URL: https://dev-app.worldjen.com/files/5889a9ebf3f9489a0fba48ffb83aa5afa830280c12e23686b65105b5d2affbba.mp4
- Grouped chart: [charts/angelic-clock_ll_vs_baseline_metrics_total.png](charts/angelic-clock_ll_vs_baseline_metrics_total.png)
- Per-video chart: [charts/angelic-clock_videos_by_metric.png](charts/angelic-clock_videos_by_metric.png)
- Thumbnail: ![](thumbnails/672401.jpg)

| Metric | Score |
| --- | ---: |
| subject consistency | 77.50% |
| scene consistency | 75.00% |
| motion smoothness | 57.50% |
| temporal flickering | 57.50% |
| physical mechanics | 55.00% |
| object permanence | 72.50% |
| human fidelity | 70.00% |
| dynamic degree | 25.00% |
| semantic adherence | 95.00% |
| spatial relationship | 75.00% |
| semantic drift | 90.00% |

### 925612 · baseline__run_01-03_seed_868276.mp4

- Prompt filename: angelic-clock
- Prompt group: angels clock
- Variant: baseline
- Original filename: `baseline__run_01-03_seed_868276.mp4`
- Hash-mapped local file group: `angelic-clock__baseline__run_01-03_seed_868276.mp4`
- Average: 67.50%
- Video URL: https://dev-app.worldjen.com/files/6b6aa714febfeb86b19be5f8891a49654332eea979eb46c11c86298e6435bd24.mp4
- Grouped chart: [charts/angelic-clock_ll_vs_baseline_metrics_total.png](charts/angelic-clock_ll_vs_baseline_metrics_total.png)
- Per-video chart: [charts/angelic-clock_videos_by_metric.png](charts/angelic-clock_videos_by_metric.png)
- Thumbnail: ![](thumbnails/925612.jpg)

| Metric | Score |
| --- | ---: |
| subject consistency | 67.50% |
| scene consistency | 65.00% |
| motion smoothness | 60.00% |
| temporal flickering | 70.00% |
| physical mechanics | 57.50% |
| object permanence | 65.00% |
| human fidelity | 70.00% |
| dynamic degree | 22.50% |
| semantic adherence | 97.50% |
| spatial relationship | 75.00% |
| semantic drift | 92.50% |

### 474878 · litelinear__run_01-03_seed_868276.mp4

- Prompt filename: angelic-clock
- Prompt group: angels clock
- Variant: LL
- Original filename: `litelinear__run_01-03_seed_868276.mp4`
- Hash-mapped local file group: `angelic-clock__litelinear__run_01-03_seed_868276.mp4`
- Average: 83.86%
- Video URL: https://dev-app.worldjen.com/files/aec11fcb07a8d7d4ee7d6a3ca3000e032f655fc03ed98cb217d2308887b9df8c.mp4
- Grouped chart: [charts/angelic-clock_ll_vs_baseline_metrics_total.png](charts/angelic-clock_ll_vs_baseline_metrics_total.png)
- Per-video chart: [charts/angelic-clock_videos_by_metric.png](charts/angelic-clock_videos_by_metric.png)
- Thumbnail: ![](thumbnails/474878.jpg)

| Metric | Score |
| --- | ---: |
| subject consistency | 95.00% |
| scene consistency | 92.50% |
| motion smoothness | 77.50% |
| temporal flickering | 82.50% |
| physical mechanics | 75.00% |
| object permanence | 87.50% |
| human fidelity | 95.00% |
| dynamic degree | 27.50% |
| semantic adherence | 100.00% |
| spatial relationship | 95.00% |
| semantic drift | 95.00% |

## Skipped Rows

| Video | Prompt Filename | Variant | Original Filename | Hash-Mapped Local File Group | Reason | Video URL |
| --- | --- | --- | --- | --- | --- | --- |
| 527370 · litelinear__run_01-03_seed_868276.mp4 | surf-cat | LL | `litelinear__run_01-03_seed_868276.mp4` | `surf-cat__litelinear__run_01-03_seed_868276.mp4` | Still evaluating / no scores in provided HTML | https://dev-app.worldjen.com/files/315448445e28ae00e70fa8c050697a78db38ef7643dd300cd1ea08e54e8fc9c3.mp4 |
| 298207 · litelinear__run_01-03_seed_868276.mp4 | angelic-clock | LL | `litelinear__run_01-03_seed_868276.mp4` | `angelic-clock__litelinear__run_01-03_seed_868276.mp4` | Still evaluating / no scores in provided HTML | https://dev-app.worldjen.com/files/81b5bb9ff88b101c6f64272c224ae1ae1dc1f8de8a5ebc4403c20a388ed35c80.mp4 |
