#!/usr/bin/env python3
"""Generate extracted WorldJen benchmark result artifacts."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent
CHART_DIR = OUTPUT_DIR / "charts"
THUMB_DIR = OUTPUT_DIR / "thumbnails"

METRICS = [
    "subject consistency",
    "scene consistency",
    "motion smoothness",
    "temporal flickering",
    "physical mechanics",
    "object permanence",
    "human fidelity",
    "dynamic degree",
    "semantic adherence",
    "spatial relationship",
    "semantic drift",
]

PROMPT_FILENAMES = {
    "cat surfboard": "surf-cat",
    "angels clock": "angelic-clock",
}

VARIANT_FILENAMES = {
    "baseline": "baseline",
    "LiteLinear": "litelinear",
}

VARIANT_LABELS = {
    "baseline": "baseline",
    "LiteLinear": "LL",
}

VARIANT_COLORS = {
    "baseline": ["#1d4ed8", "#3b82f6", "#93c5fd"],
    "LiteLinear": ["#c2410c", "#f97316", "#fdba74"],
}

CAT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, distant "
    "green hills, and a blue sky dotted with white clouds. The cat assumes a naturally "
    "relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot "
    "highlights the feline's intricate details and the refreshing atmosphere of the seaside"
)

ANGEL_PROMPT = (
    "A serene, ethereal scene featuring two angelic beings with flowing white robes and "
    "golden halos. One angel gently holds a large ornate clock with intricate designs, "
    "while the other tenderly watches over it. Both angels have fair skin, delicate "
    "features, and long, flowing hair that matches their robes. They stand in a heavenly "
    "garden filled with soft, glowing light and floating celestial flowers. The background "
    "showcases a starry night sky with a gentle, luminous glow. The angels are positioned "
    "in a close-up, medium shot, emphasizing their graceful postures and serene expressions."
)

VIDEOS = [
    {
        "id": "167501",
        "prompt_group": "cat surfboard",
        "model_variant": "LiteLinear",
        "prompt": CAT_PROMPT,
        "video_url": "https://dev-app.worldjen.com/files/25b472e5c5b4d45da0c38b53336ff87ae7c5d1754975d269c0840e76a89c9d4c.mp4",
        "scores": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    },
    {
        "id": "834323",
        "prompt_group": "cat surfboard",
        "model_variant": "baseline",
        "prompt": CAT_PROMPT,
        "video_url": "https://dev-app.worldjen.com/files/28301c75fabe1be00ebbccfdd9c9b587194bf4d1c47f1a47433c76199afdc425.mp4",
        "scores": [92.5, 95, 75, 85, 85, 92.5, 90, 47.5, 100, 90, 100],
    },
    {
        "id": "896203",
        "prompt_group": "cat surfboard",
        "model_variant": "LiteLinear",
        "prompt": CAT_PROMPT,
        "video_url": "https://dev-app.worldjen.com/files/bcbd68c8b1f8c090bc1c9c6370a44f7254288c4b750fa64e86a51a971f2fbaa8.mp4",
        "scores": [95, 97.5, 97.5, 97.5, 92.5, 97.5, 97.5, 75, 100, 100, 100],
    },
    {
        "id": "519042",
        "prompt_group": "cat surfboard",
        "model_variant": "baseline",
        "prompt": CAT_PROMPT,
        "video_url": "https://dev-app.worldjen.com/files/4062e229caa9f268a8de98b2d7ff4512e663f2ff2ee332ac5b3b9fbb374fcafa.mp4",
        "scores": [95, 100, 75, 82.5, 82.5, 100, 95, 75, 100, 100, 100],
    },
    {
        "id": "395822",
        "prompt_group": "cat surfboard",
        "model_variant": "baseline",
        "prompt": CAT_PROMPT,
        "video_url": "https://dev-app.worldjen.com/files/ceb118037cabc98cb8a4bd729fd217fd8c1aadeb64999b575980dc5baf2d5671.mp4",
        "scores": [92.5, 95, 75, 92.5, 85, 90, 90, 47.5, 100, 92.5, 100],
    },
    {
        "id": "779763",
        "prompt_group": "angels clock",
        "model_variant": "baseline",
        "prompt": ANGEL_PROMPT,
        "video_url": "https://dev-app.worldjen.com/files/f8b5f62039fb23e4183da1f60a913b08d25e41dc4a1e17e23231cf4bf61a5b6a.mp4",
        "scores": [97.5, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100],
    },
    {
        "id": "873195",
        "prompt_group": "angels clock",
        "model_variant": "baseline",
        "prompt": ANGEL_PROMPT,
        "video_url": "https://dev-app.worldjen.com/files/16f1203fbb1a1bdeee2ff1725eb49cf940fb8d3c1be9c80888a48a8bee36169b.mp4",
        "scores": [97.5, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100],
    },
    {
        "id": "672401",
        "prompt_group": "angels clock",
        "model_variant": "LiteLinear",
        "prompt": ANGEL_PROMPT,
        "video_url": "https://dev-app.worldjen.com/files/5889a9ebf3f9489a0fba48ffb83aa5afa830280c12e23686b65105b5d2affbba.mp4",
        "scores": [77.5, 75, 57.5, 57.5, 55, 72.5, 70, 25, 95, 75, 90],
    },
    {
        "id": "925612",
        "prompt_group": "angels clock",
        "model_variant": "baseline",
        "prompt": ANGEL_PROMPT,
        "video_url": "https://dev-app.worldjen.com/files/6b6aa714febfeb86b19be5f8891a49654332eea979eb46c11c86298e6435bd24.mp4",
        "scores": [67.5, 65, 60, 70, 57.5, 65, 70, 22.5, 97.5, 75, 92.5],
    },
    {
        "id": "474878",
        "prompt_group": "angels clock",
        "model_variant": "LiteLinear",
        "prompt": ANGEL_PROMPT,
        "video_url": "https://dev-app.worldjen.com/files/aec11fcb07a8d7d4ee7d6a3ca3000e032f655fc03ed98cb217d2308887b9df8c.mp4",
        "scores": [95, 92.5, 77.5, 82.5, 75, 87.5, 95, 27.5, 100, 95, 95],
    },
]

SKIPPED = [
    {
        "id": "527370",
        "prompt_group": "cat surfboard",
        "model_variant": "LiteLinear",
        "reason": "Still evaluating / no scores in provided HTML",
        "video_url": "https://dev-app.worldjen.com/files/315448445e28ae00e70fa8c050697a78db38ef7643dd300cd1ea08e54e8fc9c3.mp4",
    },
    {
        "id": "298207",
        "prompt_group": "angels clock",
        "model_variant": "LiteLinear",
        "reason": "Still evaluating / no scores in provided HTML",
        "video_url": "https://dev-app.worldjen.com/files/81b5bb9ff88b101c6f64272c224ae1ae1dc1f8de8a5ebc4403c20a388ed35c80.mp4",
    },
]


def average(scores: list[float]) -> float:
    return sum(scores) / len(scores)


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def prompt_group_slug(prompt_group: str) -> str:
    return prompt_group.replace(" ", "_")


def local_prompt_filename(video: dict[str, object]) -> str:
    return PROMPT_FILENAMES[str(video["prompt_group"])]


def local_file_group(video: dict[str, object]) -> str:
    prompt_filename = local_prompt_filename(video)
    variant_filename = VARIANT_FILENAMES[str(video["model_variant"])]
    return f"{prompt_filename}__{variant_filename}__run_01-03_seed_868276.mp4"


def original_filename_without_prompt(video: dict[str, object]) -> str:
    variant_filename = VARIANT_FILENAMES[str(video["model_variant"])]
    return f"{variant_filename}__run_01-03_seed_868276.mp4"


def video_label(video: dict[str, object]) -> str:
    return f"{video['id']} · {original_filename_without_prompt(video)}"


def variant_color(variant: str, index: int) -> str:
    colors = VARIANT_COLORS[variant]
    return colors[index % len(colors)]


def ll_vs_baseline_chart_path(prompt_group: str) -> str:
    return f"charts/{PROMPT_FILENAMES[prompt_group]}_ll_vs_baseline_metrics_total.png"


def per_video_chart_path(prompt_group: str) -> str:
    return f"charts/{PROMPT_FILENAMES[prompt_group]}_videos_by_metric.png"


def prompt_groups() -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for video in VIDEOS:
        groups.setdefault(str(video["prompt_group"]), []).append(video)
    return groups


def variant_metric_rows() -> list[dict[str, object]]:
    rows = []
    for prompt_group, videos in prompt_groups().items():
        for variant in ["baseline", "LiteLinear"]:
            matching = [video for video in videos if video["model_variant"] == variant]
            if not matching:
                continue
            metric_averages = {}
            for metric_index, metric in enumerate(METRICS):
                metric_averages[metric] = sum(video["scores"][metric_index] for video in matching) / len(matching)
            total = sum(metric_averages.values()) / len(METRICS)
            rows.append(
                {
                    "prompt_group": prompt_group,
                    "model_variant": variant,
                    "total": total,
                    "video_count": len(matching),
                    "video_ids": ", ".join(str(video["id"]) for video in matching),
                    "metrics": metric_averages,
                }
            )
    return rows


def make_prompt_video_chart(prompt_group: str, videos: list[dict[str, object]], filename: str) -> None:
    videos = sorted(videos, key=lambda video: (0 if video["model_variant"] == "baseline" else 1, str(video["id"])))
    fig, ax = plt.subplots(figsize=(22, 8.5))
    group_width = 1.05
    metric_gap = 1.35
    bar_width = group_width / len(videos)
    metric_positions = [index * metric_gap for index in range(len(METRICS))]
    variant_seen = {"baseline": 0, "LiteLinear": 0}

    for index, video in enumerate(videos):
        scores = video["scores"]
        assert isinstance(scores, list)
        variant = str(video["model_variant"])
        color = variant_color(variant, variant_seen[variant])
        variant_seen[variant] += 1
        offsets = [
            position - group_width / 2 + bar_width / 2 + index * bar_width
            for position in metric_positions
        ]
        label = f"{video_label(video)} ({average(scores):.1f}%)"
        bars = ax.bar(offsets, scores, width=bar_width, label=label, color=color)
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(score + 1.2, 106),
                f"{score:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(f"Per-video WorldJen scores by metric - {PROMPT_FILENAMES[prompt_group]}", fontsize=14, weight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_xticks(metric_positions)
    ax.set_xticklabels(METRICS, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="WorldJen ID + original filename", loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=160)
    plt.close(fig)


def make_variant_metric_chart(scope_name: str, videos: list[dict[str, object]], filename: str) -> None:
    variants = ["baseline", "LiteLinear"]
    labels = [*METRICS, "total"]

    fig, ax = plt.subplots(figsize=(16, 7))
    group_width = 0.6
    bar_width = group_width / len(variants)
    x_positions = list(range(len(labels)))

    for index, variant in enumerate(variants):
        matching = [video for video in videos if video["model_variant"] == variant]
        if not matching:
            continue
        metric_averages = [
            sum(video["scores"][metric_index] for video in matching) / len(matching)
            for metric_index in range(len(METRICS))
        ]
        scores = [*metric_averages, average(metric_averages)]
        offsets = [
            position - group_width / 2 + bar_width / 2 + index * bar_width
            for position in x_positions
        ]
        bars = ax.bar(
            offsets,
            scores,
            width=bar_width,
            label=f"{VARIANT_LABELS[variant]} (n={len(matching)})",
            color=variant_color(variant, 1),
        )
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(score + 1.2, 103),
                f"{score:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(f"LL vs baseline by metric - {scope_name}", fontsize=14, weight="bold")
    ax.set_ylabel("Average score (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=2)
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=160)
    plt.close(fig)


def make_thumbnail(video: dict[str, object]) -> bool:
    thumbnail_path = THUMB_DIR / f"{video['id']}.jpg"
    if thumbnail_path.exists():
        return True

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        "0.5",
        "-i",
        str(video["video_url"]),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(thumbnail_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=45)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return thumbnail_path.exists()


def write_csv() -> None:
    fieldnames = [
        "video_id",
        "prompt_filename",
        "prompt_group",
        "model_variant",
        "local_file_group",
        "original_filename_without_prompt",
        "average",
        "video_url",
        *METRICS,
        "prompt",
    ]
    with (OUTPUT_DIR / "worldjen_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for video in VIDEOS:
            scores = video["scores"]
            assert isinstance(scores, list)
            row = {
                "video_id": video["id"],
                "prompt_filename": local_prompt_filename(video),
                "prompt_group": video["prompt_group"],
                "model_variant": video["model_variant"],
                "local_file_group": local_file_group(video),
                "original_filename_without_prompt": original_filename_without_prompt(video),
                "average": f"{average(scores):.4f}",
                "video_url": video["video_url"],
                "prompt": video["prompt"],
            }
            row.update({metric: f"{score:.2f}" for metric, score in zip(METRICS, scores)})
            writer.writerow(row)


def write_variant_csv() -> None:
    fieldnames = ["prompt_filename", "prompt_group", "model_variant", *METRICS, "total", "video_count", "videos"]
    with (OUTPUT_DIR / "worldjen_variant_averages.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in variant_metric_rows():
            row_videos = [
                video
                for video in VIDEOS
                if video["model_variant"] == row["model_variant"] and video["prompt_group"] == row["prompt_group"]
            ]
            writer.writerow(
                {
                    "prompt_filename": PROMPT_FILENAMES[str(row["prompt_group"])],
                    "prompt_group": row["prompt_group"],
                    "model_variant": row["model_variant"],
                    **{metric: f"{row['metrics'][metric]:.4f}" for metric in METRICS},
                    "total": f"{row['total']:.4f}",
                    "video_count": row["video_count"],
                    "videos": "; ".join(video_label(video) for video in row_videos),
                }
            )


def write_markdown(thumbnail_status: dict[str, bool]) -> None:
    lines = [
        "# WorldJen Results",
        "",
        "Extracted from the provided WorldJen playground HTML snapshot.",
        "",
        f"- Evaluated videos: {len(VIDEOS)}",
        f"- Skipped/no-score videos: {len(SKIPPED)} ({', '.join(video_label(item) for item in SKIPPED)})",
        "- Each prompt chart has 5 scored videos because 1 upload per prompt was still evaluating/no-score in the HTML snapshot.",
        "- Scores are percentages. Average is the unweighted mean across the 11 metrics.",
        "- Baseline/LiteLinear labels are verified by SHA-256 hash matching against `LiteLinear/benchmarks/wan_i2v_20260424_025712/videos_flattened`; no-score rows are excluded from averages.",
        "",
        "## LL vs Baseline",
        "",
    ]

    lines.extend(
        [
        "| Prompt Filename | Prompt Group | Variant | Total | Evaluated Videos | Videos |",
        "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )

    for row in variant_metric_rows():
        prompt_filename = PROMPT_FILENAMES[str(row["prompt_group"])]
        row_videos = [video for video in VIDEOS if video["model_variant"] == row["model_variant"] and video["prompt_group"] == row["prompt_group"]]
        videos_label = "<br>".join(video_label(video) for video in row_videos)
        lines.append(
            f"| {prompt_filename} | {row['prompt_group']} | {VARIANT_LABELS[str(row['model_variant'])]} | {format_percent(row['total'])} | "
            f"{row['video_count']} | {videos_label} |"
        )

    lines.extend(
        [
            "",
            "## No-Score Videos",
            "",
            "| WorldJen ID | Prompt Filename | Variant | Original Filename | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for item in SKIPPED:
        lines.append(
            f"| {item['id']} | {local_prompt_filename(item)} | {VARIANT_LABELS[str(item['model_variant'])]} | "
            f"`{original_filename_without_prompt(item)}` | {item['reason']} |"
        )

    lines.extend(
        [
        "",
        "## Grouped Prompt Charts",
        "",
        ]
    )

    for prompt_group in prompt_groups():
        prompt_filename = PROMPT_FILENAMES[prompt_group]
        chart_path = ll_vs_baseline_chart_path(prompt_group)
        lines.extend(
            [
                f"### {prompt_filename}",
                "",
                f"- Prompt group: {prompt_group}",
                f"- Hash-mapped local file groups: `{prompt_filename}__baseline__run_01-03_seed_868276.mp4`, `{prompt_filename}__litelinear__run_01-03_seed_868276.mp4`",
                "",
                f"![]({chart_path})",
                "",
            ]
        )

    lines.extend(
        [
            "## Per-Video Metric Charts",
            "",
        ]
    )

    for prompt_group in prompt_groups():
        prompt_filename = PROMPT_FILENAMES[prompt_group]
        lines.extend(
            [
                f"### {prompt_filename}",
                "",
                f"![]({per_video_chart_path(prompt_group)})",
                "",
            ]
        )

    lines.extend(
        [
        "## Summary",
        "",
            "| Video | Prompt Filename | Variant | Original Filename | Average | LL vs Baseline Chart | Per-Video Chart | Thumbnail |",
            "| --- | --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )

    for video in sorted(VIDEOS, key=lambda item: average(item["scores"]), reverse=True):
        scores = video["scores"]
        assert isinstance(scores, list)
        video_id = str(video["id"])
        thumb = f"thumbnails/{video_id}.jpg" if thumbnail_status.get(video_id) else "not generated"
        chart = ll_vs_baseline_chart_path(str(video["prompt_group"]))
        per_video_chart = per_video_chart_path(str(video["prompt_group"]))
        lines.append(
            f"| {video_label(video)} | {local_prompt_filename(video)} | {VARIANT_LABELS[str(video['model_variant'])]} | "
            f"`{original_filename_without_prompt(video)}` | {format_percent(average(scores))} | "
            f"[chart]({chart}) | [chart]({per_video_chart}) | {thumb} |"
        )

    lines.extend(
        [
            "",
            "## Per-Video Metrics",
            "",
        ]
    )

    for video in VIDEOS:
        scores = video["scores"]
        assert isinstance(scores, list)
        video_id = str(video["id"])
        lines.extend(
            [
                f"### {video_label(video)}",
                "",
                f"- Prompt filename: {local_prompt_filename(video)}",
                f"- Prompt group: {video['prompt_group']}",
                f"- Variant: {VARIANT_LABELS[str(video['model_variant'])]}",
                f"- Original filename: `{original_filename_without_prompt(video)}`",
                f"- Hash-mapped local file group: `{local_file_group(video)}`",
                f"- Average: {format_percent(average(scores))}",
                f"- Video URL: {video['video_url']}",
                f"- Grouped chart: [{ll_vs_baseline_chart_path(str(video['prompt_group']))}]({ll_vs_baseline_chart_path(str(video['prompt_group']))})",
                f"- Per-video chart: [{per_video_chart_path(str(video['prompt_group']))}]({per_video_chart_path(str(video['prompt_group']))})",
            ]
        )
        if thumbnail_status.get(video_id):
            lines.append(f"- Thumbnail: ![](thumbnails/{video_id}.jpg)")
        lines.extend(
            [
                "",
                "| Metric | Score |",
                "| --- | ---: |",
            ]
        )
        for metric, score in zip(METRICS, scores):
            lines.append(f"| {metric} | {format_percent(score)} |")
        lines.append("")

    lines.extend(
        [
            "## Skipped Rows",
            "",
            "| Video | Prompt Filename | Variant | Original Filename | Hash-Mapped Local File Group | Reason | Video URL |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in SKIPPED:
        lines.append(
            f"| {video_label(item)} | {local_prompt_filename(item)} | {VARIANT_LABELS[str(item['model_variant'])]} | "
            f"`{original_filename_without_prompt(item)}` | `{local_file_group(item)}` | "
            f"{item['reason']} | {item['video_url']} |"
        )
    lines.append("")

    (OUTPUT_DIR / "worldjen_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    CHART_DIR.mkdir(exist_ok=True)
    THUMB_DIR.mkdir(exist_ok=True)

    write_csv()
    write_variant_csv()

    for chart_path in CHART_DIR.glob("*.png"):
        chart_path.unlink()

    for prompt_group, videos in prompt_groups().items():
        make_variant_metric_chart(
            prompt_group,
            videos,
            f"{PROMPT_FILENAMES[prompt_group]}_ll_vs_baseline_metrics_total.png",
        )
        make_prompt_video_chart(prompt_group, videos, f"{PROMPT_FILENAMES[prompt_group]}_videos_by_metric.png")

    thumbnail_status = {}
    for video in VIDEOS:
        thumbnail_status[str(video["id"])] = make_thumbnail(video)

    write_markdown(thumbnail_status)


if __name__ == "__main__":
    main()
