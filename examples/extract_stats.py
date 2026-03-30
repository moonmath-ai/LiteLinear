import re
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_stats.py <trace.json>")
        return

    path = sys.argv[1]
    print(f"Reading {path}...")
    with open(path, "r") as f:
        content = f.read()

    # 1. Total Layer Stats (w1 vs w2)
    # Using re.DOTALL to match across newlines
    layer_pattern = re.compile(
        r'"name"\s*:\s*"(LowRankDeltaLinear\.transformer_blocks\.\d+\.ff\.w[12])".*?"dur"\s*:\s*([\d.]+)',
        re.DOTALL,
    )
    layer_matches = layer_pattern.findall(content)

    # 1b. Baseline FFN Stats (if no LowRank markers found)
    base_ffn_pattern = re.compile(
        r'"name"\s*:\s*"(nn\.Module:\s*FeedForward_\d+)".*?"dur"\s*:\s*([\d.]+)',
        re.DOTALL,
    )
    base_matches = base_ffn_pattern.findall(content)

    layer_stats = defaultdict(list)
    for name, dur in layer_matches:
        layer_stats[name].append(float(dur))

    # If we only have baseline matches
    if not layer_matches and base_matches:
        print("[stats] Identified Baseline FeedForward markers.")
        for name, dur in base_matches:
            layer_stats[name].append(float(dur))

    # 2. Sub-component Stats
    sub_markers = ["LowRankPath", "FP8Path", "Quantize", "ScaledMM", "Bias"]
    sub_stats = {m: [] for m in sub_markers}

    for m in sub_markers:
        pattern = re.compile(rf'"name"\s*:\s*"{m}".*?"dur"\s*:\s*([\d.]+)', re.DOTALL)
        matches = pattern.findall(content)
        sub_stats[m] = [float(x) for x in matches]

    # Calculate Totals
    w1_list = [sum(v) / len(v) for k, v in layer_stats.items() if ".w1" in k]
    w2_list = [sum(v) / len(v) for k, v in layer_stats.items() if ".w2" in k]

    # If baseline, they are just named FeedForward_X
    base_list = [sum(v) / len(v) for k, v in layer_stats.items() if "FeedForward" in k]

    w1_avg = sum(w1_list) / len(w1_list) if w1_list else 0
    w2_avg = sum(w2_list) / len(w2_list) if w2_list else 0
    base_avg = sum(base_list) / len(base_list) if base_list else 0

    total_ffn_layer_avg = (
        (w1_avg + w2_avg) / 2 if (w1_avg and w2_avg) else (w1_avg or w2_avg or base_avg)
    )

    print("\n" + "=" * 75)
    print(
        f"{'Marker/Region':<45} | {'Avg (us)':>10} | {'% of Block':>10} | {'Count':>6}"
    )
    print("-" * 75)

    # Print Block Regions
    if base_avg:
        print(
            f"{'Baseline FeedForward (Block Total)':<45} | {base_avg:>10.2f} | {'100.0%':>10} | {len(base_list):>6}"
        )
    if w1_avg:
        print(
            f"{'FFN w1 (proj)':<45} | {w1_avg:>10.2f} | {'100.0%':>10} | {len(w1_list):>6}"
        )
    if w2_avg:
        print(
            f"{'FFN w2 (out)':<45} | {w2_avg:>10.2f} | {'100.0%':>10} | {len(w2_list):>6}"
        )

    print("-" * 75)

    # Print Components
    for m in sub_markers:
        durs = sub_stats[m]
        if durs:
            avg = sum(durs) / len(durs)
            # For decomposition, percent is relative to the FFN block
            percent = (avg / total_ffn_layer_avg * 100) if total_ffn_layer_avg else 0
            print(f"{m:<45} | {avg:>10.2f} | {percent:>9.1f}%  | {len(durs):>6}")

    print("=" * 75)

    # Summary Analysis
    if (
        "Quantize" in sub_stats
        and "ScaledMM" in sub_stats
        and sub_stats["Quantize"]
        and sub_stats["ScaledMM"]
    ):
        avg_q = sum(sub_stats["Quantize"]) / len(sub_stats["Quantize"])
        avg_mm = sum(sub_stats["ScaledMM"]) / len(sub_stats["ScaledMM"])
        avg_lr = (
            sum(sub_stats["LowRankPath"]) / len(sub_stats["LowRankPath"])
            if sub_stats["LowRankPath"]
            else 0
        )

        print(f"\nPotential Decomposition Insights:")
        print(
            f"  - FP8 Overhead (Q+MM):  {avg_q + avg_mm:.1f} us ({((avg_q+avg_mm)/total_ffn_layer_avg)*100:.1f}% of layer)"
        )
        print(
            f"  - Low-Rank Path (LR):   {avg_lr:.1f} us ({((avg_lr)/total_ffn_layer_avg)*100:.1f}% of layer)"
        )
        print(f"  - Kernel Fusion Goal:   Targeting <150us for entire fused FFN delta.")


if __name__ == "__main__":
    main()
