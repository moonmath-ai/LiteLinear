import re
import sys
from collections import defaultdict


def analyze_trace_regex(trace_path):
    print(f"Analyzing trace (Regex mode): {trace_path}", flush=True)

    stats = defaultdict(lambda: {"total_dur": 0, "count": 0})

    dur_pattern = re.compile(r'"dur":(\d+\.?\d*)')

    with open(trace_path, "r") as f:
        chunk_size = 50 * 1024 * 1024  # 50MB
        overlap = 1000

        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            content = buffer + chunk

            # 1. Search for named markers
            for m in ["LowRankPath", "FP8Path", "Quantize", "ScaledMM", "Bias"]:
                # find all occurrences
                it = re.finditer(f'"{m}"', content)
                for match in it:
                    off = match.start()
                    window = content[off : off + 500]
                    dur_match = dur_pattern.search(window)
                    if dur_match:
                        stats[m]["total_dur"] += float(dur_match.group(1))
                        stats[m]["count"] += 1

            # 2. Search for block markers
            w_matches = re.finditer(
                r'"LowRankDeltaLinear\.transformer_blocks\.\d+\.ff\.w[12]"', content
            )
            for wm in w_matches:
                off = wm.start()
                window = content[off : off + 500]
                dur_match = dur_pattern.search(window)
                if dur_match:
                    name = wm.group(0).strip('"')
                    stats[name]["total_dur"] += float(dur_match.group(1))
                    stats[name]["count"] += 1

            buffer = content[-overlap:]
            print(f"  Processed {f.tell() / (1024*1024):.1f} MB...", flush=True)

    if not stats:
        print("No matching markers found in trace.", flush=True)
        return

    print("\n=== Profiling Results (Average per Occurrence) ===", flush=True)
    print(f"{'Marker':<45} | {'Avg Dur (us)':<15} | {'Count':<6}", flush=True)
    print("-" * 70, flush=True)

    w1_durs = []
    w2_durs = []

    for name in sorted(stats.keys()):
        s = stats[name]
        avg = s["total_dur"] / s["count"] if s["count"] > 0 else 0
        if s["count"] > 0:
            print(f"{name:<45} | {avg:>12.2f} | {s['count']:>6}", flush=True)

            if ".ff.w1" in name:
                w1_durs.append(avg)
            if ".ff.w2" in name:
                w2_durs.append(avg)

    if w1_durs:
        print(
            f"\nAverage FFN w1 (proj): {sum(w1_durs)/len(w1_durs):.2f} us", flush=True
        )
    if w2_durs:
        print(f"Average FFN w2 (out):  {sum(w2_durs)/len(w2_durs):.2f} us", flush=True)

    if "Quantize" in stats and "ScaledMM" in stats:
        q_avg = stats["Quantize"]["total_dur"] / stats["Quantize"]["count"]
        mm_avg = stats["ScaledMM"]["total_dur"] / stats["ScaledMM"]["count"]
        print(f"\nFP8 Overhead Analysis:", flush=True)
        print(f"  Quantization: {q_avg:.2f} us", flush=True)
        print(f"  Scaled Matmul: {mm_avg:.2f} us", flush=True)
        print(f"  Total Residual: {q_avg + mm_avg:.2f} us", flush=True)


if __name__ == "__main__":
    trace_file = (
        sys.argv[1] if len(sys.argv) > 1 else "ffn_delta_outputs/profile_trace.json"
    )
    analyze_trace_regex(trace_file)
