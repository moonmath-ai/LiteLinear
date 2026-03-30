import subprocess
import re
import sys
import os

MODES = ["reduce-overhead", "max-autotune"]
CMD_TEMPLATE = [
    "./ffn/4_decompose.sh",
    "--rank",
    "32",
    "--num_frames",
    "48",
    "--static_scale",
    "12.0",
    "--compile",
    "--benchmark_runs",
    "3",
]


def parse_latency(stdout):
    # Looking for: [bench] Decomposed: 4.175 s/run
    match = re.search(r"\[bench\] Decomposed:\s+([\d\.]+)\s+s/run", stdout)
    if match:
        return float(match.group(1))
    return None


def run_bench(mode, is_baseline):
    print(f"\n>>> Running: Mode='{mode}', IsBaseline={is_baseline}")

    cmd = list(CMD_TEMPLATE)
    cmd.extend(["--compile_mode", mode])

    if is_baseline:
        # Blocks 999 = No decomposition = Baseline Latency measured at 'Decomposed' step
        cmd.extend(["--blocks", "999"])
        # We don't skip baseline arg because we want the script to run through
        # But wait, if blocks=999, it runs baseline measurement (if not skipped) AND "decomposed" (which is just baseline).
        # To be purely isolated and fast, we can use --skip_baseline and read the "Decomposed" value which is effectively reloaded baseline.
        cmd.append("--skip_baseline")
    else:
        # Real Decomposed
        cmd.extend(["--blocks", "0-27"])
        cmd.append("--skip_baseline")  # Isolated run

    # Environment variables
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, check=True
        )
        lat = parse_latency(result.stdout)
        print(f"    Result: {lat} s")
        return lat
    except subprocess.CalledProcessError as e:
        print(f"    FAILED: Exit Code {e.returncode}")
        print(f"    Stderr: {e.stderr[-500:]}")  # Last 500 chars
        return None


def main():
    print("==================================================")
    print("   Compilation Mode Sweep: Baseline vs Decomposed")
    print("==================================================")

    results = {}

    for mode in MODES:
        base_lat = run_bench(mode, is_baseline=True)
        if base_lat is None:
            results[mode] = (None, None)
            continue

        decomp_lat = run_bench(mode, is_baseline=False)
        results[mode] = (base_lat, decomp_lat)

    print("\n\n==================================================")
    print("                FINAL REPORT")
    print("==================================================")
    print(
        f"{'Mode':<20} | {'Baseline (s)':<12} | {'Decomposed (s)':<14} | {'Speedup':<10}"
    )
    print("-" * 65)

    for mode in MODES:
        base, decomp = results[mode]
        if base and decomp:
            speedup = base / decomp
            status = (
                "WIN" if speedup > 1.05 else ("LOSS" if speedup < 1.0 else "NEUTRAL")
            )
            print(
                f"{mode:<20} | {base:<12.3f} | {decomp:<14.3f} | {speedup:.2f}x ({status})"
            )
        else:
            print(f"{mode:<20} | {'FAIL':<12} | {'FAIL':<14} | -")

    print("==================================================")


if __name__ == "__main__":
    main()
