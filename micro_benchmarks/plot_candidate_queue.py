#!/usr/bin/env python3
"""
Run bench_candidate_queue and plot throughput (items/s) for try_push,
try_push_evict, and pop_best_unexplored, with optional SIMD distance baseline.

Usage:
    # Run with default settings (includes SIMD baseline)
    python plot_candidate_queue.py

    # Custom capacities
    python plot_candidate_queue.py -k 16,32,64,128,256

    # Custom benchmark binary path
    python plot_candidate_queue.py --bin ./build/micro_benchmarks/bench_candidate_queue

    # Skip SIMD baseline measurement
    python plot_candidate_queue.py --no-simd-baseline

    # Custom SIMD benchmark binary path
    python plot_candidate_queue.py --simd-bin ./custom/path/bench_simd_distance

    # Pass extra args to benchmark binary
    python plot_candidate_queue.py --benchmark_repetitions=3 --benchmark_min_time=1.0
"""

import argparse
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Palette ──────────────────────────────────────────────────────────────────
QUEUE_PALETTE = {
    "StdQueue":    "#4C72B0",   # muted blue
    "LinearQueue": "#DD8452",   # warm orange
    "FHQueue":     "#55A868",   # sage green
    "BoostQueue":  "#8172B3",   # muted purple
}
QUEUE_MARKERS = {
    "StdQueue":    "o",
    "LinearQueue": "s",
    "FHQueue":     "D",
    "BoostQueue":  "^",
}

# ── Benchmark groups we care about (removed Mixed) ───────────────────────────
GROUPS = {
    "TryPush":      {"prefix": "BM_TryPush_",      "title": "try_push (fill)"},
    "TryPushEvict": {"prefix": "BM_TryPushEvict_",  "title": "try_push (with eviction)"},
    "GetBest":      {"prefix": "BM_GetBest_",        "title": "pop_best_unexplored (drain)"},
}

QUEUE_SUFFIX_MAP = {
    "StdQueue":    "StdQueue",
    "LinearQueue": "LinearQueue",
    "FHQueue":     "FHQueue",
    "BoostQueue":  "BoostQueue",
}


def run_benchmark(bin_path: str, capacities: str, seed: int, extra_args: list[str]) -> str:
    """Run the benchmark binary and return the text output."""
    cmd = [
        bin_path,
        "-k", capacities,
        "-s", str(seed),
    ] + extra_args

    print(f"[*] Running: {' '.join(cmd)}")
    print("[*] Benchmark output:")
    print("-" * 80)

    # Use Popen for real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )

    # Collect output while printing in real-time
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # Print to terminal in real-time
        output_lines.append(line)

    # Wait for process to complete
    process.wait()

    print("-" * 80)

    if process.returncode != 0:
        stderr = process.stderr.read()
        print(f"[!] Benchmark exited with code {process.returncode}", file=sys.stderr)
        print(stderr, file=sys.stderr)
        sys.exit(1)

    return ''.join(output_lines)


def parse_benchmark_output(output: str) -> pd.DataFrame:
    """Parse Google Benchmark text output into a DataFrame.

    Expected format:
    BM_TryPush_StdQueue/64    1234 ns     1234 ns      567890   items_per_second=123.456M/s
    """
    rows = []

    # Pattern to match benchmark output lines
    # Format: name   time   time   iterations   items_per_second=123.456M/s
    pattern = r'^(BM_\w+/\d+)\s+[\d.]+\s+[a-z]+\s+[\d.]+\s+[a-z]+\s+\d+\s+items_per_second=([\d.]+)([KMG]?)/s'

    print("\n[*] Parsing benchmark output...")
    matched_lines = 0

    for line in output.split('\n'):
        match = re.search(pattern, line.strip())
        if match:
            matched_lines += 1
            name = match.group(1)
            throughput_value = float(match.group(2))
            throughput_unit = match.group(3)

            # Convert to items/s
            multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, '': 1.0}
            items_per_second = throughput_value * multipliers.get(throughput_unit, 1.0)

            # Parse name: "BM_TryPush_StdQueue/64"
            parts = name.rsplit("/", 1)
            if len(parts) != 2:
                continue
            func_name, capacity_str = parts
            capacity = int(capacity_str)

            # Identify group and queue
            group_key = None
            queue_name = None
            for gk, gv in GROUPS.items():
                if func_name.startswith(gv["prefix"]):
                    group_key = gk
                    suffix = func_name[len(gv["prefix"]):]
                    if suffix in QUEUE_SUFFIX_MAP:
                        queue_name = QUEUE_SUFFIX_MAP[suffix]
                    break

            if group_key is None or queue_name is None:
                continue

            rows.append({
                "group":       group_key,
                "queue":       queue_name,
                "capacity":    capacity,
                "items_per_second": items_per_second,
            })

    print(f"[*] Matched {matched_lines} benchmark lines, parsed {len(rows)} valid entries")

    if matched_lines == 0:
        print("\n[!] No lines matched the pattern. Showing sample output lines:")
        for i, line in enumerate(output.split('\n')[:30]):
            if line.strip():
                print(f"  Line {i+1}: {line}")

    return pd.DataFrame(rows)


def run_simd_benchmark(bin_path: str) -> float:
    """Run bench_simd_distance and extract average items_per_second from output."""
    if not os.path.exists(bin_path):
        print(f"[!] SIMD benchmark binary not found: {bin_path}", file=sys.stderr)
        return 0.0

    cmd = [bin_path]

    print(f"\n[*] Running SIMD baseline: {' '.join(cmd)}")
    print("[*] SIMD benchmark output:")
    print("-" * 80)

    # Use Popen for real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )

    # Collect output while printing in real-time
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # Print to terminal in real-time
        output_lines.append(line)

    # Wait for process to complete
    process.wait()

    print("-" * 80)

    if process.returncode != 0:
        stderr = process.stderr.read()
        print(f"[!] SIMD benchmark failed with code {process.returncode}", file=sys.stderr)
        print(stderr, file=sys.stderr)
        return 0.0

    output = ''.join(output_lines)

    # Parse text output
    try:
        items_per_second_values = []

        # Pattern: Artea_L2_U1    14.3 ns    14.3 ns   44705261   items_per_second=69.9M/s
        pattern = r'^\w+\s+[\d.]+\s+[a-z]+\s+[\d.]+\s+[a-z]+\s+\d+\s+items_per_second=([\d.]+)([KMG]?)/s'

        for line in output.split('\n'):
            match = re.search(pattern, line.strip())
            if match:
                throughput_value = float(match.group(1))
                throughput_unit = match.group(2)

                # Convert to items/s
                multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, '': 1.0}
                items_per_second = throughput_value * multipliers.get(throughput_unit, 1.0)
                items_per_second_values.append(items_per_second)

        if not items_per_second_values:
            print("[!] No items_per_second found in SIMD benchmark output", file=sys.stderr)
            return 0.0

        # Return average across all SIMD benchmarks
        avg = np.mean(items_per_second_values)
        print(f"[*] SIMD baseline: {avg/1e6:.2f} M items/s (avg of {len(items_per_second_values)} benchmarks)")
        return avg
    except Exception as e:
        print(f"[!] Failed to parse SIMD benchmark output: {e}", file=sys.stderr)
        return 0.0


def plot_results(df: pd.DataFrame, output_dir: str, simd_baseline: float = None):
    """Generate 3 throughput plots (one per group) with optional SIMD baseline."""
    sns.set_theme(style="whitegrid", font_scale=1.2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("CandidateQueue Throughput Benchmark", fontsize=18, fontweight="bold", y=1.02)

    for col_idx, (gk, gv) in enumerate(GROUPS.items()):
        sub = df[df["group"] == gk]
        if sub.empty:
            continue

        ax = axes[col_idx]

        # ── Throughput (items/s → M items/s) ────────────────────────────
        for qname in ["StdQueue", "LinearQueue", "FHQueue", "BoostQueue"]:
            qsub = sub[sub["queue"] == qname].sort_values("capacity")
            if qsub.empty:
                continue
            throughput_m = qsub["items_per_second"] / 1e6
            ax.plot(
                qsub["capacity"], throughput_m,
                marker=QUEUE_MARKERS[qname], color=QUEUE_PALETTE[qname],
                label=qname, linewidth=2.5, markersize=8,
            )

        # Add SIMD baseline if provided
        if simd_baseline is not None and simd_baseline > 0:
            simd_m = simd_baseline / 1e6
            capacities = sub["capacity"].unique()
            if len(capacities) > 0:
                cap_min, cap_max = capacities.min(), capacities.max()
                ax.axhline(y=simd_m, color='red', linestyle='--', linewidth=2,
                          label=f'SIMD Distance ({simd_m:.1f} M/s)', alpha=0.7)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Capacity (K)", fontsize=12)
        ax.set_ylabel("Throughput (M items/s)", fontsize=12)
        ax.set_title(f"{gv['title']}", fontsize=14, fontweight="bold")
        ax.legend(frameon=True, fancybox=True, shadow=True, loc='best')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "candidate_queue_benchmark.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[*] Plot saved to {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run and plot CandidateQueue benchmarks")
    parser.add_argument("--bin", default="./build/micro_benchmarks/bench_candidate_queue",
                        help="Path to benchmark binary")
    parser.add_argument("-k", "--capacities", default="16,32,64,128,256,512,1024,2048,4096",
                        help="Comma-separated capacity list")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("-o", "--output-dir", default="./micro_benchmarks/plots",
                        help="Directory to save plots")
    parser.add_argument("--simd-bin", default="./build/micro_benchmarks/bench_simd_distance",
                        help="Path to SIMD distance benchmark binary (for baseline)")
    parser.add_argument("--no-simd-baseline", action="store_false", dest="simd_baseline",
                        help="Skip SIMD baseline measurement")
    parser.add_argument("extra", nargs="*",
                        help="Extra args forwarded to benchmark binary (e.g. --benchmark_repetitions=3)")
    args = parser.parse_args()

    # Run benchmark and get output
    output = run_benchmark(args.bin, args.capacities, args.seed, args.extra)
    print(f"[*] Benchmark completed")

    # Parse output
    df = parse_benchmark_output(output)
    if df.empty:
        print("[!] No benchmark data parsed. Check the benchmark output.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Parsed {len(df)} benchmark entries across {df['group'].nunique()} groups")

    # Get SIMD baseline if requested
    simd_baseline = None
    if args.simd_baseline:
        simd_baseline = run_simd_benchmark(args.simd_bin)

    plot_results(df, args.output_dir, simd_baseline)


if __name__ == "__main__":
    main()
