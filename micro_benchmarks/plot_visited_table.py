#!/usr/bin/env python3
"""
Run bench_visited_table and plot throughput (items/s) for clear/set/test/mixed
operations across ThreadLocalBitmap and VersionTagTable, with optional SIMD
distance baseline.

Usage:
    # Default settings
    python plot_visited_table.py

    # Custom table sizes
    python plot_visited_table.py -t 10000,100000,1000000,10000000

    # Custom binary
    python plot_visited_table.py --bin ./build/micro_benchmarks/bench_visited_table

    # Skip SIMD baseline
    python plot_visited_table.py --no-simd-baseline

    # Custom num_queries / visit_count
    python plot_visited_table.py -n 2048 -v 512

    # Pass extra args through to the benchmark binary
    python plot_visited_table.py --benchmark_repetitions=3 --benchmark_min_time=1.0
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
TABLE_PALETTE = {
    "Bitmap":     "#4C72B0",   # muted blue
    "VersionTag": "#DD8452",   # warm orange
}
TABLE_MARKERS = {
    "Bitmap":     "o",
    "VersionTag": "s",
}

# ── Throughput groups (items/s) ──────────────────────────────────────────────
THROUGHPUT_GROUPS = {
    "Clear": {"prefix": "BM_Clear_", "title": "clear (per-query)"},
    "Set":   {"prefix": "BM_Set_",   "title": "set (per visited vertex)"},
    "Test":  {"prefix": "BM_Test_",  "title": "test (per lookup)"},
    "Mixed": {"prefix": "BM_Mixed_", "title": "test-then-set (beam-search pattern)"},
}

# ── Latency groups (ns/op) ───────────────────────────────────────────────────
LATENCY_GROUPS = {
    "Latency_Set":   {"prefix": "BM_Latency_Set_",   "title": "set latency (single-thread)"},
    "Latency_Test":  {"prefix": "BM_Latency_Test_",  "title": "test latency (single-thread)"},
    "Latency_Clear": {"prefix": "BM_Latency_Clear_", "title": "clear latency (single-thread)"},
}

TABLE_SUFFIX_MAP = {
    "Bitmap":     "Bitmap",
    "VersionTag": "VersionTag",
}


def run_benchmark(bin_path: str, table_sizes: str, num_queries: int,
                  visit_count: int, seed: int, extra_args: list) -> str:
    cmd = [
        bin_path,
        "-s", str(seed),
        "-n", str(num_queries),
        "-v", str(visit_count),
        "-t", table_sizes,
    ] + extra_args

    print(f"[*] Running: {' '.join(cmd)}")
    print("[*] Benchmark output:")
    print("-" * 80)

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)

    process.wait()
    print("-" * 80)

    if process.returncode != 0:
        stderr = process.stderr.read()
        print(f"[!] Benchmark exited with code {process.returncode}", file=sys.stderr)
        print(stderr, file=sys.stderr)
        sys.exit(1)

    return ''.join(output_lines)


def parse_throughput(output: str) -> pd.DataFrame:
    """Parse items_per_second from throughput benchmark rows."""
    rows = []
    pattern = (
        r'^(BM_\w+/\d+)\s+[\d.]+\s+[a-z]+\s+[\d.]+\s+[a-z]+\s+\d+\s+'
        r'items_per_second=([\d.]+)([KMG]?)/s'
    )
    multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, '': 1.0}

    for line in output.split('\n'):
        match = re.search(pattern, line.strip())
        if not match:
            continue
        name = match.group(1)
        items_per_second = float(match.group(2)) * multipliers.get(match.group(3), 1.0)

        parts = name.rsplit("/", 1)
        if len(parts) != 2:
            continue
        func_name, size_str = parts
        size = int(size_str)

        group_key, table_name = None, None
        for gk, gv in THROUGHPUT_GROUPS.items():
            if func_name.startswith(gv["prefix"]):
                # Avoid matching BM_Clear_ when the real row is BM_Latency_Clear_
                if "Latency" in func_name:
                    continue
                suffix = func_name[len(gv["prefix"]):]
                if suffix in TABLE_SUFFIX_MAP:
                    group_key = gk
                    table_name = TABLE_SUFFIX_MAP[suffix]
                break

        if group_key is None:
            continue

        rows.append({
            "group": group_key,
            "table": table_name,
            "size":  size,
            "items_per_second": items_per_second,
        })

    return pd.DataFrame(rows)


def parse_latency(output: str) -> pd.DataFrame:
    """Parse per-op time (ns) from latency benchmark rows.

    Google Benchmark row layout (with Unit(kNanosecond)):
        BM_Latency_Set_Bitmap/1000000     1.23 ns    1.23 ns    500000000
    """
    rows = []
    pattern = (
        r'^(BM_Latency_\w+/\d+)\s+([\d.]+)\s+ns\s+[\d.]+\s+ns\s+\d+'
    )

    for line in output.split('\n'):
        match = re.search(pattern, line.strip())
        if not match:
            continue
        name = match.group(1)
        ns_per_op = float(match.group(2))

        parts = name.rsplit("/", 1)
        if len(parts) != 2:
            continue
        func_name, size_str = parts
        size = int(size_str)

        group_key, table_name = None, None
        for gk, gv in LATENCY_GROUPS.items():
            if func_name.startswith(gv["prefix"]):
                suffix = func_name[len(gv["prefix"]):]
                if suffix in TABLE_SUFFIX_MAP:
                    group_key = gk
                    table_name = TABLE_SUFFIX_MAP[suffix]
                break

        if group_key is None:
            continue

        rows.append({
            "group": group_key,
            "table": table_name,
            "size":  size,
            "ns_per_op": ns_per_op,
            "items_per_second": 1e9 / ns_per_op if ns_per_op > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def run_simd_benchmark(bin_path: str) -> float:
    if not os.path.exists(bin_path):
        print(f"[!] SIMD benchmark binary not found: {bin_path}", file=sys.stderr)
        return 0.0

    cmd = [bin_path]
    print(f"\n[*] Running SIMD baseline: {' '.join(cmd)}")
    print("[*] SIMD benchmark output:")
    print("-" * 80)

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)

    process.wait()
    print("-" * 80)

    if process.returncode != 0:
        stderr = process.stderr.read()
        print(f"[!] SIMD benchmark failed with code {process.returncode}", file=sys.stderr)
        print(stderr, file=sys.stderr)
        return 0.0

    output = ''.join(output_lines)

    pattern = r'^\w+\s+[\d.]+\s+[a-z]+\s+[\d.]+\s+[a-z]+\s+\d+\s+items_per_second=([\d.]+)([KMG]?)/s'
    multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, '': 1.0}
    values = []
    for line in output.split('\n'):
        match = re.search(pattern, line.strip())
        if match:
            values.append(float(match.group(1)) * multipliers.get(match.group(2), 1.0))

    if not values:
        print("[!] No items_per_second found in SIMD benchmark output", file=sys.stderr)
        return 0.0

    avg = float(np.mean(values))
    print(f"[*] SIMD baseline: {avg/1e6:.2f} M items/s (avg of {len(values)} benchmarks)")
    return avg


def plot_throughput(df: pd.DataFrame, output_dir: str, simd_baseline: float = None):
    """Four throughput panels (clear/set/test/mixed)."""
    if df.empty:
        print("[!] No throughput data to plot", file=sys.stderr)
        return

    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("VisitedTable Throughput Benchmark (items/s)",
                 fontsize=18, fontweight="bold", y=1.02)

    for col_idx, (gk, gv) in enumerate(THROUGHPUT_GROUPS.items()):
        sub = df[df["group"] == gk]
        ax = axes[col_idx]

        for tname in ["Bitmap", "VersionTag"]:
            tsub = sub[sub["table"] == tname].sort_values("size")
            if tsub.empty:
                continue
            throughput_m = tsub["items_per_second"] / 1e6
            ax.plot(
                tsub["size"], throughput_m,
                marker=TABLE_MARKERS[tname], color=TABLE_PALETTE[tname],
                label=tname, linewidth=2.5, markersize=8,
            )

        if simd_baseline is not None and simd_baseline > 0 and not sub.empty:
            simd_m = simd_baseline / 1e6
            ax.axhline(y=simd_m, color='red', linestyle='--', linewidth=2,
                       label=f'SIMD Distance ({simd_m:.1f} M/s)', alpha=0.7)

        ax.set_xscale("log", base=10)
        ax.set_yscale("log")
        ax.set_xlabel("Table size (N vertices)", fontsize=12)
        ax.set_ylabel("Throughput (M items/s)", fontsize=12)
        ax.set_title(gv['title'], fontsize=14, fontweight="bold")
        ax.legend(frameon=True, fancybox=True, shadow=True, loc='best')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "visited_table_throughput.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[*] Throughput plot saved to {out_path}")
    plt.close(fig)


def plot_latency(df: pd.DataFrame, output_dir: str, simd_baseline: float = None):
    """Three latency panels (set/test/clear)."""
    if df.empty:
        print("[!] No latency data to plot", file=sys.stderr)
        return

    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("VisitedTable Single-Thread Latency (ns/op)",
                 fontsize=18, fontweight="bold", y=1.02)

    for col_idx, (gk, gv) in enumerate(LATENCY_GROUPS.items()):
        sub = df[df["group"] == gk]
        ax = axes[col_idx]

        for tname in ["Bitmap", "VersionTag"]:
            tsub = sub[sub["table"] == tname].sort_values("size")
            if tsub.empty:
                continue
            ax.plot(
                tsub["size"], tsub["ns_per_op"],
                marker=TABLE_MARKERS[tname], color=TABLE_PALETTE[tname],
                label=tname, linewidth=2.5, markersize=8,
            )

        if simd_baseline is not None and simd_baseline > 0 and not sub.empty:
            # Express SIMD throughput as equivalent ns/op for an inverse comparison
            simd_ns = 1e9 / simd_baseline
            ax.axhline(y=simd_ns, color='red', linestyle='--', linewidth=2,
                       label=f'SIMD Distance ({simd_ns:.2f} ns/op)', alpha=0.7)

        ax.set_xscale("log", base=10)
        ax.set_yscale("log")
        ax.set_xlabel("Table size (N vertices)", fontsize=12)
        ax.set_ylabel("Time per op (ns)", fontsize=12)
        ax.set_title(gv['title'], fontsize=14, fontweight="bold")
        ax.legend(frameon=True, fancybox=True, shadow=True, loc='best')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "visited_table_latency.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[*] Latency plot saved to {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run and plot VisitedTable benchmarks")
    parser.add_argument("--bin", default="./build/micro_benchmarks/bench_visited_table",
                        help="Path to benchmark binary")
    parser.add_argument("-t", "--table-sizes",
                        default="10000,100000,1000000,10000000",
                        help="Comma-separated list of table sizes (N vertices)")
    parser.add_argument("-n", "--num-queries", type=int, default=1024,
                        help="Number of queries per benchmark iteration")
    parser.add_argument("-v", "--visit-count", type=int, default=256,
                        help="Number of vertices visited per query")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("-o", "--output-dir", default="./micro_benchmarks/plots",
                        help="Directory to save plots")
    parser.add_argument("--simd-bin",
                        default="./build/micro_benchmarks/bench_simd_distance",
                        help="Path to SIMD distance benchmark binary (for baseline)")
    parser.add_argument("--no-simd-baseline", action="store_false",
                        dest="simd_baseline",
                        help="Skip SIMD baseline measurement")
    parser.add_argument("extra", nargs="*",
                        help="Extra args forwarded to benchmark binary")
    args = parser.parse_args()

    output = run_benchmark(args.bin, args.table_sizes, args.num_queries,
                           args.visit_count, args.seed, args.extra)
    print("[*] Benchmark completed")

    tp_df = parse_throughput(output)
    lat_df = parse_latency(output)
    print(f"[*] Parsed {len(tp_df)} throughput rows, {len(lat_df)} latency rows")

    simd_baseline = None
    if args.simd_baseline:
        simd_baseline = run_simd_benchmark(args.simd_bin)

    plot_throughput(tp_df, args.output_dir, simd_baseline)
    plot_latency(lat_df, args.output_dir, simd_baseline)


if __name__ == "__main__":
    main()
