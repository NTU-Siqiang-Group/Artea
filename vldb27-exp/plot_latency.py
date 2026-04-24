#!/usr/bin/env python3
# Copyright 2026 Weitang Ye
# Licensed under the Apache License, Version 2.0.
"""
Plot pXX latency comparisons produced by latency_profile.cpp.

Reads the JSON dump and draws a grouped bar chart: each group is one of
{p50, p90, p95, p99}, and within each group there are two bars —
Hierarchical Router vs. Single Layer Router — showing their measured
latency in microseconds. One dataset → one figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


CURVE_LABELS = {
    "hierarchical_artea": "Hierarchical Router",
    "l0_only":            "Single Layer Router",
}
CURVE_ORDER = ["hierarchical_artea", "l0_only"]
CURVE_COLORS = {
    "hierarchical_artea": "#E11382",   # magenta / pink
    "l0_only":            "#00A0B0",   # cyan / teal
}
PERCENTILES = [
    ("p50_us", "p50"),
    ("p90_us", "p90"),
    ("p95_us", "p95"),
    ("p99_us", "p99"),
]


def plot(payload: dict, out_path: Path) -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.25,
        rc={
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "axes.linewidth":    1.1,
            "grid.linewidth":    0.6,
            "grid.alpha":        0.4,
        },
    )

    latency = payload.get("latency_us", {})
    group_labels = [pct_label for _, pct_label in PERCENTILES]
    n_groups = len(group_labels)
    n_bars   = len(CURVE_ORDER)

    x = np.arange(n_groups)
    bar_width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)

    for i, curve_key in enumerate(CURVE_ORDER):
        curve = latency.get(curve_key)
        if curve is None:
            continue
        heights = [curve.get(key, 0.0) for key, _ in PERCENTILES]
        offsets = x + (i - (n_bars - 1) / 2.0) * bar_width
        bars = ax.bar(
            offsets, heights,
            width=bar_width,
            color=CURVE_COLORS.get(curve_key, "#888888"),
            edgecolor="white",
            linewidth=0.8,
            label=CURVE_LABELS.get(curve_key, curve_key),
        )
        # Annotate each bar with the numeric value (µs).
        for rect, h in zip(bars, heights):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(),
                f"{h:.1f}",
                ha="center", va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_xlabel("Latency percentile")
    ax.set_ylabel(r"Latency ($\mu$s)")

    dataset = payload.get("dataset", "dataset")
    nq      = payload.get("total_queries", "?")
    qs      = payload.get("candidate_queue_size", "?")
    ax.set_title(f"1-NN routing latency — {dataset} (n={nq}, beam={qs})", pad=12)

    ax.legend(title=None, frameon=False, loc="upper left", fontsize=11)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote plot to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("vldb27-exp/results/latency_profile_results.json"),
        help="Input JSON emitted by latency_profile.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("vldb27-exp/results"),
        help="Directory the PDF is written to. Filename is auto-generated "
             "as '<dataset>_1nn_0.9recall_latency.pdf'.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}")
        return 1

    with args.input.open() as f:
        payload = json.load(f)

    # Build a dataset-aware filename. The '0.9recall' tag reflects the
    # operating point this experiment is conventionally reported at;
    # keep it in the filename so measurements at other recall targets
    # don't overwrite it.
    dataset = str(payload.get("dataset", "dataset"))
    out_path = args.output_dir / f"{dataset}_1nn_0.9recall_latency.pdf"

    plot(payload, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
