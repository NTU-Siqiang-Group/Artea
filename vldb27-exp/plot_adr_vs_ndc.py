#!/usr/bin/env python3
# Copyright 2026 Weitang Ye
# Licensed under the Apache License, Version 2.0.
"""
Plot per-query 1-NN routing trajectories produced by
adr_vs_ndc_profile.cpp.

Reads the JSON dump and draws, on the same axes, every sampled query's
(NDC, log2(ADR - 1)) trajectory as a thin, semi-transparent line. Each
of the two curves (hierarchical artea vs. L0-only) is drawn in its own
color, so overlapping families of trajectories remain visually
separable. No aggregation is performed — each sampled query produces
one line.

  x = cumulative NDC at each successful cursor move
  y = log2(ADR - 1), where ADR = d(q, best) / d(q, true_nn)

The @c log2(ADR - 1) transform spreads the "approach to the true NN"
region across the negative axis with finer resolution than log10:
ADR=2 maps to 0, ADR=1.5 to -1, ADR=1.25 to -2, ADR=1.125 to -3, etc.
Points with @c ADR == 1 (exact convergence) are dropped since the log
is undefined there.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


CURVE_LABELS = {
    "hierarchical_artea": "Hierarchical artea",
    "l0_only":            "L0-only (single layer)",
}
# Keep the iteration order deterministic for consistent color assignment.
CURVE_ORDER = ["hierarchical_artea", "l0_only"]
# Explicit palette: magenta for curve 1, teal for curve 2.
CURVE_COLORS = {
    "hierarchical_artea": "#E11382",   # magenta / pink
    "l0_only":            "#00A0B0",   # cyan / teal
}
CURVE_ALPHA = 0.3


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

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)

    curves = payload.get("curves", {})
    for curve_key in CURVE_ORDER:
        curve = curves.get(curve_key)
        if curve is None:
            continue
        label = CURVE_LABELS.get(curve_key, curve_key)
        color = CURVE_COLORS.get(curve_key, "#888888")

        # Draw each query's trajectory as its own thin translucent line.
        # Only the first line of each curve carries the legend label so the
        # legend stays a single entry per curve.
        first = True
        for traj in curve.get("trajectories", []):
            points = traj.get("points", [])
            if not points:
                continue   # skipped query (empty trajectory)
            # Transform y to log2(ADR - 1); drop points where ADR <= 1
            # (log undefined — typically the final "exact convergence"
            # point at most).
            xs = []
            ys = []
            for p in points:
                adr = p["adr"]
                if adr <= 1.0:
                    continue
                xs.append(p["ndc"])
                ys.append(math.log2(adr - 1.0))
            if not xs:
                continue
            ax.plot(
                xs, ys,
                color=color,
                linewidth=0.8,
                alpha=CURVE_ALPHA,
                label=label if first else None,
            )
            first = False

    ax.set_xlabel("Cumulative NDC per query")
    ax.set_ylabel(r"$\log_{2}(\mathrm{ADR} - 1)$")

    # Integer ticks every 1 octave across the (typically negative) range
    # of log2(ADR - 1). Matplotlib's default picker already handles
    # mixed positive/negative values, but force a MultipleLocator to
    # make the negative octaves read cleanly.
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

    dataset   = payload.get("dataset", "dataset")
    n_sampled = payload.get("num_sampled", "?")
    title     = f"Per-query routing trajectories — {dataset} ({n_sampled} sampled queries)"
    ax.set_title(title, pad=12)

    # y == 0 corresponds to ADR == 2 (best_dist = 2 × true_nn_dist).
    # Mark it as a reference threshold for "close but not converged".
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    leg = ax.legend(title=None, frameon=False, loc="upper right", fontsize=11)
    # Undo the low alpha on the legend line handles so they stay readable.
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
        handle.set_linewidth(2.0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote plot to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("vldb27-exp/results/adr_vs_ndc_profile_results.json"),
        help="Input JSON emitted by adr_vs_ndc_profile.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("vldb27-exp/results/adr_vs_ndc_per_query.pdf"),
        help="Output figure path (format inferred from extension; defaults to vector PDF).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}")
        return 1

    with args.input.open() as f:
        payload = json.load(f)

    plot(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
