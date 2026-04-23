#!/usr/bin/env python3
# Copyright 2026 Weitang Ye
# Licensed under the Apache License, Version 2.0.
"""
Plot per-hop 1-NN routing profiles produced by artea_profile.cpp.

Reads router_profiler_results.json and draws two curves (hierarchical
artea vs. L0-only) on the same axes:
  x = Average NDC at each hop
  y = log10(ADR - 1) at each hop
Each hop contributes one point; hops with ADR <= 1 are dropped (log is
undefined).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CURVE_LABELS = {
    "hierarchical_artea": "Hierarchical artea",
    "l0_only":            "L0-only (single layer)",
}
CURVE_MARKERS = {
    "hierarchical_artea": "o",
    "l0_only":            "s",
}


def _load_long_form(json_path: Path) -> tuple[pd.DataFrame, dict]:
    """Flatten the per-curve per-hop entries into a long-form DataFrame."""
    with json_path.open() as f:
        payload = json.load(f)

    rows: list[dict] = []
    for curve_key, curve in payload["curves"].items():
        label = CURVE_LABELS.get(curve_key, curve_key)
        for hop_entry in curve["per_hop"]:
            adr = hop_entry["adr"]
            if adr <= 1.0:
                # log10(ADR - 1) is undefined / -inf; skip.
                continue
            rows.append({
                "curve":   label,
                "hop":     hop_entry["hop"],
                "avg_ndc": hop_entry["avg_ndc"],
                "adr":     adr,
                "log_adr_minus_1": math.log10(adr - 1.0),
            })

    df = pd.DataFrame(rows)
    return df, payload


def plot(df: pd.DataFrame, payload: dict, out_path: Path) -> None:
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

    palette = sns.color_palette("deep", n_colors=max(df["curve"].nunique(), 2))

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)

    for (curve_label, curve_df), color in zip(df.groupby("curve"), palette):
        curve_df = curve_df.sort_values("avg_ndc")
        # Find a stable marker key by reverse-looking-up the CURVE_LABELS map.
        marker = "o"
        for key, label in CURVE_LABELS.items():
            if label == curve_label:
                marker = CURVE_MARKERS.get(key, "o")
                break
        ax.plot(
            curve_df["avg_ndc"],
            curve_df["log_adr_minus_1"],
            label=curve_label,
            color=color,
            marker=marker,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=1.8,
            alpha=0.95,
        )

    ax.set_xlabel("Average NDC per query")
    ax.set_ylabel(r"$\log_{10}(\mathrm{ADR} - 1)$")

    dataset = payload.get("dataset", "dataset")
    nq = payload.get("num_queries", "?")
    title = f"Per-hop routing quality — {dataset} ({nq} queries)"
    ax.set_title(title, pad=12)

    ax.legend(
        title=None,
        frameon=False,
        loc="upper right",
        fontsize=11,
    )

    # Tight integer ticks for NDC when reasonable.
    x_max = df["avg_ndc"].max()
    if x_max > 0 and x_max < 1e4:
        ax.set_xlim(left=0.0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote plot to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("vldb27-exp/results/router_profiler_results.json"),
        help="Input JSON emitted by artea_profile.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("vldb27-exp/results/router_profile_adr_vs_ndc.png"),
        help="Output figure path (format inferred from extension).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}")
        return 1

    df, payload = _load_long_form(args.input)
    if df.empty:
        print("ERROR: no plottable points (every hop had ADR <= 1).")
        return 1

    plot(df, payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
