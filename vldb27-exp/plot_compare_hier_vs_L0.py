#!/usr/bin/env python3
# Copyright 2026 Weitang Ye
# Licensed under the Apache License, Version 2.0.
"""
Plot Recall-vs-QPS and ADR-vs-QPS curves produced by compare_hier_vs_L0.

Input JSON shape:
    {
      "dataset": "sift-1m",
      "topk": 1,
      "rows": [
        {"queue_size": 10,
         "hier": {"recall": ..., "adr": ..., "qps": ..., "batch_ms": ...},
         "l0":   {"recall": ..., "adr": ..., "qps": ..., "batch_ms": ...}},
        ...
      ]
    }

ADR (Average Distance Ratio, for 1-NN) is defined as
    ADR = (1/|Q|) * Σ_q  d(q, retrieved_q) / d(q, true_NN_q)
where `retrieved_q` is the top-1 returned by the router and `true_NN_q`
is the ground-truth nearest neighbour. ADR is >= 1, with 1.0 meaning
exact retrieval — so lower ADR is better.

Emits two PDFs into --output-dir:
  - compare_hier_vs_L0_recall.pdf  : x = Recall      (linear),        y = QPS (linear)
  - compare_hier_vs_L0_adr.pdf     : x = ADR - 1     (log, inverted), y = QPS (linear)
The ADR x-axis is inverted so left→right corresponds to search quality
improving (i.e., ADR - 1 shrinks toward 0). Both plots apply
Pareto-frontier filtering to drop dominated points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


# (json_key, display_label, color, marker)
SERIES = [
    ("hier", "Hierarchical Router",         "#E11382", "*"),
    ("l0",   "L0-only Single-Layer Router", "#00A0B0", "o"),
]

MARKERSIZE = 9
LINEWIDTH = 1.8
ALPHA = 0.85
_ADR_EPS = 1e-6   # treat ADR within eps of 1.0 as exact and drop the point on the log-scale plot


def _apply_theme() -> None:
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


# =============================================================================
#  Pareto frontier helpers
# =============================================================================

def _pareto_recall_qps(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """(recall, qps) frontier: higher recall AND higher QPS dominates."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    frontier: list[tuple[float, float]] = []
    max_qps = float("-inf")
    for recall, qps in reversed(sorted_pts):
        if qps > max_qps:
            frontier.append((recall, qps))
            max_qps = qps
    frontier.reverse()
    return frontier


def _pareto_adr_qps(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """(adr, qps) frontier: lower ADR AND higher QPS dominates. Returned
    points are sorted by QPS ascending so downstream line plots trace
    left-to-right along the x axis."""
    if not points:
        return []
    # Reduce to (-adr, qps) so "higher-is-better" sweep works uniformly.
    negated = [(-a, q) for (a, q) in points]
    sorted_neg = sorted(negated, key=lambda p: (p[0], p[1]))
    frontier_neg: list[tuple[float, float]] = []
    max_qps = float("-inf")
    for neg_adr, qps in reversed(sorted_neg):
        if qps > max_qps:
            frontier_neg.append((neg_adr, qps))
            max_qps = qps
    # Un-negate and order by QPS ascending.
    frontier = [(-a, q) for (a, q) in frontier_neg]
    frontier.sort(key=lambda p: p[1])
    return frontier


# =============================================================================
#  Shared helpers
# =============================================================================

def _apply_qps_tick_formatter(ax, axis: str, qps_values):
    """Fold the power-of-ten factor into a "QPS (×10^k)" axis label with
    integer-mantissa ticks. Lifted from the project's plot_accuracy.py."""
    qps_max = max(qps_values) if qps_values else 0
    if qps_max < 100:
        return "QPS"

    default_exp = 3 if qps_max < 20000 else int(np.floor(np.log10(qps_max)))
    chosen = None
    for exp_cand in (default_exp, default_exp - 1, default_exp + 1):
        if exp_cand < 0:
            continue
        for k in (1, 2, 5):
            step = k * 10 ** exp_cand
            n_ticks = int(qps_max // step) + 1
            if 5 <= n_ticks <= 10:
                chosen = (exp_cand, step)
                break
        if chosen:
            break
    if chosen is None:
        chosen = (default_exp, 10 ** default_exp)
    exp, step = chosen

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(ticker.MultipleLocator(step))
    target.set_major_formatter(ticker.FuncFormatter(
        lambda val, pos, e=exp: f"{int(round(val / 10 ** e))}"
    ))
    return rf"QPS ($\times 10^{{{exp}}}$)"


def _extract_series(rows: list[dict], key: str, metric: str) -> list[tuple[float, float]]:
    """Collect (metric_value, qps) tuples from @p rows for one series,
    dropping rows with non-positive QPS."""
    out = []
    for r in rows:
        sub = r.get(key)
        if sub is None:
            continue
        metric_val = sub.get(metric)
        qps = sub.get("qps")
        if metric_val is None or qps is None:
            continue
        if qps <= 0.0:
            continue
        out.append((float(metric_val), float(qps)))
    return out


# =============================================================================
#  Recall vs QPS
# =============================================================================

def plot_recall_vs_qps(data: dict, out_path: Path) -> None:
    rows = data.get("rows", [])
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=150)

    all_qps: list[float] = []
    for key, label, color, marker in SERIES:
        pts = _extract_series(rows, key, "recall")
        pts = _pareto_recall_qps(pts)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        all_qps.extend(ys)
        ax.plot(
            xs, ys,
            marker=marker, color=color, linestyle="-",
            linewidth=LINEWIDTH, alpha=ALPHA,
            markersize=MARKERSIZE,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.6,
            label=label,
        )

    ylabel = _apply_qps_tick_formatter(ax, "y", all_qps)
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    dataset = data.get("dataset", "?")
    topk    = data.get("topk", "?")
    ax.set_title(f"Recall–QPS  ({dataset}, top-{topk})", fontsize=13, pad=8)
    ax.legend(frameon=True, fancybox=True, fontsize=12, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# =============================================================================
#  ADR vs QPS
#    x = ADR - 1 (log10), axis inverted so higher-quality (smaller ADR - 1)
#                         sits on the right — reading left→right traces
#                         "search quality improving".
#    y = QPS    (linear, folded ×10^k label)
#    Drop points with ADR ≤ 1 + eps (unrepresentable on log axis).
# =============================================================================

def plot_adr_vs_qps(data: dict, out_path: Path) -> None:
    rows = data.get("rows", [])
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=150)

    all_qps: list[float] = []
    for key, label, color, marker in SERIES:
        pts = _extract_series(rows, key, "adr")
        # Drop points where ADR ~= 1 (exact 1-NN, can't place on log axis).
        pts = [(a, q) for (a, q) in pts if (a - 1.0) > _ADR_EPS]
        pts = _pareto_adr_qps(pts)
        if not pts:
            continue

        # Sort by ADR descending so the polyline is drawn from the worst
        # (leftmost after axis inversion) to the best (rightmost).
        pts.sort(key=lambda p: -p[0])
        adr_minus_1 = [p[0] - 1.0 for p in pts]
        qps_vals    = [p[1] for p in pts]
        all_qps.extend(qps_vals)

        ax.plot(
            adr_minus_1, qps_vals,
            marker=marker, color=color, linestyle="-",
            linewidth=LINEWIDTH, alpha=ALPHA,
            markersize=MARKERSIZE,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.6,
            label=label,
        )

    ax.set_xscale("log")
    # Invert the x-axis so small (ADR - 1) → right. Reading left→right
    # now corresponds to search quality improving.
    ax.invert_xaxis()
    ylabel = _apply_qps_tick_formatter(ax, "y", all_qps)
    ax.set_xlabel(r"ADR $-$ 1  (smaller is better →)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    dataset = data.get("dataset", "?")
    topk    = data.get("topk", "?")
    ax.set_title(f"ADR–QPS  ({dataset}, top-{topk})", fontsize=13, pad=8)
    ax.legend(frameon=True, fancybox=True, fontsize=12, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# =============================================================================
#  Entry point
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-recall",
        type=Path,
        default=Path("vldb27-exp/results/compare_hier_vs_L0_recall_results.json"),
        help="JSON produced by the recall-sweep run of compare_hier_vs_L0.",
    )
    parser.add_argument(
        "--input-adr",
        type=Path,
        default=Path("vldb27-exp/results/compare_hier_vs_L0_adr_results.json"),
        help="JSON produced by the ADR-sweep run of compare_hier_vs_L0.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("vldb27-exp/results"),
        help="Directory where the two PDFs are written.",
    )
    args = parser.parse_args()

    missing = [p for p in (args.input_recall, args.input_adr) if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: input not found: {p}")
        return 1

    with args.input_recall.open() as f:
        data_recall = json.load(f)
    with args.input_adr.open() as f:
        data_adr = json.load(f)

    _apply_theme()

    plot_recall_vs_qps(data_recall, args.output_dir / "compare_hier_vs_L0_recall.pdf")
    plot_adr_vs_qps(data_adr,       args.output_dir / "compare_hier_vs_L0_adr.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
