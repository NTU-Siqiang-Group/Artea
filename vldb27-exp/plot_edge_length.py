#!/usr/bin/env python3
# Copyright 2026 Weitang Ye
# Licensed under the Apache License, Version 2.0.
"""
Plot the distribution of edge distances adopted by the hierarchical
router (all layers) and the L0-only single-layer router, from the JSON
dump produced by edge_length_profile.cpp.

Each subplot draws a density-normalized histogram plus a KDE fit, and
annotates the layer's r-net covering radius R_h as a black dashed
vertical line. All subplots share the same x-axis range so the
distributions are directly comparable.

The number of hierarchical layers is read from the JSON at runtime, so
a run with 3 layers produces 3 hier panels (+1 for single_L0) and a
run with 6 layers produces 6 hier panels (+1). Output PDFs are named:
  - edge_length_distribution_hier_L{i}.pdf   (one per hier layer)
  - edge_length_distribution_single_L0.pdf
  - edge_length_distribution_total.pdf       (grid + legend above)
  - edge_length_distribution_legend.pdf      (standalone legend strip)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator


BAR_ALPHA = 0.65
# Lower alpha dedicated to the overlay panel where many hier layers stack
# on the same axes — readers need to see through the front layers.
OVERLAY_BAR_ALPHA = 0.25
OVERLAY_RADIUS_ALPHA = 0.85
BAR_RWIDTH = 0.9
N_BINS = 60
KDE_LINEWIDTH = 1.8
RADIUS_LINEWIDTH = 1.5
RADIUS_COLOR = "black"
RADIUS_LINESTYLE = "--"
# Target number of major x-ticks per panel (denser than matplotlib's ~5 default).
X_TICK_NBINS = 10
FILENAME_PREFIX = "edge_length_distribution"

# Background layer that shows the distribution of *every* graph-stored
# neighbor distance at each hier level (pre-compaction). Overlaid under
# the adopted-edges distribution in every hier panel as transparent
# gray.
NBR_BG_COLOR = "#7F7F7F"
NBR_BG_HIST_ALPHA = 0.22
NBR_BG_KDE_ALPHA = 0.55
NBR_BG_KDE_LINEWIDTH = 1.3

# Color for the L0-only single-layer baseline — kept distinct from the
# hier palette so it always reads as "the other router".
SINGLE_L0_COLOR = "#00A0B0"

# Hier-layer palette: "plasma" goes dark-purple → magenta → orange →
# yellow, which (a) scales to arbitrary layer counts and (b) never
# produces a teal that would clash with SINGLE_L0_COLOR.
HIER_CMAP_NAME = "plasma"


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


def _hier_color(layer_idx: int, n_hier: int):
    cmap = plt.get_cmap(HIER_CMAP_NAME)
    if n_hier <= 1:
        return cmap(0.3)
    return cmap(0.1 + 0.8 * (layer_idx / (n_hier - 1)))


def _build_series(payload: dict) -> list[tuple]:
    """Inspect the JSON payload and produce one series per hierarchical
    layer present plus one for the L0-only baseline. Returned tuples:
    (tag, curve_key, layer_idx, label, color)."""
    hg_levels = (
        payload.get("curves", {}).get("hierarchical_artea", {}).get("edges_by_level", [])
    )
    n_hier = len(hg_levels)

    series: list[tuple] = []
    for i in range(n_hier):
        series.append((
            f"hier_L{i}",
            "hierarchical_artea",
            i,
            f"Hierarchical Router @ L{i}",
            _hier_color(i, n_hier),
        ))
    series.append((
        "single_L0",
        "l0_only",
        0,
        "Single Layer Router @ L0",
        SINGLE_L0_COLOR,
    ))
    return series


def _extract_level(payload: dict, curve_key: str, layer_idx: int) -> dict | None:
    curve = payload.get("curves", {}).get(curve_key)
    if curve is None:
        return None
    levels = curve.get("edges_by_level", [])
    if layer_idx >= len(levels):
        return None
    return levels[layer_idx]


def _extract_lengths(payload: dict, curve_key: str, layer_idx: int) -> np.ndarray:
    level = _extract_level(payload, curve_key, layer_idx)
    if level is None:
        return np.asarray([], dtype=np.float64)
    return np.asarray(level.get("lengths", []), dtype=np.float64)


def _extract_radius(payload: dict, curve_key: str, layer_idx: int) -> float | None:
    level = _extract_level(payload, curve_key, layer_idx)
    if level is None:
        return None
    r = level.get("radius")
    return float(r) if r is not None else None


def _extract_nbr_lengths(payload: dict, layer_idx: int) -> np.ndarray:
    """Fetch the pre-compaction all-graph-edges sample at hier level
    @p layer_idx, if present. Returns an empty array when the JSON
    doesn't carry `nbr_dists_by_level` or when the layer is missing."""
    curve = payload.get("curves", {}).get("hierarchical_artea", {})
    levels = curve.get("nbr_dists_by_level", [])
    if layer_idx >= len(levels):
        return np.asarray([], dtype=np.float64)
    return np.asarray(levels[layer_idx].get("lengths", []), dtype=np.float64)


def _compute_global_xmax(payload: dict, series: list[tuple]) -> float:
    """Max across every series' samples AND every radius marker AND the
    pre-compaction nbr distribution tails — so nothing gets clipped in
    any panel."""
    candidates: list[float] = []
    for _, curve_key, layer_idx, _, _ in series:
        lengths = _extract_lengths(payload, curve_key, layer_idx)
        if lengths.size > 0:
            candidates.append(float(lengths.max()))
        radius = _extract_radius(payload, curve_key, layer_idx)
        if radius is not None:
            candidates.append(radius)
        if curve_key == "hierarchical_artea":
            nbr_lengths = _extract_nbr_lengths(payload, layer_idx)
            if nbr_lengths.size > 0:
                candidates.append(float(nbr_lengths.max()))
    if not candidates:
        return 1.0
    return max(candidates) * 1.05   # small padding so the radius line isn't flush against the right edge


def _legend_handles(series: list[tuple]) -> list:
    patches = [
        mpatches.Patch(facecolor=color, edgecolor=color, alpha=BAR_ALPHA, label=label)
        for _, _, _, label, color in series
    ]
    nbr_bg_handle = mpatches.Patch(
        facecolor=NBR_BG_COLOR,
        edgecolor=NBR_BG_COLOR,
        alpha=NBR_BG_HIST_ALPHA,
        label="All graph edges @ $L_h$ (pre-compaction)",
    )
    radius_handle = mlines.Line2D(
        [], [],
        color=RADIUS_COLOR,
        linestyle=RADIUS_LINESTYLE,
        linewidth=RADIUS_LINEWIDTH,
        label=r"r-net radius $R_h$",
    )
    return patches + [nbr_bg_handle, radius_handle]


def _draw_panel(
    ax: plt.Axes,
    lengths: np.ndarray,
    radius: float | None,
    color,
    xmax: float,
    nbr_lengths: np.ndarray | None = None,
) -> None:
    """Histogram + KDE for @p lengths, with an optional transparent-gray
    background histogram + KDE for @p nbr_lengths (all pre-compaction
    graph edges at this layer). Radius line overlaid on top when @p
    radius is a finite positive number. zorder is set explicitly so the
    main series always paints on top of the gray background."""
    # Background: full graph-edge distribution, transparent gray. Drawn
    # first so the foreground series sits on top.
    if nbr_lengths is not None and nbr_lengths.size > 0:
        ax.hist(
            nbr_lengths,
            bins=N_BINS,
            density=True,
            color=NBR_BG_COLOR,
            alpha=NBR_BG_HIST_ALPHA,
            edgecolor=NBR_BG_COLOR,
            linewidth=0.3,
            rwidth=BAR_RWIDTH,
            zorder=1,
        )
        sns.kdeplot(
            x=nbr_lengths,
            ax=ax,
            color=NBR_BG_COLOR,
            linewidth=NBR_BG_KDE_LINEWIDTH,
            alpha=NBR_BG_KDE_ALPHA,
            bw_adjust=1.0,
            clip=(0.0, None),
            zorder=2,
        )

    if lengths.size == 0:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(
            lengths,
            bins=N_BINS,
            density=True,
            color=color,
            alpha=BAR_ALPHA,
            edgecolor=color,
            linewidth=0.6,
            rwidth=BAR_RWIDTH,
            zorder=3,
        )
        # KDE overlay — same series color, full alpha so the curve reads
        # cleanly against the semi-transparent bars.
        sns.kdeplot(
            x=lengths,
            ax=ax,
            color=color,
            linewidth=KDE_LINEWIDTH,
            alpha=1.0,
            bw_adjust=1.0,
            clip=(0.0, None),
            zorder=4,
        )

    if radius is not None and radius > 0.0:
        ax.axvline(
            radius,
            color=RADIUS_COLOR,
            linestyle=RADIUS_LINESTYLE,
            linewidth=RADIUS_LINEWIDTH,
            zorder=5,
        )
        # Small in-axis numeric annotation near the top so readers can
        # read off the value without a separate legend entry per layer.
        ax.annotate(
            f"$R_h$={radius:.1f}",
            xy=(radius, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9,
            color=RADIUS_COLOR,
        )

    ax.set_xlim(0.0, xmax)
    ax.set_xlabel(r"Edge distance  $d(u, v)$")
    ax.set_ylabel("Density")
    # Flexible scientific notation: tick values with |x| >= 1e3 (or very
    # small) switch to a compact "a × 10^k" form so long integers don't
    # overflow the axis.
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3), useMathText=True)
    # Denser x-axis ticks than matplotlib's default.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=X_TICK_NBINS))


def _draw_overlay_panel(
    ax: plt.Axes,
    hier_series: list[tuple],
    payload: dict,
    xmax: float,
) -> None:
    """Overlay every hier layer on a single axes. Uses a lower histogram
    alpha so stacked distributions remain legible, and draws each layer's
    r-net radius in that layer's own color (dashed) — the common black
    dashed style would collapse all radii into one visual line otherwise.
    Carries an in-panel legend so readers can tell layers apart without
    referring to the strip above the grid."""
    for _, curve_key, layer_idx, label, color in hier_series:
        lengths = _extract_lengths(payload, curve_key, layer_idx)
        radius = _extract_radius(payload, curve_key, layer_idx)
        if lengths.size > 0:
            ax.hist(
                lengths,
                bins=N_BINS,
                density=True,
                color=color,
                alpha=OVERLAY_BAR_ALPHA,
                edgecolor=color,
                linewidth=0.4,
                rwidth=BAR_RWIDTH,
                label=label,
            )
            sns.kdeplot(
                x=lengths,
                ax=ax,
                color=color,
                linewidth=KDE_LINEWIDTH,
                alpha=1.0,
                bw_adjust=1.0,
                clip=(0.0, None),
            )
        if radius is not None and radius > 0.0:
            ax.axvline(
                radius,
                color=color,
                linestyle=RADIUS_LINESTYLE,
                linewidth=1.2,
                alpha=OVERLAY_RADIUS_ALPHA,
            )

    ax.set_xlim(0.0, xmax)
    ax.set_xlabel(r"Edge distance  $d(u, v)$")
    ax.set_ylabel("Density")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3), useMathText=True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=X_TICK_NBINS))
    ax.set_title("All hierarchical layers overlaid", pad=6, fontsize=11)
    ax.legend(fontsize=8, loc="upper right", frameon=False)


def plot_individual(payload: dict, entry: tuple, xmax: float, out_path: Path) -> None:
    _, curve_key, layer_idx, label, color = entry
    lengths = _extract_lengths(payload, curve_key, layer_idx)
    radius = _extract_radius(payload, curve_key, layer_idx)
    # The pre-compaction nbr-distance bucket lives under the hier curve;
    # reuse it for the single_L0 panel too — same L0 edges, just a
    # different routing strategy on top.
    nbr_lengths = _extract_nbr_lengths(payload, layer_idx)

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=150)
    _draw_panel(ax, lengths, radius, color, xmax, nbr_lengths=nbr_lengths)
    if lengths.size > 0:
        ax.set_title(f"{label}  (n={lengths.size})", pad=10)
    else:
        ax.set_title(label, pad=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _grid_shape(n_panels: int) -> tuple[int, int]:
    """Fixed 2-columns layout; rows grow with panel count. Two panels per
    row keeps each axes wide enough that the denser x-tick labels don't
    crowd into each other."""
    ncols = 2
    nrows = max(1, math.ceil(n_panels / ncols))
    return nrows, ncols


def plot_total(payload: dict, series: list[tuple], xmax: float, out_path: Path) -> None:
    """Grid of panels — one per series, plus one final overlay panel that
    stacks every hier layer on the same axes. Fixed 3 panels per row; row
    count follows the layer count. Common x-range + legend strip above."""
    hier_series = [s for s in series if s[1] == "hierarchical_artea"]
    # Total panels = per-series individual panels + 1 overlay.
    n = len(series) + 1
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 6.0, nrows * 3.4),
        dpi=150,
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, entry in zip(axes_flat, series):
        _, curve_key, layer_idx, label, color = entry
        lengths = _extract_lengths(payload, curve_key, layer_idx)
        radius = _extract_radius(payload, curve_key, layer_idx)
        # Same L0 edges underlie both hier_L0 and single_L0 — draw the
        # all-graph-edges background on every panel, hier or otherwise.
        nbr_lengths = _extract_nbr_lengths(payload, layer_idx)
        _draw_panel(ax, lengths, radius, color, xmax, nbr_lengths=nbr_lengths)
        if lengths.size > 0:
            ax.set_title(f"{label}  (n={lengths.size})", pad=6, fontsize=11)
        else:
            ax.set_title(label, pad=6, fontsize=11)

    # Overlay panel right after the last per-series panel.
    _draw_overlay_panel(axes_flat[len(series)], hier_series, payload, xmax)

    # Hide any empty axes on the last row.
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Reserve space at the top for the legend strip.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    handles = _legend_handles(series)
    # Keep legend columns <= 5 so text doesn't cramp; matplotlib auto-
    # wraps remaining entries onto a second row.
    legend_ncols = min(len(handles), 5)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=legend_ncols,
        frameon=False,
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_legend_only(series: list[tuple], out_path: Path) -> None:
    """Standalone wide legend strip — same visual as the total-figure
    strip; includes one patch per series and the radius line. Width and
    row count scale with the number of handles."""
    handles = _legend_handles(series)
    ncols = min(len(handles), 5)
    nrows = math.ceil(len(handles) / ncols)
    fig = plt.figure(figsize=(max(11.0, ncols * 2.4), 0.55 * nrows), dpi=150)
    fig.legend(
        handles=handles,
        loc="center",
        ncol=ncols,
        frameon=False,
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("vldb27-exp/results/edge_length_profile_results.json"),
        help="Input JSON emitted by edge_length_profile.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("vldb27-exp/results"),
        help="Directory where the PDF files are written.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}")
        return 1

    with args.input.open() as f:
        payload = json.load(f)

    _apply_theme()

    series = _build_series(payload)
    if not series:
        print("ERROR: no series extracted from payload (empty edges_by_level?)")
        return 1

    xmax = _compute_global_xmax(payload, series)
    out_dir: Path = args.output_dir
    for entry in series:
        tag = entry[0]
        plot_individual(payload, entry, xmax, out_dir / f"{FILENAME_PREFIX}_{tag}.pdf")

    plot_total(payload, series, xmax, out_dir / f"{FILENAME_PREFIX}_total.pdf")
    plot_legend_only(series, out_dir / f"{FILENAME_PREFIX}_legend.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
