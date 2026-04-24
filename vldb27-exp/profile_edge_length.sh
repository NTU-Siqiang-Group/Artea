#!/usr/bin/env bash
# Build the artea_graph, collect the distribution of edge distances
# adopted by the hierarchical router (L0, L1, L2) and the L0-only
# single-layer router, then plot the four-way KDE overlay.
#
# Usage:
#   vldb27-exp/profile_edge_length.sh [<extra flags forwarded to edge_length_profile>]
#
# Expected layout:
#   ./build/vldb27-exp/edge_length_profile  — compiled binary
#   ./vldb27-exp/plot_edge_length.py        — plotter
#   ./vldb27-exp/results/                   — JSON + PDF output

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXP_BIN="${REPO_ROOT}/build/vldb27-exp/edge_length_profile"
PY_SCRIPT="${REPO_ROOT}/vldb27-exp/plot_edge_length.py"
RESULTS_DIR="${REPO_ROOT}/vldb27-exp/results"
JSON_PATH="${RESULTS_DIR}/edge_length_profile_results.json"
# Plotter writes 6 PDFs into this directory:
#   edge_length_distribution_{hier_L0,hier_L1,hier_L2,single_L0,total,legend}.pdf
PLOT_DIR="${RESULTS_DIR}"

if [[ ! -x "${EXP_BIN}" ]]; then
    echo "ERROR: ${EXP_BIN} not found or not executable."
    echo "Build it first (e.g. 'cmake --build build --target edge_length_profile')."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

# Scrub old outputs so a shorter run can't leave stale plots from a
# previous longer run lying around (e.g. hier_L5.pdf after a 3-layer run).
echo "=== Cleaning old outputs ==="
rm -fv "${JSON_PATH}" || true
rm -fv "${PLOT_DIR}"/edge_length_distribution_*.pdf || true
echo

DEFAULT_FLAGS=(
    --dataset sift-1m
    --l0-radius 9285.00
    --beta 2.0
    --ul-select-nbrs-qs 64
    --search-nn-qs 30
    --ul-max-nbr-size 32
    --bl-max-nbr-size 64
    --num-build-loops 5
    --num-triu-iters 12
    --prefill-ratio 0.34
    --scale-coeffs 1.2
    --shifted-coeffs 0.0
    --num-routing-loops 0
    --candidate-queue-size 10
    --samples-per-layer-cap 100000
    --seed 42
    --output "${JSON_PATH}"
)

EXTRA_FLAGS=("$@")

echo "=== Running edge_length_profile ==="
echo "Binary:  ${EXP_BIN}"
echo "Output:  ${JSON_PATH}"
echo

numactl --interleave=all "${EXP_BIN}" "${DEFAULT_FLAGS[@]}" "${EXTRA_FLAGS[@]}"

echo
echo "=== Plotting ==="
python3 "${PY_SCRIPT}" -i "${JSON_PATH}" -o "${PLOT_DIR}"

echo
echo "Done."
echo "  JSON:  ${JSON_PATH}"
echo "  PDFs:  ${PLOT_DIR}/edge_length_distribution_{hier_L0,hier_L1,hier_L2,single_L0,total,legend}.pdf"
