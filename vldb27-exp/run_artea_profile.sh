#!/usr/bin/env bash
# Build artea_graph profiles (hierarchical vs. L0-only) and plot the
# ADR-vs-NDC curves.
#
# Usage:
#   vldb27-exp/run_artea_profile.sh [-- <extra flags forwarded to artea_profile>]
#
# Expected layout:
#   ./build/vldb27-exp/artea_profile       — compiled binary
#   ./vldb27-exp/plot_router_profile.py    — plotter
#   ./vldb27-exp/results/                  — JSON + PNG output

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXP_BIN="${REPO_ROOT}/build/vldb27-exp/artea_profile"
PY_SCRIPT="${REPO_ROOT}/vldb27-exp/plot_router_profile.py"
RESULTS_DIR="${REPO_ROOT}/vldb27-exp/results"
JSON_PATH="${RESULTS_DIR}/router_profiler_results.json"
PNG_PATH="${RESULTS_DIR}/router_profile_adr_vs_ndc.png"

if [[ ! -x "${EXP_BIN}" ]]; then
    echo "ERROR: ${EXP_BIN} not found or not executable."
    echo "Build it first (e.g. 'cmake --build build --target artea_profile')."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

# Default artea_profile flags — override by passing extras after "--".
DEFAULT_FLAGS=(
    --dataset sift-1m
    --l0-radius 9285.00
    --beta 2.0
    --ul-select-nbrs-qs 64
    --bl-select-nbrs-qs 64
    --search-nn-qs 20
    --max-nbr-size 48
    --refining-max-nbr-size 96
    --num-build-loops 5
    --num-triu-iters 12
    --prefill-ratio 0.34
    --scale-coeffs 1.0
    --shifted-coeffs 0.0
    --num-routing-loops 0
    --routing-topk 96
    --routing-queue-size 128
    --output "${JSON_PATH}"
)

EXTRA_FLAGS=("$@")

echo "=== Running artea_profile ==="
echo "Binary:  ${EXP_BIN}"
echo "Output:  ${JSON_PATH}"
echo

numactl --interleave=all "${EXP_BIN}" "${DEFAULT_FLAGS[@]}" "${EXTRA_FLAGS[@]}"

echo
echo "=== Plotting ==="
python3 "${PY_SCRIPT}" -i "${JSON_PATH}" -o "${PNG_PATH}"

echo
echo "Done."
echo "  JSON:  ${JSON_PATH}"
echo "  Plot:  ${PNG_PATH}"
