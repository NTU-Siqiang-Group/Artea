#!/usr/bin/env bash
# Build the artea_graph, profile per-query 1-NN routing (hierarchical
# vs. L0-only) on a random subset of queries, then plot the per-query
# ADR-vs-NDC trajectories.
#
# Usage:
#   vldb27-exp/profile_adr_vs_ndc.sh [<extra flags forwarded to adr_vs_ndc_profile>]
#
# Expected layout:
#   ./build/vldb27-exp/adr_vs_ndc_profile  — compiled binary
#   ./vldb27-exp/plot_adr_vs_ndc.py        — plotter
#   ./vldb27-exp/results/                  — JSON + PNG output

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXP_BIN="${REPO_ROOT}/build/vldb27-exp/adr_vs_ndc_profile"
PY_SCRIPT="${REPO_ROOT}/vldb27-exp/plot_adr_vs_ndc.py"
RESULTS_DIR="${REPO_ROOT}/vldb27-exp/results"
JSON_PATH="${RESULTS_DIR}/adr_vs_ndc_profile_results.json"
PDF_PATH="${RESULTS_DIR}/adr_vs_ndc_per_query.pdf"

if [[ ! -x "${EXP_BIN}" ]]; then
    echo "ERROR: ${EXP_BIN} not found or not executable."
    echo "Build it first (e.g. 'cmake --build build --target adr_vs_ndc_profile')."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

# Default adr_vs_ndc_profile flags — override by passing extras on the
# command line.
DEFAULT_FLAGS=(
    --dataset sift-1m
    --l0-radius 45000.00
    --beta 2.0
    --ul-select-nbrs-qs 64
    --search-nn-qs 30
    --ul-max-nbr-size 32
    --bl-max-nbr-size 64
    --num-build-loops 5
    --num-triu-iters 12
    --prefill-ratio 0.34
    --scale-coeffs 1.1
    --shifted-coeffs 0.0
    --num-routing-loops 0
    --routing-topk 96
    --routing-queue-size 128
    --num-queries 20
    --seed 42
    --candidate-queue-size 10
    --output "${JSON_PATH}"
)

EXTRA_FLAGS=("$@")

echo "=== Running adr_vs_ndc_profile ==="
echo "Binary:  ${EXP_BIN}"
echo "Output:  ${JSON_PATH}"
echo

numactl --interleave=all "${EXP_BIN}" "${DEFAULT_FLAGS[@]}" "${EXTRA_FLAGS[@]}"

echo
echo "=== Plotting ==="
python3 "${PY_SCRIPT}" -i "${JSON_PATH}" -o "${PDF_PATH}"

echo
echo "Done."
echo "  JSON:  ${JSON_PATH}"
echo "  Plot:  ${PDF_PATH}"
