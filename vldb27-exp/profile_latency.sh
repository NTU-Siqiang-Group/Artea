#!/usr/bin/env bash
# Build the artea_graph, measure per-query 1-NN latency (hierarchical
# router vs. L0-only router) over the full query set, then plot pXX
# comparisons.
#
# Usage:
#   vldb27-exp/profile_latency.sh [<extra flags forwarded to latency_profile>]
#
# Expected layout:
#   ./build/vldb27-exp/latency_profile  — compiled binary
#   ./vldb27-exp/plot_latency.py        — plotter
#   ./vldb27-exp/results/               — JSON + PDF output

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXP_BIN="${REPO_ROOT}/build/vldb27-exp/latency_profile"
PY_SCRIPT="${REPO_ROOT}/vldb27-exp/plot_latency.py"
RESULTS_DIR="${REPO_ROOT}/vldb27-exp/results"
JSON_PATH="${RESULTS_DIR}/latency_profile_results.json"
# Plotter auto-names the PDF as "<dataset>_1nn_0.9recall_latency.pdf"
# inside ${RESULTS_DIR}; the dataset is read from the JSON payload.

if [[ ! -x "${EXP_BIN}" ]]; then
    echo "ERROR: ${EXP_BIN} not found or not executable."
    echo "Build it first (e.g. 'cmake --build build --target latency_profile')."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

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
    --scale-coeffs 1.1
    --shifted-coeffs 0.0
    --num-routing-loops 0
    --routing-topk 96
    --routing-queue-size 128
    --candidate-queue-size 20
    --output "${JSON_PATH}"
)

EXTRA_FLAGS=("$@")

echo "=== Running latency_profile ==="
echo "Binary:  ${EXP_BIN}"
echo "Output:  ${JSON_PATH}"
echo

numactl --interleave=all "${EXP_BIN}" "${DEFAULT_FLAGS[@]}" "${EXTRA_FLAGS[@]}"

echo
echo "=== Plotting ==="
python3 "${PY_SCRIPT}" -i "${JSON_PATH}" -o "${RESULTS_DIR}"

echo
echo "Done."
echo "  JSON:  ${JSON_PATH}"
echo "  Plot:  ${RESULTS_DIR}/<dataset>_1nn_0.9recall_latency.pdf"
