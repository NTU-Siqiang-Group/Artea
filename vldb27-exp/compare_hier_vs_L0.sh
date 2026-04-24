#!/usr/bin/env bash
# Hier-vs-L0 comparison. Runs the binary twice with independent
# queue-size sweeps tuned for each plot:
#   - recall sweep → compare_hier_vs_L0_recall_results.json
#   - ADR    sweep → compare_hier_vs_L0_adr_results.json
# Then the Python plotter reads both JSONs and emits two PDFs:
#   - compare_hier_vs_L0_recall.pdf   (x = Recall, y = QPS, both linear)
#   - compare_hier_vs_L0_adr.pdf      (x = QPS,    y = ADR - 1 log scale)
#
# Usage:
#   vldb27-exp/compare_hier_vs_L0.sh [<extra flags forwarded to both runs>]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXP_BIN="${REPO_ROOT}/build/vldb27-exp/compare_hier_vs_L0"
PY_SCRIPT="${REPO_ROOT}/vldb27-exp/plot_compare_hier_vs_L0.py"
RESULTS_DIR="${REPO_ROOT}/vldb27-exp/results"

JSON_RECALL_PATH="${RESULTS_DIR}/compare_hier_vs_L0_recall_results.json"
JSON_ADR_PATH="${RESULTS_DIR}/compare_hier_vs_L0_adr_results.json"
PDF_RECALL_PATH="${RESULTS_DIR}/compare_hier_vs_L0_recall.pdf"
PDF_ADR_PATH="${RESULTS_DIR}/compare_hier_vs_L0_adr.pdf"

if [[ ! -x "${EXP_BIN}" ]]; then
    echo "ERROR: ${EXP_BIN} not found or not executable."
    echo "Build it first (e.g. 'cmake --build build --target compare_hier_vs_L0')."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "=== Cleaning old outputs ==="
rm -fv \
    "${JSON_RECALL_PATH}" "${JSON_ADR_PATH}" \
    "${PDF_RECALL_PATH}"  "${PDF_ADR_PATH}" || true
echo

# Shared build-side + search-side flags. The two plots only differ on
# --candidate-queue-config and --output; everything else is identical.
SHARED_FLAGS=(
    --dataset sift-1m
    --l0-radius 9285.00
    --beta 2.0
    --ul-select-nbrs-qs 64
    --search-nn-qs 20
    --ul-max-nbr-size 32
    --bl-max-nbr-size 64
    --num-build-loops 5
    --num-triu-iters 12
    --query-topk 1
    --test-runs 20
    --warmup-runs 10
    --prefill-ratio 0.34
    --scale-coeffs 1.10
    --shifted-coeffs 0.00
    --num-routing-loops 0
    # --perform-arc
    # --aspect-ratio-constraint 8.00
    # --insert-on-l0
    # --bl-select-nbrs-qs 32
)

# Queue-size sweep dedicated to the Recall–QPS plot. Fine-grained around
# the recall-saturated regime so the frontier reads cleanly.
RECALL_FLAGS=(
    --candidate-queue-config 1,10,1
    --output "${JSON_RECALL_PATH}"
)

# Queue-size sweep dedicated to the ADR–QPS plot. ADR has a much wider
# dynamic range (log-scale y-axis), so we sample more broadly and drive
# the tail down with a larger top-end queue.
ADR_FLAGS=(
    --candidate-queue-config 1,10,1
    --output "${JSON_ADR_PATH}"
)

EXTRA_FLAGS=("$@")

echo "=== Run 1/2: Recall sweep ==="
echo "Binary:  ${EXP_BIN}"
echo "Output:  ${JSON_RECALL_PATH}"
echo
numactl --interleave=all "${EXP_BIN}" \
    "${SHARED_FLAGS[@]}" "${RECALL_FLAGS[@]}" "${EXTRA_FLAGS[@]}"

echo
echo "=== Run 2/2: ADR sweep ==="
echo "Binary:  ${EXP_BIN}"
echo "Output:  ${JSON_ADR_PATH}"
echo
numactl --interleave=all "${EXP_BIN}" \
    "${SHARED_FLAGS[@]}" "${ADR_FLAGS[@]}" "${EXTRA_FLAGS[@]}"

echo
echo "=== Plotting ==="
python3 "${PY_SCRIPT}" \
    --input-recall "${JSON_RECALL_PATH}" \
    --input-adr    "${JSON_ADR_PATH}" \
    --output-dir   "${RESULTS_DIR}"

echo
echo "Done."
echo "  Recall JSON: ${JSON_RECALL_PATH}"
echo "  ADR    JSON: ${JSON_ADR_PATH}"
echo "  Recall PDF:  ${PDF_RECALL_PATH}"
echo "  ADR    PDF:  ${PDF_ADR_PATH}"

# ----------------------------------------------------
# --dataset sift-1m --l0-radius 4574.00 --beta 3.0
# --dataset gist-1m --l0-radius 1.13 --beta 2.0
