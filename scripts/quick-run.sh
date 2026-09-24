#!/usr/bin/env sh
# Run from the ARTEA repository root. Set num_skip_levels, rnet_beta, and
# l0_min_distance in the workload's indexes-config.artea entry.
set -eu
export OMP_NUM_THREADS="$(nproc)"
numactl --interleave=all ./build/unit_tests/test_artea_graph \
    --workload "${1:-workloads/sift1m-bench.jsonc}"
