#!/bin/bash

# Artea Micro Benchmarks Runner Script
# Usage: ./run_bench.sh <benchmark_name> [additional_args...]

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default build directory
BUILD_DIR="${BUILD_DIR:-build}"
BENCH_DIR="${BUILD_DIR}/micro_benchmarks"

# Check if benchmark name is provided
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: No benchmark name provided${NC}"
    echo "Usage: $0 <benchmark_name> [additional_args...]"
    echo ""
    echo "Available benchmarks:"
    echo "  - bench_random_eg"
    echo "  - bench_propagate_engine"
    echo "  - bench_lsh_table"
    echo "  - bench_radius_prober"
    echo "  - bench_simd_distance"
    echo "  - bench_simd_fma"
    echo "  - bench_simd_linear"
    echo "  - bench_vertex_generator"
    echo ""
    echo "Example: $0 bench_random_eg --init-nbrs 64 -i 20"
    exit 1
fi

BENCH_NAME="$1"
shift  # Remove first argument, rest are passed to benchmark

# Add prefix if not already present
if [[ ! "$BENCH_NAME" =~ ^bench_ ]]; then
    BENCH_NAME="bench_${BENCH_NAME}"
fi

BENCH_PATH="${BENCH_DIR}/${BENCH_NAME}"

# Check if benchmark exists
if [ ! -f "$BENCH_PATH" ]; then
    echo -e "${RED}Error: Benchmark '${BENCH_NAME}' not found at ${BENCH_PATH}${NC}"
    echo ""
    echo "Available benchmarks in ${BENCH_DIR}:"
    ls -1 "${BENCH_DIR}"/bench_* 2>/dev/null | xargs -n1 basename || echo "  (none found)"
    echo ""
    echo "Tip: Run 'cmake --build ${BUILD_DIR}' to build benchmarks"
    exit 1
fi

# Check if benchmark is executable
if [ ! -x "$BENCH_PATH" ]; then
    echo -e "${RED}Error: Benchmark '${BENCH_NAME}' is not executable${NC}"
    exit 1
fi

# Default parameters for each benchmark
declare -A DEFAULT_PARAMS
DEFAULT_PARAMS["bench_random_eg"]="--config ./datasets.json --dataset sift-1m --init-nbrs 32 --reserved-nbrs 32 -i 10"
DEFAULT_PARAMS["bench_propagate_engine"]="--config ./datasets.json --dataset sift-1m --init-nbrs 32 --reserved-nbrs 32 --max-nbrs 16 --num-iters 5 --scale-coeffs 1.0 --shifted-coeffs 0.0 -i 10"

# Get default parameters for this benchmark
DEFAULT_ARGS="${DEFAULT_PARAMS[$BENCH_NAME]:-}"

# Print banner
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Running: ${BENCH_NAME}${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# If no additional arguments provided, use defaults
if [ $# -eq 0 ] && [ -n "$DEFAULT_ARGS" ]; then
    echo -e "${YELLOW}Using default parameters:${NC}"
    echo "  $DEFAULT_ARGS"
    echo ""
    # Parse default args into array
    eval "set -- $DEFAULT_ARGS"
fi

# Display command being run
echo -e "${YELLOW}Command:${NC}"
echo "  ${BENCH_PATH} $@"
echo ""
echo -e "${GREEN}========================================${NC}"
echo ""

# Run the benchmark
"$BENCH_PATH" "$@"

# Print completion message
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Benchmark completed successfully${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Benchmark failed with exit code: $EXIT_CODE${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE