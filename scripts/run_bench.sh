#!/usr/bin/env bash

# Artea Micro Benchmarks Runner Script
# Usage: ./scripts/run_bench.sh <benchmark_name> [additional_args...]

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default build directory
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
BENCH_DIR="${BUILD_DIR}/micro_benchmarks"

print_available_benchmarks() {
    echo "Available benchmarks:"
    if [ -d "${BENCH_DIR}" ] && ls -1 "${BENCH_DIR}"/bench_* >/dev/null 2>&1; then
        ls -1 "${BENCH_DIR}"/bench_* | xargs -n1 basename
    else
        echo "  (none found)"
    fi
    echo "  - all"
}

run_single_benchmark() {
    local bench_name="$1"
    shift

    # Add prefix if not already present
    if [[ ! "$bench_name" =~ ^bench_ ]]; then
        bench_name="bench_${bench_name}"
    fi

    local bench_path="${BENCH_DIR}/${bench_name}"

    # Check if benchmark exists
    if [ ! -f "$bench_path" ]; then
        echo -e "${RED}Error: Benchmark '${bench_name}' not found at ${bench_path}${NC}"
        echo ""
        echo "Available benchmarks in ${BENCH_DIR}:"
        ls -1 "${BENCH_DIR}"/bench_* 2>/dev/null | xargs -n1 basename || echo "  (none found)"
        echo ""
        echo "Tip: Run 'cmake --build ${BUILD_DIR}' to build benchmarks"
        return 1
    fi

    # Check if benchmark is executable
    if [ ! -x "$bench_path" ]; then
        echo -e "${RED}Error: Benchmark '${bench_name}' is not executable${NC}"
        return 1
    fi

    # Get default parameters for this benchmark
    local default_args="${DEFAULT_PARAMS[$bench_name]:-}"

    # Print banner
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Running: ${bench_name}${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    local -a run_args=("$@")
    if [ ${#run_args[@]} -eq 0 ] && [ -n "$default_args" ]; then
        echo -e "${YELLOW}Using default parameters:${NC}"
        echo "  $default_args"
        echo ""
        # Parse default args into array
        eval "set -- $default_args"
        run_args=("$@")
    fi

    # Display command being run
    echo -e "${YELLOW}Command:${NC}"
    echo "  ${bench_path} ${run_args[*]}"
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo ""

    "${bench_path}" "${run_args[@]}"
    local exit_code=$?
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  Benchmark completed successfully${NC}"
        echo -e "${GREEN}========================================${NC}"
    else
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}  Benchmark failed with exit code: $exit_code${NC}"
        echo -e "${RED}========================================${NC}"
    fi
    echo ""

    return $exit_code
}

# Check if benchmark name is provided
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: No benchmark name provided${NC}"
    echo "Usage: $0 <benchmark_name> [additional_args...]"
    echo ""
    print_available_benchmarks
    echo ""
    echo "Example: $0 bench_random_eg --init-nbrs 64 -i 20"
    echo "Example: $0 all"
    exit 1
fi

BENCH_NAME="$1"
shift  # Remove first argument, rest are passed to benchmark

# Default parameters for each benchmark
declare -A DEFAULT_PARAMS
DEFAULT_PARAMS["bench_random_eg"]="--config ${PROJECT_ROOT}/configs/datasets.json --dataset sift-1m --init-nbrs 32 -i 10"
DEFAULT_PARAMS["bench_propagate_engine"]="--config ${PROJECT_ROOT}/configs/datasets.json --dataset sift-1m --init-nbrs 32 --max-nbrs 16 --num-iters 5 --scale-coeffs 1.0 --shifted-coeffs 0.0 -i 10"

if [ "$BENCH_NAME" = "all" ]; then
    if [ ! -d "${BENCH_DIR}" ] || ! ls -1 "${BENCH_DIR}"/bench_* >/dev/null 2>&1; then
        echo -e "${RED}Error: No benchmarks found in ${BENCH_DIR}${NC}"
        echo "Tip: Run 'cmake --build ${BUILD_DIR}' to build benchmarks"
        exit 1
    fi

    overall_exit_code=0
    for bench_path in "${BENCH_DIR}"/bench_*; do
        bench_file="$(basename "$bench_path")"
        if [ ! -x "$bench_path" ]; then
            echo -e "${YELLOW}Skipping non-executable benchmark: ${bench_file}${NC}"
            overall_exit_code=1
            continue
        fi
        if ! run_single_benchmark "$bench_file" "$@"; then
            overall_exit_code=1
        fi
    done
    exit $overall_exit_code
fi

run_single_benchmark "$BENCH_NAME" "$@"
exit $?
