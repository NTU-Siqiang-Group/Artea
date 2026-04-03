#!/usr/bin/env bash
# Copyright 2026 Weitang Ye
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

# shellcheck source=./utils/log.sh
source "${SCRIPT_DIR}/utils/log.sh"
# shellcheck source=./utils/oneapi.sh
source "${SCRIPT_DIR}/utils/oneapi.sh"
# shellcheck source=./utils/version.sh
source "${SCRIPT_DIR}/utils/version.sh"

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-64}"
MIN_CMAKE_VERSION="${MIN_CMAKE_VERSION:-3.24.0}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
CXX_COMPILER="${CXX_COMPILER:-$HOME/.local/bin/g++}"
C_COMPILER="${C_COMPILER:-$HOME/.local/bin/gcc}"
PROFILING_DEFS=""
EXTRA_CXX_FLAGS="${CMAKE_CXX_FLAGS_EXTRA:-}"
EXTRA_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS_EXTRA:-}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            BUILD_TYPE="Release"
            PROFILING_DEFS=""
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            PROFILING_DEFS="-DARTEA_PROFILING"
            shift
            ;;
        --profile)
            BUILD_TYPE="Release"
            PROFILING_DEFS="-DARTEA_PROFILING"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Usage: $0 [--debug|--release|--profile]"
            exit 1
            ;;
    esac
done

log_info "Project root: ${PROJECT_ROOT}"
log_info "Build directory: ${BUILD_DIR}"
log_info "Build type: ${BUILD_TYPE}"
log_info "CMake binary: ${CMAKE_BIN}"
log_info "Using C++ compiler: ${CXX_COMPILER}"
log_info "Using C compiler: ${C_COMPILER}"
if [[ -n "${PROFILING_DEFS}" ]]; then
  log_info "Profiling definitions: ${PROFILING_DEFS}"
fi
if [[ -n "${EXTRA_CXX_FLAGS}" ]]; then
  log_info "Extra CXX flags: ${EXTRA_CXX_FLAGS}"
fi
check_cmake_version "${CMAKE_BIN}" "${MIN_CMAKE_VERSION}"

if ! oneapi_is_active; then
  log_warn "oneAPI environment not detected (needed for TBB). Activating..."
  oneapi_activate
else
  log_success "oneAPI environment already active."
fi

log_info "Running CMake configure..."
"${CMAKE_BIN}" -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_C_COMPILER="${C_COMPILER}" \
      -DCMAKE_CXX_FLAGS="${PROFILING_DEFS} ${EXTRA_CXX_FLAGS}" \
      -DCMAKE_EXE_LINKER_FLAGS="${EXTRA_LINKER_FLAGS}" \
      "${PROJECT_ROOT}"
log_success "CMake configure completed."

log_info "Building project (jobs=${JOBS})..."
"${CMAKE_BIN}" --build "${BUILD_DIR}" -j"${JOBS}"
log_success "Build finished successfully."
