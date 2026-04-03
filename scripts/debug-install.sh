#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec env BUILD_TYPE="${BUILD_TYPE:-Debug}" \
     CMAKE_CXX_FLAGS_EXTRA="-fsanitize=address -fno-omit-frame-pointer -Wno-maybe-uninitialized" \
     CMAKE_EXE_LINKER_FLAGS_EXTRA="-fsanitize=address" \
     "${SCRIPT_DIR}/install.sh" "$@"
