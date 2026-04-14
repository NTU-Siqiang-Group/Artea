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

# ThreadSanitizer build. Separate BUILD_DIR from the ASan build — the two
# runtimes are mutually exclusive. TSan needs -O1 (or higher) + frame
# pointers to produce reliable stacks.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

exec env \
    BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build-tsan}" \
    BUILD_TYPE="${BUILD_TYPE:-Debug}" \
    CMAKE_CXX_FLAGS_EXTRA="-fsanitize=thread -fno-omit-frame-pointer -O1 -Wno-maybe-uninitialized" \
    CMAKE_EXE_LINKER_FLAGS_EXTRA="-fsanitize=thread" \
    "${SCRIPT_DIR}/install.sh" "$@"
