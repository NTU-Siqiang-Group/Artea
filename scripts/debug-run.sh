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
#
# Run an ASan-built binary repeatedly until it crashes (or the run cap is
# reached). Intended for chasing intermittent memory-safety bugs that only
# reproduce a fraction of the time under normal execution.
#
# Usage:
#   scripts/debug-run.sh <binary> [binary-args...]
#
# Env overrides:
#   MAX_RUNS      Maximum runs before giving up (default: 20).
#   LOG_DIR       Where to drop per-run logs (default: ./debug-logs).
#   ASAN_OPTIONS  Override the default ASan options.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./utils/log.sh
source "${SCRIPT_DIR}/utils/log.sh"

if [[ $# -lt 1 ]]; then
    log_error "Usage: $0 <binary> [binary-args...]"
    exit 1
fi

BINARY="$1"; shift
if [[ ! -x "${BINARY}" ]]; then
    log_error "Binary not executable: ${BINARY}"
    exit 1
fi

MAX_RUNS="${MAX_RUNS:-20}"
LOG_FILE="${LOG_FILE:-./debug-output.txt}"
ASAN_OPTIONS_DEFAULT="abort_on_error=0:halt_on_error=1:detect_stack_use_after_return=1:print_stacktrace=1:strict_string_checks=1:detect_leaks=0"
export ASAN_OPTIONS="${ASAN_OPTIONS:-${ASAN_OPTIONS_DEFAULT}}"

# Enable core dumps for the invoked process so a crash under a non-sanitized
# (Release / RelWithDebInfo) build leaves behind a file we can inspect in gdb.
ulimit -c unlimited || true

# Truncate the aggregate log once at start; each run appends.
: > "${LOG_FILE}"

log_info "Binary:       ${BINARY}"
log_info "Args:         $*"
log_info "MAX_RUNS:     ${MAX_RUNS}"
log_info "LOG_FILE:     ${LOG_FILE}"
log_info "ASAN_OPTIONS: ${ASAN_OPTIONS}"

for i in $(seq 1 "${MAX_RUNS}"); do
    {
        echo ""
        echo "==================== run ${i} / ${MAX_RUNS} ===================="
        echo ""
    } >> "${LOG_FILE}"

    if "${BINARY}" "$@" >> "${LOG_FILE}" 2>&1; then
        log_info "run ${i}: OK"
    else
        rc=$?
        log_error "run ${i}: CRASH (exit=${rc}) -> ${LOG_FILE}"
        echo ""
        echo "---- tail of ${LOG_FILE} ----"
        tail -n 80 "${LOG_FILE}"
        exit "${rc}"
    fi
done

log_success "All ${MAX_RUNS} runs passed without crash."
