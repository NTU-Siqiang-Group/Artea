#!/usr/bin/env bash
# Test script for oneapi_activate function

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Source the required utilities
source "${SCRIPT_DIR}/utils/log.sh"
source "${SCRIPT_DIR}/utils/oneapi.sh"

echo "=========================================="
echo "Testing oneapi_activate function"
echo "=========================================="
echo

# Test 1: Check if oneapi_is_active works
echo "Test 1: Checking if oneAPI is currently active..."
if oneapi_is_active; then
    log_success "oneAPI is already active (icpx and icx found in PATH)"
else
    log_info "oneAPI is not currently active"
fi
echo

# Test 2: Try to find setvars.sh
echo "Test 2: Attempting to find setvars.sh..."
if setvars_path="$(oneapi_find_setvars)"; then
    log_success "Found setvars.sh at: ${setvars_path}"
else
    log_warn "Could not find setvars.sh"
    log_info "Checked locations:"
    log_info "  - \$ONEAPI_SETVARS: ${ONEAPI_SETVARS:-<not set>}"
    log_info "  - \$ONEAPI_ROOT/setvars.sh: ${ONEAPI_ROOT:-<not set>}"
    log_info "  - ${HOME}/intel/oneapi/setvars.sh"
    log_info "  - /opt/intel/oneapi/setvars.sh"
fi
echo

# Test 3: Test oneapi_activate
echo "Test 3: Testing oneapi_activate function..."
if oneapi_activate; then
    log_success "oneapi_activate completed successfully"
else
    log_error "oneapi_activate failed with exit code $?"
fi
echo

# Test 4: Verify activation by checking compilers
echo "Test 4: Verifying compiler availability after activation..."
if command -v icpx >/dev/null 2>&1; then
    icpx_path="$(command -v icpx)"
    log_success "icpx found at: ${icpx_path}"
    log_info "icpx version: $(icpx --version 2>&1 | head -n1)"
else
    log_error "icpx not found in PATH"
fi

if command -v icx >/dev/null 2>&1; then
    icx_path="$(command -v icx)"
    log_success "icx found at: ${icx_path}"
    log_info "icx version: $(icx --version 2>&1 | head -n1)"
else
    log_error "icx not found in PATH"
fi
echo

# Test 5: Test oneapi_compiler_path function
echo "Test 5: Testing oneapi_compiler_path function..."
if icpx_full_path="$(oneapi_compiler_path icpx)"; then
    log_success "oneapi_compiler_path icpx: ${icpx_full_path}"
else
    log_error "oneapi_compiler_path icpx failed"
fi

if icx_full_path="$(oneapi_compiler_path icx)"; then
    log_success "oneapi_compiler_path icx: ${icx_full_path}"
else
    log_error "oneapi_compiler_path icx failed"
fi
echo

# Test 6: Test idempotency (calling activate again should be safe)
echo "Test 6: Testing idempotency (calling oneapi_activate again)..."
if oneapi_activate; then
    log_success "Second call to oneapi_activate succeeded (should report already active)"
else
    log_error "Second call to oneapi_activate failed"
fi
echo

echo "=========================================="
echo "Test suite completed"
echo "=========================================="
