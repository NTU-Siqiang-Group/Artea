#!/usr/bin/env bash

_oneapi_log() {
  local level_fn="$1"
  shift
  if command -v "${level_fn}" >/dev/null 2>&1; then
    "${level_fn}" "$*"
  else
    printf "%s\n" "$*"
  fi
}

oneapi_is_active() {
  command -v icpx >/dev/null 2>&1 && command -v icx >/dev/null 2>&1
}

oneapi_find_setvars() {
  local candidate

  if [[ -n "${ONEAPI_SETVARS:-}" && -f "${ONEAPI_SETVARS}" ]]; then
    printf "%s\n" "${ONEAPI_SETVARS}"
    return 0
  fi

  if [[ -n "${ONEAPI_ROOT:-}" && -f "${ONEAPI_ROOT%/}/setvars.sh" ]]; then
    printf "%s\n" "${ONEAPI_ROOT%/}/setvars.sh"
    return 0
  fi

  for candidate in \
    "${HOME}/intel/oneapi/setvars.sh" \
    "/opt/intel/oneapi/setvars.sh"
  do
    if [[ -f "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done

  return 1
}

oneapi_activate() {
  local setvars
  local had_errexit=0
  local had_nounset=0
  local had_pipefail=0
  local source_rc=0

  if oneapi_is_active; then
    _oneapi_log log_info "oneAPI environment is already active."
    return 0
  fi

  if ! setvars="$(oneapi_find_setvars)"; then
    _oneapi_log log_error "Cannot find oneAPI setvars.sh. Set ONEAPI_SETVARS or install oneAPI first."
    return 1
  fi

  _oneapi_log log_info "Activating oneAPI environment: ${setvars}"
  [[ $- == *e* ]] && had_errexit=1
  [[ $- == *u* ]] && had_nounset=1
  if set -o | grep -q '^pipefail[[:space:]]*on'; then
    had_pipefail=1
  fi

  # setvars.sh is not fully compatible with nounset/errexit in all versions.
  set +e
  set +u
  set +o pipefail

  # shellcheck source=/dev/null
  source "${setvars}" >/dev/null 2>&1
  source_rc=$?

  (( had_errexit )) && set -e || true
  (( had_nounset )) && set -u || true
  (( had_pipefail )) && set -o pipefail || set +o pipefail

  if (( source_rc != 0 )); then
    _oneapi_log log_error "Failed to source oneAPI setvars.sh: ${setvars}"
    return 1
  fi

  if oneapi_is_active; then
    _oneapi_log log_success "oneAPI environment activated."
    return 0
  fi

  _oneapi_log log_error "oneAPI activation ran, but icx/icpx are still unavailable in PATH."
  return 1
}

oneapi_compiler_path() {
  local compiler="$1"
  local path_from_env=""
  local fallback_path="${HOME}/intel/oneapi/compiler/latest/bin/${compiler}"

  path_from_env="$(command -v "${compiler}" 2>/dev/null || true)"
  if [[ -n "${path_from_env}" ]]; then
    printf "%s\n" "${path_from_env}"
    return 0
  fi

  if [[ -x "${fallback_path}" ]]; then
    printf "%s\n" "${fallback_path}"
    return 0
  fi

  return 1
}
