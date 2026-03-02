#!/usr/bin/env bash

_version_log() {
  local level_fn="$1"
  shift
  if command -v "${level_fn}" >/dev/null 2>&1; then
    "${level_fn}" "$*"
  else
    printf "%s\n" "$*"
  fi
}

version_ge() {
  # Returns 0 when $1 >= $2
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]
}

check_binary_min_version() {
  # stdout: detected version on success or when version is too low
  # return codes:
  #   0: binary exists and version >= min version
  #   1: binary exists but version < min version
  #   2: binary not found
  #   3: failed to parse version
  local bin="$1"
  local min_version="$2"
  local found_version=""

  if ! command -v "${bin}" >/dev/null 2>&1; then
    return 2
  fi

  found_version="$("${bin}" --version | awk 'NR==1{print $3}')"
  if [[ -z "${found_version}" ]]; then
    return 3
  fi

  printf '%s\n' "${found_version}"

  if version_ge "${found_version}" "${min_version}"; then
    return 0
  fi

  return 1
}

check_cmake_version() {
  local cmake_bin="$1"
  local min_cmake_version="$2"
  local cmake_check_rc=0
  local cmake_version=""

  cmake_version="$(check_binary_min_version "${cmake_bin}" "${min_cmake_version}")" || cmake_check_rc=$?

  if (( cmake_check_rc == 2 )); then
    _version_log log_error "CMake binary '${cmake_bin}' is not available."
    return 1
  fi

  if (( cmake_check_rc == 3 )); then
    _version_log log_error "Failed to detect cmake version."
    return 1
  fi

  if (( cmake_check_rc == 1 )); then
    _version_log log_error "cmake ${min_cmake_version}+ is required, but found ${cmake_version}."
    _version_log log_error "Please upgrade cmake and re-run scripts/install.sh."
    return 1
  fi

  _version_log log_success "cmake version check passed (${cmake_version})."
}
