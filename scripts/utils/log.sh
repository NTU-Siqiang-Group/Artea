#!/usr/bin/env bash

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  LOG_COLOR_INFO='\033[1;34m'
  LOG_COLOR_SUCCESS='\033[1;32m'
  LOG_COLOR_WARN='\033[1;33m'
  LOG_COLOR_ERROR='\033[1;31m'
  LOG_COLOR_RESET='\033[0m'
else
  LOG_COLOR_INFO=''
  LOG_COLOR_SUCCESS=''
  LOG_COLOR_WARN=''
  LOG_COLOR_ERROR=''
  LOG_COLOR_RESET=''
fi

_log_print() {
  local level="$1"
  local color="$2"
  local message="$3"
  printf "%b[%s]%b %s\n" "${color}" "${level}" "${LOG_COLOR_RESET}" "${message}"
}

log_info() {
  _log_print "INFO" "${LOG_COLOR_INFO}" "$*"
}

log_success() {
  _log_print " OK " "${LOG_COLOR_SUCCESS}" "$*"
}

log_warn() {
  _log_print "WARN" "${LOG_COLOR_WARN}" "$*"
}

log_error() {
  _log_print "ERR " "${LOG_COLOR_ERROR}" "$*"
}
