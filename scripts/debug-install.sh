#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec env BUILD_TYPE=Debug "${SCRIPT_DIR}/install.sh" "$@"
