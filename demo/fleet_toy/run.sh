#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export MIRAGE_HOME="${ROOT}"
export ROCM_PATH="${ROCM_PATH:-/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel}"

if [[ -z "${AMDGPU_TARGETS:-}" ]]; then
  AMDGPU_ARCH_TOOL="$(command -v amdgpu-arch)"
  AMDGPU_TARGETS="$("${AMDGPU_ARCH_TOOL}" | awk 'NR == 1 { print; exit }')"
  export AMDGPU_TARGETS
fi

cd "${ROOT}"
python3 demo/fleet_toy/run.py
