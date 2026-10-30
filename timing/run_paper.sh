#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
"${PYTHON_BIN:-python}" timing/benchmark_all.py --manifest timing/methods.json "$@"
