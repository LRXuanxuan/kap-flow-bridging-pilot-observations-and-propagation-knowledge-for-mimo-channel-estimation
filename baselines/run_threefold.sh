#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for method in pnpflow lmmse; do
  python baselines/run.py evaluate --method "$method" --track threefold --resume "$@"
done
