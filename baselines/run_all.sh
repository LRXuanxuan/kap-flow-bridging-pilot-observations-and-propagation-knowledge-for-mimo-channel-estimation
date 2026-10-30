#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for method in pnpflow rcflow pnp-cm ls lmmse; do
  python baselines/run.py evaluate --method "$method" --resume "$@"
done
python baselines/run.py evaluate --method srf --conditions ckm --resume "$@"
