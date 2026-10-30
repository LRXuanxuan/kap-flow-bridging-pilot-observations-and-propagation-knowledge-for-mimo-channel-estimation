#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reproduce the four historically completed cells. Both M and other CLI settings remain configurable.
python main/run.py evaluate --track robustness --conditions ckm --priors standard robust --resume "$@"
