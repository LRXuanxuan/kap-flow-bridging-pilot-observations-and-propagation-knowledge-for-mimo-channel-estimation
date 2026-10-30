#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python main/run.py evaluate --track standard --m 1 2 4 8 16 --resume "$@"
