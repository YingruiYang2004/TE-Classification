#!/usr/bin/env bash
# Launch a Round-2 smoke from the round2 dir without zsh quoting hazards.
# Usage:  bash _run_smoke.sh T1 [extra args...]
set -euo pipefail
cd "$(dirname "$0")"
TRACK="$1"; shift || true
mkdir -p "results/${TRACK}"
exec ../../../../.venv/bin/python -u run_smoke.py --track "${TRACK}" --ab "$@"
