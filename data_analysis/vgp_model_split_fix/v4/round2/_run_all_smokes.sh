#!/usr/bin/env bash
# Run all three Round-2 smokes back-to-back on the local accelerator.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/T1 results/T2 results/T3
PY=../../../../.venv/bin/python
for T in T1 T2 T3; do
  echo "=== ${T} ==="
  "${PY}" -u run_smoke.py --track "${T}" --ab \
    > "results/${T}/run_all.log" 2>&1
  echo "  done  -> results/${T}/run_all.log"
done
