#!/usr/bin/env bash
# Round-2.5 sweep:
#   1. 6-epoch nostack baselines for T1, T2, T3 (does longer training rescue
#      SF F1 / per-clade hAT recall?)
#   2. T3 stack=ON with weakened invariance:
#        weak1: lambda=0.05, eta=0.005 (very gentle)
#        weak2: lambda=0.15, eta=0.02  (moderate)
#      both at 6 epochs, paired against T3_nostack_6ep for delta.
#
# All runs use 5000-seq subset, MPS, batch 16. Total ~5h wall clock.
set -uo pipefail
cd "$(dirname "$0")"
PY=../../../../.venv/bin/python
mkdir -p results/T1 results/T2 results/T3

run() {
    local tag="$1"; shift
    echo "=== $tag ==="
    "$PY" -u run_smoke.py "$@" 2>&1 | tee -a "results/round25_${tag}.log"
    echo "  done -> results/round25_${tag}.log"
}

# ---- 1. 6-epoch nostack baselines ----
run T1_nostack_6ep --track T1 --no-stack --epochs 6 --tag _6ep
run T2_nostack_6ep --track T2 --no-stack --epochs 6 --tag _6ep
run T3_nostack_6ep --track T3 --no-stack --epochs 6 --tag _6ep

# ---- 2. T3 weakened-stack arms (only stack arm; pair against T3 nostack 6ep) ----
run T3_weak1_6ep --track T3 --epochs 6 --dann-lambda 0.05 --dann-warmup 2 \
    --gdro-eta 0.005 --tag _weak1_6ep
run T3_weak2_6ep --track T3 --epochs 6 --dann-lambda 0.15 --dann-warmup 2 \
    --gdro-eta 0.02 --tag _weak2_6ep

echo "ALL ROUND-2.5 ARMS DONE"
