#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec ../../../.venv/bin/python -u eval_per_species.py \
  --ckpt "cluster session/hybrid_v4_epoch8.pt" \
  --ckpt "cluster session/hybrid_v4_epoch30.pt"
