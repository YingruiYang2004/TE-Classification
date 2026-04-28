#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec ../../../../.venv/bin/python -u run_smoke_aug.py --both
