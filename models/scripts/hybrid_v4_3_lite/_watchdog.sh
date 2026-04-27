#!/usr/bin/env bash
# Monitors overnight run; falls back to "fix B" (eval-mode patch + disable PE/aug) on NaN.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
LOG="$HERE/_watchdog.log"
echo "[$(date)] watchdog start; current PID(s): $(pgrep -f run_overnight.py | tr '\n' ' ')" >> "$LOG"

apply_fix_B() {
  echo "[$(date)] applying fix B" >> "$LOG"
  ../../../.venv/bin/python <<'PY' >> "$LOG" 2>&1
import json, pathlib, re
nb = pathlib.Path('02_train.ipynb')
data = json.loads(nb.read_text())
patched_eval = patched_aug = patched_pe = 0
for c in data['cells']:
    if c['cell_type'] != 'code':
        continue
    src = c['source']
    new = []
    for line in src:
        # 1) evaluate(): change model.eval() -> model.train() (BN live stats)
        if 'def evaluate' in ''.join(src):
            if line.strip() == 'model.eval()':
                line = line.replace('model.eval()', 'model.train()  # FIX_B: BN live stats avoid NaN')
                patched_eval += 1
        # 2) disable augment + PE in CONFIG
        if "'augment'" in line and 'True' in line:
            line = re.sub(r"True", "False", line, count=1)
            patched_aug += 1
        new.append(line)
    c['source'] = new
# Disable POS_ENC_CHANNELS in _lib.py is done separately
nb.write_text(json.dumps(data, indent=1))
print(f'eval-patches={patched_eval}  aug-patches={patched_aug}')

# Patch _lib.py POS_ENC_CHANNELS = 4 -> 0
lib = pathlib.Path('_lib.py')
t = lib.read_text()
t2 = re.sub(r'POS_ENC_CHANNELS\s*=\s*4', 'POS_ENC_CHANNELS = 0  # FIX_B disabled', t, count=1)
if t2 != t:
    lib.write_text(t2)
    print('PE channels disabled')
PY
}

while true; do
  sleep 60
  PID=$(pgrep -f run_overnight.py | head -1)
  if [ -z "$PID" ]; then
    echo "[$(date)] runner gone; watchdog exiting" >> "$LOG"
    exit 0
  fi
  # Look at the most-recent log file
  CUR=$(ls -t logs/*.log 2>/dev/null | head -1)
  [ -z "$CUR" ] && continue
  NAN=$(grep -cE "non-finite loss|nan/nan|NaN" "$CUR" 2>/dev/null || echo 0)
  if [ "$NAN" -ge 3 ]; then
    echo "[$(date)] NaN detected in $CUR ($NAN hits) -> killing + applying fix B" >> "$LOG"
    pkill -9 -f run_overnight.py 2>/dev/null
    pkill -9 -f ipykernel_launcher 2>/dev/null
    sleep 3
    # Archive broken results
    TS=$(date +%Y%m%d_%H%M%S)
    [ -d results ] && mv results "results_broken_$TS"
    [ -d logs ] && mv logs "logs_broken_$TS"
    mkdir -p logs
    apply_fix_B
    echo "[$(date)] relaunching run_overnight.py" >> "$LOG"
    nohup ../../../.venv/bin/python -u run_overnight.py > overnight_stdout.log 2>&1 &
    echo "[$(date)] new PID=$!" >> "$LOG"
    # Done; let watchdog continue monitoring the new run too
  fi
done
