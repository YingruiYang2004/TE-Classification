"""Quick 5-epoch full-data MPS smoke for three_class_balanced.

Patches CONFIG to: variant=three_class_balanced, smoke=False, epochs=5, patience=10.
Runs 02_train.ipynb in-memory (does NOT overwrite the original .ipynb).
Streams cell outputs to logs/smoke5.log so we can monitor live.
"""
from pathlib import Path
import sys, time, traceback
import nbformat, nbclient

HERE = Path(__file__).parent.resolve()
LOG_DIR = HERE / 'logs'; LOG_DIR.mkdir(exist_ok=True)
LOG = LOG_DIR / 'smoke5.log'

nb = nbformat.read(HERE / '02_train.ipynb', as_version=4)
for c in nb.cells:
    if c.cell_type != 'code':
        continue
    src = c.source
    if "'variant'" in src and 'CONFIG' in src[:30]:
        for v in ('three_class_balanced', 'three_class_unbalanced', 'binary_dna'):
            src = src.replace(f"'variant': '{v}'", "'variant': 'three_class_balanced'")
        src = src.replace("'smoke':   True", "'smoke':   False")
        # Force epochs=5, patience=10
        import re
        src = re.sub(r"'epochs':\s*\d+", "'epochs': 5", src)
        src = re.sub(r"'patience':\s*\d+", "'patience': 10", src)
        c.source = src
        break

t0 = time.time()
log = LOG.open('w')
log.write(f'# 5-epoch MPS smoke\n# start : {time.ctime()}\n\n'); log.flush()
client = nbclient.NotebookClient(
    nb, timeout=60*60*4, kernel_name='python3',
    resources={'metadata': {'path': str(HERE)}},
)
ok = True
try:
    client.execute()
except Exception:
    ok = False
    traceback.print_exc(file=log)
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    for o in c.get('outputs', []):
        if o.get('output_type') == 'stream':
            log.write(f'--- cell #{i} {o.get("name", "stream")} ---\n')
            log.write(o['text'])
            if not o['text'].endswith('\n'):
                log.write('\n')
        elif o.get('output_type') == 'error':
            log.write(f'--- cell #{i} ERROR ---\n')
            log.write('\n'.join(o.get('traceback', [])))
            log.write('\n')
elapsed = (time.time() - t0) / 60
log.write(f'\n# end : {time.ctime()}\n# elapsed : {elapsed:.1f} min\n# success : {ok}\n')
log.close()
print(f'smoke5 {"OK" if ok else "FAILED"}  {elapsed:.1f} min  -> {LOG}')
sys.exit(0 if ok else 1)
