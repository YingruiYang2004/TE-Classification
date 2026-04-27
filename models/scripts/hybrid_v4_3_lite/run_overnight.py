#!/usr/bin/env python
"""Run the three v4.3-lite training variants then the evaluation notebook.

Usage: python run_overnight.py
Logs go to ./logs/<variant>.log and ./logs/eval.log.
"""
from pathlib import Path
import sys
import time
import traceback
import nbformat
import nbclient

HERE = Path(__file__).parent.resolve()
LOG_DIR = HERE / 'logs'
LOG_DIR.mkdir(exist_ok=True)

VARIANTS = ['three_class_balanced', 'three_class_unbalanced', 'binary_dna', 'binary_dna_natural']


def run_notebook(nb_path: Path, log_path: Path, variant: str | None = None,
                 timeout: int = 60 * 60 * 8):
    """Execute a notebook in-place. If variant is given, patch CONFIG to that variant
    and force smoke=False. Writes the executed notebook back to the same path."""
    nb = nbformat.read(nb_path, as_version=4)
    if variant is not None:
        for c in nb.cells:
            if c.cell_type != 'code':
                continue
            src = c.source
            if "'variant'" in src and 'CONFIG' in src[:30]:
                # Force this variant + ensure smoke is False.
                for v in VARIANTS:
                    src = src.replace(f"'variant': '{v}'", f"'variant': '{variant}'")
                src = src.replace("'smoke':   True", "'smoke':   False")
                c.source = src
                break
    t0 = time.time()
    log = log_path.open('w')
    log.write(f'# notebook: {nb_path.name}\n# variant : {variant}\n# start   : {time.ctime()}\n\n')
    log.flush()
    client = nbclient.NotebookClient(
        nb, timeout=timeout, kernel_name='python3',
        resources={'metadata': {'path': str(HERE)}},
    )
    try:
        client.execute()
        ok = True
    except Exception:
        ok = False
        traceback.print_exc(file=log)
    # Persist the executed notebook (with outputs) so we can inspect later.
    out_path = nb_path.with_name(nb_path.stem + (f'_{variant}' if variant else '') + '.executed.ipynb')
    nbformat.write(nb, out_path)
    # Also dump every code cell's stdout to the log.
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
    log.write(f'\n# end     : {time.ctime()}\n# elapsed : {elapsed:.1f} min\n# success : {ok}\n')
    log.close()
    return ok, elapsed


def main():
    overall = []
    for v in VARIANTS:
        print(f'\n========== {v} ==========', flush=True)
        ok, mins = run_notebook(HERE / '02_train.ipynb', LOG_DIR / f'{v}.log', variant=v)
        print(f'{v}: {"OK" if ok else "FAILED"}  ({mins:.1f} min)', flush=True)
        overall.append((v, ok, mins))
        if not ok:
            print(f'  -> see {LOG_DIR / (v + ".log")}', flush=True)
    print('\n========== eval ==========', flush=True)
    ok, mins = run_notebook(HERE / '03_eval_and_compare.ipynb', LOG_DIR / 'eval.log', variant=None)
    print(f'eval: {"OK" if ok else "FAILED"}  ({mins:.1f} min)', flush=True)
    overall.append(('eval', ok, mins))

    print('\n========== SUMMARY ==========', flush=True)
    for name, ok, mins in overall:
        print(f'  {name:30s}  {"OK " if ok else "FAIL"}  {mins:6.1f} min', flush=True)
    print(f'\nLogs: {LOG_DIR}')
    sys.exit(0 if all(ok for _, ok, _ in overall) else 1)


if __name__ == '__main__':
    main()
