"""
Diagnostic: locate the first module whose forward output goes non-finite on MPS,
using REAL training data and the same CONFIG as 02_train.ipynb.

Strategy:
  - Load a small but representative subset (~800 seqs) of three_class_balanced.
  - Build the v4.3-lite model on MPS with the same loss / opt as the notebook.
  - Register a forward hook on every named submodule that records
    (max-abs, min, has-NaN, has-Inf) of each output tensor.
  - Run training steps; after every step also print param max-abs / grad max-abs
    for a fixed list of "interesting" sub-blocks.
  - As soon as ANY hook sees NaN/Inf, print the full per-module table and stop.

Run:
    cd models/scripts/hybrid_v4_3_lite
    ../../../.venv/bin/python -u _diag_mps.py [--device mps|cpu] [--n 800] [--steps 200]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import _lib as L  # noqa: E402

EXCLUDED_GENOMES = {'mOrnAna', 'bTaeGut', 'rAllMis'}
FASTA = Path(__file__).parent / '../../../data/vgp/all_vgp_tes.fa'
LABELS = Path(__file__).parent / '../../../data/vgp/20260120_features_sf'
CACHE = Path(__file__).parent / '_diag_cache.npz'


def load_data(n_keep: int, seed: int = 42):
    print(f'[data] loading FASTA + labels…', flush=True)
    headers, seqs = L.read_fasta(FASTA)
    label_dict, class_dict, _ = L.load_labels(LABELS, keep_classes=('DNA', 'LTR', 'LINE'))
    prep = L.filter_and_subsample(
        headers, seqs, label_dict, class_dict,
        exclude_genomes=EXCLUDED_GENOMES,
        min_class_count=100, max_per_sf=3000, random_state=seed,
    )
    h, s = prep['headers'], prep['sequences']
    top, sf = prep['toplevel'], prep['sf']
    n_sf = len(prep['sf_names'])
    print(f'[data] full balanced trainval pool: {len(h)} | n_sf={n_sf}', flush=True)
    rng = np.random.default_rng(seed)
    keep = sorted(rng.choice(len(h), size=min(n_keep, len(h)), replace=False).tolist())
    h = [h[i] for i in keep]; s = [s[i] for i in keep]
    top = top[keep]; sf = sf[keep]
    print(f'[data] subsampled to {len(h)} for diagnostic', flush=True)
    return h, s, top, sf, n_sf


def featurize(seqs, force=False):
    if CACHE.exists() and not force:
        try:
            d = np.load(CACHE, allow_pickle=True)
            if int(d['n']) == len(seqs):
                print(f'[feat] loaded cache {CACHE.name} ({len(seqs)} seqs)', flush=True)
                return [d[f'k{i}'] for i in range(len(seqs))]
        except Exception:
            pass
    print(f'[feat] featurising {len(seqs)} seqs…', flush=True)
    t0 = time.time()
    feat = L.KmerWindowFeaturizer(k=L.KMER_K, dim=L.KMER_DIM, window=L.KMER_WINDOW, stride=L.KMER_STRIDE)
    out = []
    for i, s in enumerate(seqs):
        out.append(feat.featurize_sequence(s))
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(seqs)}  elapsed {time.time()-t0:.1f}s', flush=True)
    payload = {'n': np.array(len(seqs))}
    for i, k in enumerate(out):
        payload[f'k{i}'] = k
    np.savez(CACHE, **payload)
    print(f'[feat] done in {time.time()-t0:.1f}s, cached → {CACHE.name}', flush=True)
    return out


# -----------------------------------------------------------------------------
# Forward-hook stats
# -----------------------------------------------------------------------------

class Stats:
    def __init__(self):
        # name -> (last_max_abs, last_has_nan, last_has_inf)
        self.s: Dict[str, Tuple[float, bool, bool]] = {}
        self.first_bad: Tuple[str, str] | None = None  # (module_name, kind)

    def update(self, name: str, t: torch.Tensor):
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            return
        with torch.no_grad():
            has_nan = bool(torch.isnan(t).any().item())
            has_inf = bool(torch.isinf(t).any().item())
            if has_nan or has_inf:
                m = float('nan')
            else:
                m = float(t.detach().abs().max().item())
        self.s[name] = (m, has_nan, has_inf)
        if (has_nan or has_inf) and self.first_bad is None:
            self.first_bad = (name, 'NaN' if has_nan else 'Inf')


def attach_hooks(model: nn.Module, stats: Stats):
    handles = []
    for name, mod in model.named_modules():
        if name == '':
            continue
        # Skip pure containers; we still want their children individually
        if len(list(mod.children())) > 0 and not isinstance(mod, (nn.MultiheadAttention, nn.Sequential)):
            # Hook on Sequentials and MHA too because they're the high-level ones we care about
            pass
        def make_hook(n):
            def _hook(_m, _inp, out):
                if isinstance(out, tuple):
                    for j, o in enumerate(out):
                        stats.update(f'{n}[{j}]', o)
                else:
                    stats.update(n, out)
            return _hook
        handles.append(mod.register_forward_hook(make_hook(name)))
    return handles


def print_stats_table(stats: Stats, top_k: int = 20, header: str = ''):
    if header:
        print(header, flush=True)
    rows = sorted(stats.s.items(), key=lambda kv: -(0 if np.isnan(kv[1][0]) else kv[1][0]))
    print(f'  {"module":50s}  {"max|.|":>12s}  NaN  Inf', flush=True)
    for name, (m, n, i) in rows[:top_k]:
        flag_n = 'Y' if n else '.'
        flag_i = 'Y' if i else '.'
        print(f'  {name[:50]:50s}  {m:>12.3e}  {flag_n:>3s}  {flag_i:>3s}', flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='mps')
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--clamp', type=float, default=30.0)
    ap.add_argument('--no-clamp', action='store_true')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--ls', type=float, default=0.1, help='label smoothing')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f'\n=== device={device} | n={args.n} | steps={args.steps} | lr={args.lr} | '
          f'clamp={"OFF" if args.no_clamp else args.clamp} | bs={args.batch} ===\n', flush=True)

    h, s, top, sf, n_sf = load_data(args.n, seed=args.seed)
    k = featurize(s)

    n_top = 3
    ds = L.HybridDataset(h, s, top, sf, k, augment=L.AugmentConfig(enabled=True), rng_seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=L.collate_hybrid)

    model = L.HybridTEClassifierV43Lite(num_toplevel=n_top, num_superfamilies=n_sf).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[model] params={n_params:,}', flush=True)

    top_w = L.compute_class_weights(top, n_top, mode='inv_sqrt')
    sf_w  = L.compute_class_weights(sf,  n_sf,  mode='inv_sqrt')
    top_w_t = torch.tensor(top_w, dtype=torch.float32, device=device)
    sf_w_t  = torch.tensor(sf_w,  dtype=torch.float32, device=device)
    print(f'[loss] top_w {top_w_t.tolist()}', flush=True)
    print(f'[loss] sf_w  min={sf_w_t.min().item():.3f} max={sf_w_t.max().item():.3f}', flush=True)

    loss_top = nn.CrossEntropyLoss(label_smoothing=args.ls, weight=top_w_t).to(device)
    loss_sf  = nn.CrossEntropyLoss(label_smoothing=args.ls, weight=sf_w_t).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    stats = Stats()
    handles = attach_hooks(model, stats)

    step = 0
    t0 = time.time()
    try:
        for epoch in range(100):
            for _, X, mask, Y_top, Y_sf, x_g, ei, bv in loader:
                X, mask, Y_top, Y_sf, x_g, ei, bv = (
                    t.to(device) for t in (X, mask, Y_top, Y_sf, x_g, ei, bv)
                )
                model.train()
                top_l, sf_l, gates = model(X, mask, x_g, ei, bv)
                if not args.no_clamp:
                    top_l = top_l.clamp(-args.clamp, args.clamp)
                    sf_l  = sf_l.clamp(-args.clamp, args.clamp)
                l_top = loss_top(top_l, Y_top)
                l_sf  = loss_sf(sf_l, Y_sf)
                loss = l_top + l_sf

                # Per-step short line
                with torch.no_grad():
                    g_mean = gates.mean(0).tolist() if torch.isfinite(gates).all() else [float('nan'), float('nan')]
                    tl_max = float(top_l.detach().abs().max().item())
                    sl_max = float(sf_l.detach().abs().max().item())
                bad = (not torch.isfinite(loss)) or (stats.first_bad is not None)
                step += 1
                marker = ' !!' if bad else ''
                print(f'step {step:4d}  loss={loss.item():+.4e}  l_top={l_top.item():+.4e}  l_sf={l_sf.item():+.4e}  '
                      f'|top_l|max={tl_max:.2e}  |sf_l|max={sl_max:.2e}  gate={g_mean}{marker}', flush=True)
                if bad:
                    print('\n[!] non-finite detected — full forward stats:', flush=True)
                    print_stats_table(stats, top_k=60)
                    if stats.first_bad:
                        print(f'\n[!] FIRST BAD MODULE: {stats.first_bad[0]}  ({stats.first_bad[1]})', flush=True)
                    return
                opt.zero_grad(); loss.backward()
                # Param/grad scan
                if step in (1, 5, 10, 20, 40, 80, 150) or step == args.steps:
                    rows = []
                    for n, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        rows.append((n, float(p.detach().abs().max().item()),
                                        float(p.grad.detach().abs().max().item())))
                    rows.sort(key=lambda r: -r[2])
                    print(f'  --- step {step} top-10 grad max-abs ---', flush=True)
                    for n, pmax, gmax in rows[:10]:
                        print(f'    {n[:55]:55s}  |w|={pmax:.2e}  |g|={gmax:.2e}', flush=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step >= args.steps:
                    break
            if step >= args.steps:
                break
    finally:
        for h_ in handles:
            h_.remove()
    print(f'\n[done] {step} steps in {time.time()-t0:.1f}s — no NaN encountered', flush=True)
    print_stats_table(stats, top_k=30, header='\nfinal forward stats (top by max-abs):')


if __name__ == '__main__':
    main()
