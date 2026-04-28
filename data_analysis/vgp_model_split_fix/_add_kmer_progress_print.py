"""Add CSD3-friendly periodic 'features have been extracted' messages
alongside the existing tqdm bars in the k-mer featurization loops.

Idempotent: skips loops whose bodies already contain the
``[features extracted]`` marker.
"""
import json
from pathlib import Path

ROOT = Path('/Users/alexyang/Documents/Part III System Biology/TE Classification/data_analysis/vgp_model_split_fix')
MARK = "[features extracted]"

# Each entry: (notebook path, list of (old_block, new_block)).
# The blocks are the FULL for-loop body up to and including the append line.
JOBS = [
    # v4.2
    (
        ROOT / 'v4.2' / 'vgp_hybrid_v4.2.ipynb',
        [
            (
                '    all_kmer_features = []\n'
                '    for seq in tqdm(all_s, desc="Featurizing", leave=False):\n'
                '        X, _ = featurizer.featurize_sequence(seq)\n'
                '        all_kmer_features.append(X)\n',
                '    all_kmer_features = []\n'
                '    _n_total = len(all_s)\n'
                '    _print_every = max(1000, _n_total // 20)\n'
                '    for _i, seq in enumerate(tqdm(all_s, desc="Featurizing", leave=False), 1):\n'
                '        X, _ = featurizer.featurize_sequence(seq)\n'
                '        all_kmer_features.append(X)\n'
                '        if _i % _print_every == 0 or _i == _n_total:\n'
                '            print(f"  [features extracted] {_i}/{_n_total} features have been extracted", flush=True)\n',
            ),
            (
                'kmer_features = []\n'
                'for seq in tqdm(sample_s, desc="Featurizing"):\n'
                '    X, _ = featurizer.featurize_sequence(seq)\n'
                '    # Average over windows to get single vector per sequence\n'
                '    kmer_features.append(X.mean(axis=0))\n',
                'kmer_features = []\n'
                '_n_total = len(sample_s)\n'
                '_print_every = max(1000, _n_total // 20)\n'
                'for _i, seq in enumerate(tqdm(sample_s, desc="Featurizing"), 1):\n'
                '    X, _ = featurizer.featurize_sequence(seq)\n'
                '    # Average over windows to get single vector per sequence\n'
                '    kmer_features.append(X.mean(axis=0))\n'
                '    if _i % _print_every == 0 or _i == _n_total:\n'
                '        print(f"  [features extracted] {_i}/{_n_total} features have been extracted", flush=True)\n',
            ),
            (
                'all_kmer_features = []\n'
                'for seq in tqdm(all_s, desc="Featurizing", leave=False):\n'
                '    X_km, _ = featurizer_eval.featurize_sequence(seq)\n'
                '    all_kmer_features.append(X_km)\n',
                'all_kmer_features = []\n'
                '_n_total = len(all_s)\n'
                '_print_every = max(1000, _n_total // 20)\n'
                'for _i, seq in enumerate(tqdm(all_s, desc="Featurizing", leave=False), 1):\n'
                '    X_km, _ = featurizer_eval.featurize_sequence(seq)\n'
                '    all_kmer_features.append(X_km)\n'
                '    if _i % _print_every == 0 or _i == _n_total:\n'
                '        print(f"  [features extracted] {_i}/{_n_total} features have been extracted", flush=True)\n',
            ),
        ],
    ),
    # v4.3
    (
        ROOT / 'v4.3' / 'vgp_features_tpase_multiclass_v4.3.ipynb',
        [
            (
                '    all_kmer_features = []\n'
                '    for seq in tqdm(all_s, desc="Featurizing", leave=False):\n'
                '        X, _ = featurizer.featurize_sequence(seq)\n'
                '        all_kmer_features.append(X)\n',
                '    all_kmer_features = []\n'
                '    _n_total = len(all_s)\n'
                '    _print_every = max(1000, _n_total // 20)\n'
                '    for _i, seq in enumerate(tqdm(all_s, desc="Featurizing", leave=False), 1):\n'
                '        X, _ = featurizer.featurize_sequence(seq)\n'
                '        all_kmer_features.append(X)\n'
                '        if _i % _print_every == 0 or _i == _n_total:\n'
                '            print(f"  [features extracted] {_i}/{_n_total} features have been extracted", flush=True)\n',
            ),
            (
                'kmer_features = []\n'
                'for seq in tqdm(sample_s, desc="Featurizing"):\n'
                '    X, _ = featurizer.featurize_sequence(seq)\n'
                '    # Average over windows to get single vector per sequence\n'
                '    kmer_features.append(X.mean(axis=0))\n',
                'kmer_features = []\n'
                '_n_total = len(sample_s)\n'
                '_print_every = max(1000, _n_total // 20)\n'
                'for _i, seq in enumerate(tqdm(sample_s, desc="Featurizing"), 1):\n'
                '    X, _ = featurizer.featurize_sequence(seq)\n'
                '    # Average over windows to get single vector per sequence\n'
                '    kmer_features.append(X.mean(axis=0))\n'
                '    if _i % _print_every == 0 or _i == _n_total:\n'
                '        print(f"  [features extracted] {_i}/{_n_total} features have been extracted", flush=True)\n',
            ),
            (
                'all_kmer = []\n'
                'for s in tqdm(all_s, desc="Featurizing", leave=False):\n'
                '    X, _ = _featurizer_ov.featurize_sequence(s)\n'
                '    all_kmer.append(X)\n',
                'all_kmer = []\n'
                '_n_total = len(all_s)\n'
                '_print_every = max(1000, _n_total // 20)\n'
                'for _i, s in enumerate(tqdm(all_s, desc="Featurizing", leave=False), 1):\n'
                '    X, _ = _featurizer_ov.featurize_sequence(s)\n'
                '    all_kmer.append(X)\n'
                '    if _i % _print_every == 0 or _i == _n_total:\n'
                '        print(f"  [features extracted] {_i}/{_n_total} features have been extracted", flush=True)\n',
            ),
        ],
    ),
    # v4
    (
        ROOT / 'v4' / 'vgp_hybrid_v4.ipynb',
        [
            (
                '    all_kmer_features = []\n'
                '    for seq in tqdm(all_s, desc="Featurizing", leave=False):\n'
                '        X, _ = featurizer.featurize_sequence(seq)\n'
                '        all_kmer_features.append(X)\n',
                '    all_kmer_features = []\n'
                '    _n_total = len(all_s)\n'
                '    _print_every = max(1000, _n_total // 20)\n'
                '    for _i, seq in enumerate(tqdm(all_s, desc="Featurizing", leave=False), 1):\n'
                '        X, _ = featurizer.featurize_sequence(seq)\n'
                '        all_kmer_features.append(X)\n'
                '        if _i % _print_every == 0 or _i == _n_total:\n'
                '            print(f"  [features extracted] {_i}/{_n_total} features have been extracted", flush=True)\n',
            ),
            (
                'kmer_features = []\n'
                'for seq in tqdm(sample_s, desc="Featurizing"):\n'
                '    X, _ = featurizer.featurize_sequence(seq)\n'
                '    # Average over windows to get single vector per sequence\n'
                '    kmer_features.append(X.mean(axis=0))\n',
                'kmer_features = []\n'
                '_n_total = len(sample_s)\n'
                '_print_every = max(1000, _n_total // 20)\n'
                'for _i, seq in enumerate(tqdm(sample_s, desc="Featurizing"), 1):\n'
                '    X, _ = featurizer.featurize_sequence(seq)\n'
                '    # Average over windows to get single vector per sequence\n'
                '    kmer_features.append(X.mean(axis=0))\n'
                '    if _i % _print_every == 0 or _i == _n_total:\n'
                '        print(f"  [features extracted] {_i}/{_n_total} features have been extracted", flush=True)\n',
            ),
            (
                'all_kmer_features = []\n'
                'for seq in tqdm(all_s, desc="Featurizing", leave=False):\n'
                '    X_km, _ = featurizer_eval.featurize_sequence(seq)\n'
                '    all_kmer_features.append(X_km)\n',
                'all_kmer_features = []\n'
                '_n_total = len(all_s)\n'
                '_print_every = max(1000, _n_total // 20)\n'
                'for _i, seq in enumerate(tqdm(all_s, desc="Featurizing", leave=False), 1):\n'
                '    X_km, _ = featurizer_eval.featurize_sequence(seq)\n'
                '    all_kmer_features.append(X_km)\n'
                '    if _i % _print_every == 0 or _i == _n_total:\n'
                '        print(f"  [features extracted] {_i}/{_n_total} features have been extracted", flush=True)\n',
            ),
        ],
    ),
]


def patch_nb(nb_path: Path, jobs):
    nb = json.loads(nb_path.read_text())
    n_changed = 0
    for old, new in jobs:
        replaced_in_nb = False
        for c in nb['cells']:
            if c.get('cell_type') != 'code':
                continue
            s = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
            if MARK in s:
                # already patched somewhere; skip if this exact loop already
                # contains the marker (idempotency).
                if old not in s:
                    continue
            if old in s:
                new_s = s.replace(old, new, 1)
                c['source'] = new_s.splitlines(keepends=True)
                c['outputs'] = []
                c['execution_count'] = None
                replaced_in_nb = True
                n_changed += 1
                break
        if not replaced_in_nb:
            print(f"  [warn] anchor not found in {nb_path.name}: {old.splitlines()[1][:80]}")
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"[ok] {nb_path.name}: {n_changed} loop(s) patched")


for path, jobs in JOBS:
    if not path.exists():
        print(f"[warn] {path} not found, skipping")
        continue
    patch_nb(path, jobs)
