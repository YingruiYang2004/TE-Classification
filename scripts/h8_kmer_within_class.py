"""H8: Within-class k-mer superfamily F1 (5-fold CV, Logistic Regression, k=5)"""
import sys, numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

FASTA   = "data/vgp/all_vgp_tes.fa"
LABELS  = "data/vgp/20260120_features_sf"
EXCLUDE = {'mOrnAna','bTaeGut','rAllMis'}

_ASCII_MAP = np.full(256, 4, dtype=np.uint8)
for ch, v in [("A",0),("C",1),("G",2),("T",3),("a",0),("c",1),("g",2),("t",3)]:
    _ASCII_MAP[ord(ch)] = v
_COMP = np.array([3,2,1,0], dtype=np.uint8)

def read_fasta(path):
    headers, seqs = [], []
    h, buf = None, []
    with open(path) as f:
        for line in f:
            if not line: continue
            if line[0] == '>':
                if h: seqs.append(''.join(buf).upper()); buf = []
                h = line[1:].strip(); headers.append(h)
            else:
                buf.append(line.strip())
    if h: seqs.append(''.join(buf).upper())
    return headers, seqs

def seq_to_kmer(seq, k=5):
    arr = _ASCII_MAP[np.frombuffer(seq.encode(), dtype=np.uint8)]
    vec = np.zeros(4**k // 2 + 2, dtype=np.float32)
    for i in range(len(arr) - k + 1):
        seg = arr[i:i+k]
        if any(v >= 4 for v in seg): continue
        fwd = rev = 0
        for v in seg:            fwd = (fwd << 2) | int(v)
        for v in seg[::-1]:      rev = (rev << 2) | int(_COMP[min(v,3)])
        code = min(fwd, rev)
        if code < len(vec): vec[code] += 1
    s = vec.sum()
    if s > 0: vec /= s
    return vec

def genome_id(h):
    nm = h.split('#')[0]
    return nm.rsplit('-',1)[-1]

print("Loading labels...", flush=True)
label_map = {}
with open(LABELS) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            label_map[parts[0].lstrip('>')] = parts[1]

print("Loading FASTA...", flush=True)
headers, seqs = read_fasta(FASTA)

by_class = defaultdict(list)
for h, s in zip(headers, seqs):
    if genome_id(h) in EXCLUDE: continue
    lbl = label_map.get(h)
    if not lbl: continue
    parts = lbl.split('/')
    if len(parts) < 2: continue
    cls, sf = parts[0], parts[1]
    if cls not in ('DNA','LTR','LINE'): continue
    by_class[cls].append((s, sf))

print("Computing k-mer features (k=5)...", flush=True)
results = {}
for cls in ['DNA','LTR','LINE']:
    items = by_class[cls]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(items))[:2000]
    items = [items[i] for i in idx]

    sf_counts = Counter(sf for _,sf in items)
    print(f"  {cls}: {len(items)} seqs, SFs={dict(sf_counts.most_common(6))}", flush=True)

    # Filter out SFs with fewer than 10 samples (too few for stratified CV)
    sf_ok = {sf for sf, cnt in Counter(sf for _,sf in items).items() if cnt >= 10}
    items = [(s, sf) for s, sf in items if sf in sf_ok]
    if not items:
        print(f"  {cls}: no SFs with >=10 samples", flush=True)
        continue
    sf_counts_ok = Counter(sf for _,sf in items)
    print(f"  {cls}: after SF filter: {dict(sf_counts_ok)}", flush=True)

    X = np.array([seq_to_kmer(s) for s,_ in items])
    labels_sf = [sf for _,sf in items]
    sf_set = sorted(set(labels_sf))
    sf2id = {s:i for i,s in enumerate(sf_set)}
    y = np.array([sf2id[s] for s in labels_sf])

    min_count = min(Counter(y).values())
    n_splits = min(5, min_count)
    if n_splits < 2:
        print(f"  {cls}: min class count {min_count}, skip", flush=True)
        continue

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1s = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=300, C=1.0, random_state=42)
        clf.fit(X[tr], y[tr])
        f1s.append(f1_score(y[te], clf.predict(X[te]), average='macro', zero_division=0))

    results[cls] = {'mean': np.mean(f1s), 'std': np.std(f1s), 'n_sf': len(sf_set), 'n_seq': len(items)}
    print(f"  {cls}: macro F1 = {np.mean(f1s):.3f} ± {np.std(f1s):.3f}", flush=True)

print("\n=== FINAL RESULTS ===")
for cls in ['DNA','LTR','LINE']:
    if cls in results:
        r = results[cls]
        print(f"{cls}: macro F1 = {r['mean']:.3f} ± {r['std']:.3f}  ({r['n_sf']} SFs, n={r['n_seq']})")
