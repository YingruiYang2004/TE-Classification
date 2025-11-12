#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lsh_cluster.py — Pure-Python approximate similarity clustering for DNA sequences.
No external bio tools required. Uses:
- canonical k-mers (reverse-complement invariant)
- MinHash sketches with multiple hash functions
- LSH banding to avoid all-vs-all comparisons
- Single-linkage clustering over candidate edges

This is approximate but scales to ~100k sequences with sensible params.

Usage:
python lsh_cluster.py --fasta seqs.fa --k 9 --num-hash 100 --bands 20 --rows 5 --jaccard 0.2 --out groups.tsv

Interpretation:
- k=9 or 10 often good for TE-scale motifs.
- num-hash = bands * rows.
- jaccard threshold is on k-mer sets; roughly identity-like but not identical.
"""
import argparse, re, sys
from pathlib import Path
from collections import defaultdict, deque

RC = str.maketrans('ACGTN', 'TGCAN')

def clean(s): return re.sub(r'[^ACGTN]', 'N', s.upper())
def revcomp(s): return s.translate(RC)[::-1]

def canonical_kmers(seq, k):
    L = len(seq)
    for i in range(L-k+1):
        kmer = seq[i:i+k]
        rck  = revcomp(kmer)
        yield kmer if kmer <= rck else rck

def read_fasta(path: Path):
    headers, seqs = [], []
    with open(path, 'r') as f:
        h, buf = None, []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if h is not None:
                    headers.append(h); seqs.append(''.join(buf))
                h = line[1:].strip(); buf = []
            else:
                buf.append(line)
        if h is not None:
            headers.append(h); seqs.append(''.join(buf))
    return headers, seqs

# simple 64-bit hash family
def hash64(x, seed):
    # SplitMix64-ish
    h = (hash(x) ^ seed) & 0xFFFFFFFFFFFFFFFF
    h = (h + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    h = (h ^ (h >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    h = h ^ (h >> 31)
    return h

def minhash(sketch_set, seeds):
    # sketch_set: iterable of strings (k-mers)
    mins = [0xFFFFFFFFFFFFFFFF] * len(seeds)
    for token in sketch_set:
        for i, s in enumerate(seeds):
            hv = hash64(token, s)
            if hv < mins[i]: mins[i] = hv
    return mins

def lsh_buckets(mins, bands, rows):
    # yield (band_index, signature) for each band
    for b in range(bands):
        start = b*rows
        sig = tuple(mins[start:start+rows])
        yield (b, sig)

def components(nodes, adj):
    seen, comps = set(), []
    for u in nodes:
        if u in seen: continue
        q, cur = deque([u]), []
        seen.add(u)
        while q:
            v = q.popleft(); cur.append(v)
            for w in adj[v]:
                if w not in seen:
                    seen.add(w); q.append(w)
        comps.append(cur)
    return comps

def jaccard(a, b):
    inter = len(a & b)
    if not inter: return 0.0
    return inter / float(len(a) + len(b) - inter)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fasta', required=True, type=Path)
    ap.add_argument('--k', type=int, default=9)
    ap.add_argument('--num-hash', type=int, default=100)
    ap.add_argument('--bands', type=int, default=20, help='LSH bands; num_hash must equal bands*rows')
    ap.add_argument('--rows', type=int, default=5, help='Rows per band')
    ap.add_argument('--jaccard', type=float, default=0.2, help='Verify edges at this Jaccard threshold')
    ap.add_argument('--out', type=Path, default=Path('groups.tsv'))
    args = ap.parse_args()
    if args.num_hash != args.bands * args.rows:
        sys.exit("num-hash must equal bands*rows")

    print("Reading FASTA...")
    headers, seqs = read_fasta(args.fasta)
    n = len(headers)
    print(f"Loaded {n} sequences.")

    print("Building k-mer sets...")
    kmersets = []
    for s in seqs:
        s = clean(s)
        kmersets.append(set(canonical_kmers(s, args.k)))

    # seeds for hash functions
    seeds = [i*0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF for i in range(args.num_hash)]

    print("Sketching with MinHash...")
    sketches = [minhash(kset, seeds) for kset in kmersets]

    print("LSH bucketing...")
    buckets = defaultdict(list)  # (band, sig) -> list of indices
    for idx, mins in enumerate(sketches):
        for key in lsh_buckets(mins, args.bands, args.rows):
            buckets[key].append(idx)

    print("Building candidate edges...")
    adj = defaultdict(set)
    for (_, _sig), idxs in buckets.items():
        if len(idxs) < 2: continue
        base = idxs[0]
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                a, b = idxs[i], idxs[j]
                adj[a].add(b); adj[b].add(a)

    print("Verifying candidates by Jaccard...")
    # filter edges that don't meet Jaccard threshold
    adj2 = defaultdict(set)
    for a, nbrs in adj.items():
        for b in nbrs:
            if jaccard(kmersets[a], kmersets[b]) >= args.jaccard:
                adj2[a].add(b); adj2[b].add(a)

    print("Finding connected components...")
    nodes = list(range(n))
    comps = components(nodes, adj2)
    print(f"Found {len(comps)} clusters.")

    print("Writing groups.tsv ...")
    with open(args.out, 'w') as w:
        w.write('header\tcluster\n')
        for cid, comp in enumerate(comps):
            for i in comp:
                w.write(f"{headers[i]}\tC{cid}\n")

    print("Done:", args.out)

if __name__ == "__main__":
    main()
