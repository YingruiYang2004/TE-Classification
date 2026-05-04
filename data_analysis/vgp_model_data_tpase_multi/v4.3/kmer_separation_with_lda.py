#!/usr/bin/env python3
"""Regenerate Figure 8 (k-mer feature separability) with an added LDA panel.

Recreates the same data source used by the original notebook cell
(`vgp_features_tpase_multiclass_v4.2.ipynb`, cells 31-32):
  - 5000-sequence stratified random sample (seed=42) from all_vgp_tes.fa
  - Top-level 3-class labels {DNA, LTR, LINE}
  - Canonical 7-mer hashed window features (KMER_K=7, KMER_DIM=2048,
    window=512, stride=256), mean-pooled across windows, L2-normalised.

Then produces:
  - kmer_separation_lda.png : 3-panel projection (PCA | LDA | t-SNE)
  - kmer_separation_cv.txt  : 5-fold CV macro F1 for LR / RF / LDA / majority

LDA is supervised, so its 2-D projection is the optimal linear projection
for class separation under Gaussian assumptions; expect noticeably tighter
clusters than PCA (unsupervised) and t-SNE (non-linear, neighbourhood-based).
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── Bring in featurizer + IO defs from the v4.2 notebook ──────────────────
HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "vgp_features_tpase_multiclass_v4.2.ipynb"
nb = json.load(open(NOTEBOOK))
ns: dict = {}
# Cells: imports(0), config(1), fasta/label utils(3), kmer featurizer(5)
for cell_idx in [0, 1, 3, 5]:
    exec("".join(nb["cells"][cell_idx]["source"]), ns)

read_fasta = ns["read_fasta"]
load_multiclass_labels = ns["load_multiclass_labels"]
KmerWindowFeaturizer = ns["KmerWindowFeaturizer"]
KEEP_CLASSES = ns["KEEP_CLASSES"]
KMER_K = ns["KMER_K"]
KMER_DIM = ns["KMER_DIM"]
KMER_WINDOW = ns["KMER_WINDOW"]
KMER_STRIDE = ns["KMER_STRIDE"]
FASTA_PATH = HERE / ns["FASTA_PATH"]
LABEL_PATH = HERE / ns["LABEL_PATH"]

# ── Reproduce sampling exactly (cell 31) ──────────────────────────────────
print("Loading FASTA + labels …")
headers, sequences = read_fasta(str(FASTA_PATH))
label_dict, class_dict = load_multiclass_labels(str(LABEL_PATH),
                                                keep_classes=KEEP_CLASSES)

sample_h, sample_s, sample_class = [], [], []
for h, s in zip(headers, sequences):
    if h in label_dict:
        sample_h.append(h)
        sample_s.append(s)
        sample_class.append(class_dict[h])

np.random.seed(42)
n_sample = min(5000, len(sample_h))
idx = np.random.choice(len(sample_h), n_sample, replace=False)
sample_h = [sample_h[i] for i in idx]
sample_s = [sample_s[i] for i in idx]
sample_class = np.array([sample_class[i] for i in idx])

print(f"Sampled {n_sample} sequences "
      f"({dict(zip(KEEP_CLASSES, np.bincount(sample_class)))})")

# ── Cache features so we don't recompute on re-runs ───────────────────────
CACHE = HERE / "kmer_features_cache_5000_seed42.npz"
if CACHE.exists():
    print(f"Loading cached features from {CACHE.name}")
    z = np.load(CACHE)
    X_kmer, y = z["X"], z["y"]
else:
    print(f"Computing 7-mer features (k={KMER_K}, dim={KMER_DIM}) …")
    feat = KmerWindowFeaturizer(k=KMER_K, dim=KMER_DIM,
                                window=KMER_WINDOW, stride=KMER_STRIDE,
                                add_pos=False, l2_normalize=True)
    X_kmer = np.zeros((n_sample, KMER_DIM), dtype=np.float32)
    for i, seq in enumerate(tqdm(sample_s, desc="featurize")):
        X, _ = feat.featurize_sequence(seq)
        X_kmer[i] = X.mean(axis=0)
    y = sample_class
    np.savez_compressed(CACHE, X=X_kmer, y=y)
    print(f"Cached features → {CACHE.name}")

print(f"X_kmer shape: {X_kmer.shape}")

# ── 5-fold CV macro-F1: LR / RF / LDA / majority ──────────────────────────
print("\nRunning 5-fold CV macro-F1 …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, clf in [
    ("LogReg",       LogisticRegression(max_iter=1000, random_state=42)),
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("LDA",          LinearDiscriminantAnalysis()),
]:
    sc = cross_val_score(clf, X_kmer, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    results[name] = (sc.mean(), sc.std())
    print(f"  {name:14s} F1 = {sc.mean():.4f} ± {sc.std():.4f}")
maj = np.bincount(y).max() / len(y)
results["Majority"] = (maj, 0.0)
print(f"  {'Majority':14s} acc = {maj:.4f}")

with open(HERE / "kmer_separation_cv.txt", "w") as f:
    f.write("5-fold stratified CV, macro F1 (k=7, canonical, mean-pooled, L2-normalised)\n")
    f.write(f"n={len(y)}  classes={KEEP_CLASSES}  counts={np.bincount(y).tolist()}\n\n")
    for k, (m, s) in results.items():
        f.write(f"{k:14s} {m:.4f} ± {s:.4f}\n")

# ── Projections: PCA / LDA / t-SNE ────────────────────────────────────────
print("\nComputing PCA …")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_kmer)

print("Computing LDA (supervised, 2 components for 3 classes) …")
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_kmer, y)

print("Computing t-SNE on 2000-sample subset …")
np.random.seed(42)
tsne_idx = np.random.choice(len(X_kmer), min(2000, len(X_kmer)), replace=False)
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42,
              init="pca", learning_rate="auto").fit_transform(X_kmer[tsne_idx])
y_tsne = y[tsne_idx]

# ── Plot: PCA | LDA | t-SNE ───────────────────────────────────────────────
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # DNA, LTR, LINE
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
for i, cls in enumerate(KEEP_CLASSES):
    m = y == i
    ax.scatter(X_pca[m, 0], X_pca[m, 1], c=colors[i], label=cls, alpha=0.5, s=18)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
ax.set_title("PCA (unsupervised)")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
for i, cls in enumerate(KEEP_CLASSES):
    m = y == i
    ax.scatter(X_lda[m, 0], X_lda[m, 1], c=colors[i], label=cls, alpha=0.5, s=18)
evr = lda.explained_variance_ratio_
ax.set_xlabel(f"LD1 ({evr[0]:.1%} between-class var)")
ax.set_ylabel(f"LD2 ({evr[1]:.1%} between-class var)")
ax.set_title(f"LDA (supervised) — CV F1 = {results['LDA'][0]:.3f}")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
for i, cls in enumerate(KEEP_CLASSES):
    m = y_tsne == i
    ax.scatter(X_tsne[m, 0], X_tsne[m, 1], c=colors[i], label=cls, alpha=0.5, s=18)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE (unsupervised, non-linear)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle(
    "K-mer (canonical 7-mer) feature separability for top-level classes — "
    f"LR F1 = {results['LogReg'][0]:.3f}, "
    f"RF F1 = {results['RandomForest'][0]:.3f}, "
    f"LDA F1 = {results['LDA'][0]:.3f}",
    fontsize=13, fontweight='bold')
plt.tight_layout()
out = HERE / "kmer_separation_lda.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nWrote {out.name}")
print(f"Wrote kmer_separation_cv.txt")
