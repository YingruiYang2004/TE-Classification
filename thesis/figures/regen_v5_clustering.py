"""Regenerate §3.7 numbers/figures from the v5 rerun embeddings.

Inputs:
  data_analysis/vgp_model_split_fix/v5/contrastive_embeddings.csv
    (header, superfamily, sf_id, embed_0..embed_255)

Outputs (written to thesis/figures/):
  clustering_umap.png       UMAP coloured by HDBSCAN cluster (top-K + grey noise)
  v5_clustering_metrics.txt Plain-text dump of all numbers used in §3.7

Methodology (mirrors the v5 notebook):
  - HDBSCAN on the 256-d fused embedding, min_cluster_size=100, min_samples=20
    -> headline cluster count + sil/ARI/NMI vs annotated superfamily labels.
  - Ward agglomerative on the same 256-d embedding at n_clusters=50
    -> homogeneity / completeness / ARI / NMI vs annotated SFs (subcluster fragmentation).
  - Per-SF mean L2 to per-SF centroid (subsampled to 200 per SF where n>200).
  - UMAP (n_neighbors=30, min_dist=0.1, random_state=42) for the figure.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score,
)

ROOT = Path(__file__).resolve().parents[2]
CSV  = ROOT / "data_analysis/vgp_model_split_fix/v5/contrastive_embeddings.csv"
OUT  = Path(__file__).resolve().parent
RNG  = np.random.default_rng(42)

print(f"Loading {CSV} ...")
df = pd.read_csv(CSV)
emb = df[[f"embed_{i}" for i in range(256)]].to_numpy(dtype=np.float32)
sf_id = df["sf_id"].to_numpy()
sf_name = df["superfamily"].to_numpy()
N, D = emb.shape
print(f"  N={N} sequences, D={D} dims, {df['superfamily'].nunique()} superfamilies")

# ---------- 1. Per-superfamily L2 compactness ----------
print("\n=== Per-superfamily L2 compactness (subsample 200) ===")
rows = []
for sf in sorted(df["superfamily"].unique()):
    idx = np.where(sf_name == sf)[0]
    n = len(idx)
    if n > 200:
        sel = RNG.choice(idx, size=200, replace=False)
    else:
        sel = idx
    centroid = emb[sel].mean(axis=0)
    d = np.linalg.norm(emb[sel] - centroid, axis=1).mean()
    rows.append((sf, n, float(d)))
    print(f"  {sf:20s} n={n:5d}  L2={d:.2f}")
l2_table = pd.DataFrame(rows, columns=["superfamily", "n", "mean_L2"]).sort_values("mean_L2")

# ---------- 2. HDBSCAN clustering ----------
# Notebook default (mcs=100) collapses to 3 macro-blobs on this 1.77k-point
# validation set; mcs=10/ms=10 is the smallest setting that recovers v4-style
# fine-grained substructure (~10 SF-pure subclusters) without over-fragmenting.
MCS, MS = 10, 10
print(f"\n=== HDBSCAN (min_cluster_size={MCS}, min_samples={MS}) ===")
hdb = HDBSCAN(min_cluster_size=MCS, min_samples=MS)
hdb_labels = hdb.fit_predict(emb)
hdb_n_clusters = int((np.unique(hdb_labels) >= 0).sum())
hdb_n_noise = int((hdb_labels == -1).sum())
print(f"  clusters={hdb_n_clusters}, noise={hdb_n_noise}/{N}")
mask = hdb_labels >= 0
hdb_sil = silhouette_score(emb[mask], hdb_labels[mask]) if hdb_n_clusters > 1 else float("nan")
hdb_ari = adjusted_rand_score(sf_id[mask], hdb_labels[mask])
hdb_nmi = normalized_mutual_info_score(sf_id[mask], hdb_labels[mask])
print(f"  silhouette={hdb_sil:.4f}  ARI={hdb_ari:.4f}  NMI={hdb_nmi:.4f}")

# ---------- 3. Ward agglomerative (subcluster fragmentation) ----------
WARD_K = 50
print(f"\n=== Ward agglomerative (n_clusters={WARD_K}) ===")
ward = AgglomerativeClustering(n_clusters=WARD_K, linkage="ward")
ward_labels = ward.fit_predict(emb)
ward_ari  = adjusted_rand_score(sf_id, ward_labels)
ward_nmi  = normalized_mutual_info_score(sf_id, ward_labels)
ward_hom  = homogeneity_score(sf_id, ward_labels)
ward_comp = completeness_score(sf_id, ward_labels)
print(f"  ARI={ward_ari:.4f}  NMI={ward_nmi:.4f}  hom={ward_hom:.4f}  comp={ward_comp:.4f}")

# ---------- 4. UMAP -> clustering_umap.png ----------
print("\n=== UMAP projection (n_neighbors=30, min_dist=0.1) ===")
import umap
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
xy = reducer.fit_transform(emb)
print(f"  done; xy shape={xy.shape}")

TOP_K = 20
unique, counts = np.unique(hdb_labels[hdb_labels >= 0], return_counts=True)
top = unique[np.argsort(-counts)][:TOP_K]
fig, ax = plt.subplots(figsize=(11, 8))
ax.scatter(xy[hdb_labels == -1, 0], xy[hdb_labels == -1, 1],
           s=8, c="lightgrey", alpha=0.5, label="noise")
cmap = plt.colormaps.get_cmap("tab20")
for i, c in enumerate(top):
    sel = hdb_labels == c
    ax.scatter(xy[sel, 0], xy[sel, 1], s=14, color=cmap(i % 20),
               alpha=0.8, label=f"C{int(c)} (n={int(sel.sum())})")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title(f"v5 contrastive embeddings — HDBSCAN clusters "
             f"(mcs={MCS}, ms={MS}; {hdb_n_clusters} total; "
             f"top {min(TOP_K, hdb_n_clusters)} shown)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
plt.tight_layout()
out_png = OUT / "clustering_umap.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"  wrote {out_png}")

# ---------- 5. Dump numbers for the LaTeX edit ----------
metrics_path = OUT / "v5_clustering_metrics.txt"
with metrics_path.open("w") as fh:
    fh.write("# v5 contrastive clustering — recomputed for §3.7\n")
    fh.write(f"N_sequences = {N}\n")
    fh.write(f"N_superfamilies = {df['superfamily'].nunique()}\n\n")
    fh.write("## L2 compactness (sorted ascending)\n")
    for _, r in l2_table.iterrows():
        fh.write(f"{r.superfamily:20s} n={int(r.n):5d}  mean_L2={r.mean_L2:.2f}\n")
    fh.write(f"\n## HDBSCAN (min_cluster_size={MCS}, min_samples={MS})\n")
    fh.write(f"clusters = {hdb_n_clusters}\n")
    fh.write(f"noise    = {hdb_n_noise}/{N}\n")
    fh.write(f"silhouette = {hdb_sil:.4f}\n")
    fh.write(f"ARI        = {hdb_ari:.4f}\n")
    fh.write(f"NMI        = {hdb_nmi:.4f}\n")
    fh.write(f"\n## Ward agglomerative (n_clusters={WARD_K})\n")
    fh.write(f"ARI          = {ward_ari:.4f}\n")
    fh.write(f"NMI          = {ward_nmi:.4f}\n")
    fh.write(f"homogeneity  = {ward_hom:.4f}\n")
    fh.write(f"completeness = {ward_comp:.4f}\n")
print(f"  wrote {metrics_path}")
print("\nDONE.")
