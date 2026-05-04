#!/usr/bin/env python3
"""
Generate meeting-ready figures for v4.3 hybrid TE classifier:
  1. Misclassification analysis figure (confusion matrix + top confusion pairs)
  2. Occlusion-based saliency maps for selected misclassified sequences

Outputs saved to current directory as PNG files.
"""

import json
import os
import sys
import glob as glob_module
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm

# ── Load model/data definitions from the training notebook ──────────────
NOTEBOOK_PATH = "vgp_features_tpase_multiclass_v4.2.ipynb"
nb = json.load(open(NOTEBOOK_PATH))
ns = {}
# Execute definition cells: imports(0), config(1), fasta/label utils(3),
# kmer featurizer(5), encoding+dataset(7), cnn blocks(9), cnn tower(10),
# gnn(12), attention fusion(14), hybrid classifier(16)
for cell_idx in [0, 1, 3, 5, 7, 9, 10, 12, 14, 16]:
    src = "".join(nb["cells"][cell_idx]["source"])
    exec(src, ns)

# Pull definitions into local namespace for clarity
HybridTEClassifierV4 = ns["HybridTEClassifierV4"]
KmerWindowFeaturizer = ns["KmerWindowFeaturizer"]
HybridDataset = ns["HybridDataset"]
collate_hybrid = ns["collate_hybrid"]
read_fasta = ns["read_fasta"]
load_multiclass_labels = ns["load_multiclass_labels"]
build_chain_edge_index = ns["build_chain_edge_index"]

DEVICE = ns["DEVICE"]
FIXED_LENGTH = ns["FIXED_LENGTH"]
KMER_K = ns["KMER_K"]
KMER_DIM = ns["KMER_DIM"]
KMER_WINDOW = ns["KMER_WINDOW"]
KMER_STRIDE = ns["KMER_STRIDE"]
CNN_WIDTH = ns["CNN_WIDTH"]
MOTIF_KERNELS = ns["MOTIF_KERNELS"]
CONTEXT_DILATIONS = ns["CONTEXT_DILATIONS"]
RC_FUSION_MODE = ns["RC_FUSION_MODE"]
GNN_HIDDEN = ns["GNN_HIDDEN"]
GNN_LAYERS = ns["GNN_LAYERS"]
FUSION_DIM = ns["FUSION_DIM"]
NUM_HEADS = ns["NUM_HEADS"]
DROPOUT = ns["DROPOUT"]

ENCODE = ns["ENCODE"]
REV_COMP = ns["REV_COMP"]

print("Definitions loaded successfully.")

# ══════════════════════════════════════════════════════════════════════════
# PART 1 : Misclassification Figure
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 1: Misclassification Figure")
print("=" * 60)

df = pd.read_csv("all_test_predictions_v4.3.csv")

# --- 1a. Class-level confusion matrix ---
class_names_sorted = sorted(df["true_class"].unique())
cls_true = df["true_class"].values
cls_pred = df["pred_class"].values
cm_cls = confusion_matrix(cls_true, cls_pred, labels=class_names_sorted)

# Row-normalize for recall interpretation
cm_cls_norm = cm_cls.astype(float) / cm_cls.sum(axis=1, keepdims=True)

# --- 1b. Top superfamily confusion pairs ---
misclass_df = df[df["sf_correct"] == False].copy()
pairs = misclass_df.groupby(["true_superfamily", "pred_superfamily"]).size()
pairs = pairs.sort_values(ascending=False).head(20).reset_index(name="count")

# --- 1c. Superfamily-level confusion matrix (top-15 by support) ---
sf_counts = Counter(df["true_superfamily"])
top15_sf = [sf for sf, _ in sf_counts.most_common(15)]
sf_mask = df["true_superfamily"].isin(top15_sf) | df["pred_superfamily"].isin(top15_sf)
df_sf = df[sf_mask]
cm_sf = confusion_matrix(
    df_sf["true_superfamily"], df_sf["pred_superfamily"], labels=top15_sf
)

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(22, 7), gridspec_kw={"width_ratios": [1, 1.8, 1.5]})
fig.suptitle("v4.3 Hybrid Model — Misclassification Analysis (Test Set)", fontsize=14, fontweight="bold", y=1.02)

# Panel A: class-level confusion
ax = axes[0]
im = ax.imshow(cm_cls_norm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
ax.set_title("A) Class Confusion (recall)", fontsize=12)
ax.set_xticks(range(len(class_names_sorted)))
ax.set_xticklabels(class_names_sorted, rotation=45, ha="right")
ax.set_yticks(range(len(class_names_sorted)))
ax.set_yticklabels(class_names_sorted)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
for i in range(len(class_names_sorted)):
    for j in range(len(class_names_sorted)):
        val = cm_cls_norm[i, j]
        raw = cm_cls[i, j]
        color = "white" if val > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}\n({raw})", ha="center", va="center", color=color, fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Panel B: Top-20 superfamily confusion pairs (horizontal bar)
ax = axes[1]
pair_labels = [f"{r.true_superfamily} → {r.pred_superfamily}" for _, r in pairs.iterrows()]
colors = []
for _, r in pairs.iterrows():
    tc = r.true_superfamily.split("/")[0]
    pc = r.pred_superfamily.split("/")[0]
    if tc != pc:
        colors.append("#E53935")  # cross-class = red
    else:
        colors.append("#1E88E5")  # within-class = blue
bars = ax.barh(range(len(pair_labels)), pairs["count"].values, color=colors, alpha=0.85)
ax.set_yticks(range(len(pair_labels)))
ax.set_yticklabels(pair_labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Count")
ax.set_title("B) Top SF Confusion Pairs", fontsize=12)
# Legend
from matplotlib.patches import Patch
ax.legend(
    handles=[Patch(color="#1E88E5", label="Within-class"), Patch(color="#E53935", label="Cross-class")],
    loc="lower right", fontsize=8
)
for bar, cnt in zip(bars, pairs["count"].values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, str(cnt),
            va="center", fontsize=7)

# Panel C: top-15 SF confusion matrix
ax = axes[2]
cm_sf_norm = cm_sf.astype(float)
row_sums = cm_sf_norm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_sf_norm = cm_sf_norm / row_sums
im = ax.imshow(cm_sf_norm, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
ax.set_title("C) SF Confusion (top-15, recall)", fontsize=12)
ax.set_xticks(range(len(top15_sf)))
ax.set_xticklabels(top15_sf, rotation=90, fontsize=7)
ax.set_yticks(range(len(top15_sf)))
ax.set_yticklabels(top15_sf, fontsize=7)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
# Annotate cells with count > 0
for i in range(len(top15_sf)):
    for j in range(len(top15_sf)):
        val = cm_sf[i, j]
        if val > 0:
            color = "white" if cm_sf_norm[i, j] > 0.4 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=6)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("misclassification_analysis_v4.3.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: misclassification_analysis_v4.3.png")

# ══════════════════════════════════════════════════════════════════════════
# PART 2 : Occlusion-based Saliency Analysis
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: Saliency Analysis")
print("=" * 60)

# --- Load model ---
results = torch.load("results_v4.3.pt", map_location="cpu", weights_only=False)
class_names = results["class_names"]
superfamily_names = results["superfamily_names"]
superfamily_to_id = results["superfamily_to_id"]

best_epoch = results.get("best_epoch", None)
if best_epoch:
    ckpt_path = f"hybrid_v4.3_epoch{best_epoch}.pt"
else:
    ckpt_path = sorted(glob_module.glob("hybrid_v4.3_epoch*.pt"))[-1]

featurizer = KmerWindowFeaturizer(k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW, stride=KMER_STRIDE)

model = HybridTEClassifierV4(
    num_classes=len(class_names),
    num_superfamilies=len(superfamily_names),
    cnn_width=CNN_WIDTH,
    motif_kernels=MOTIF_KERNELS,
    context_dilations=CONTEXT_DILATIONS,
    rc_mode=RC_FUSION_MODE,
    gnn_in_dim=KMER_DIM + 1,
    gnn_hidden=GNN_HIDDEN,
    gnn_layers=GNN_LAYERS,
    fusion_dim=FUSION_DIM,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
).to(DEVICE)

ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")


def prepare_single_sample(seq, sf_id, cls_id, featurizer, fixed_length):
    """Prepare a single sequence as a batch-of-one for the model."""
    kmer_feat, _ = featurizer.featurize_sequence(seq)

    ds = HybridDataset(
        headers=["sample"],
        sequences=[seq],
        binary_labels=np.array([sf_id]),
        class_labels=np.array([cls_id]),
        kmer_features=[kmer_feat],
        fixed_length=fixed_length,
    )
    batch = collate_hybrid([ds[0]], fixed_length=fixed_length)
    _, X_cnn, mask, Y_sf, Y_cls, x_gnn, edge_index, batch_vec = batch
    return (
        X_cnn.to(DEVICE), mask.to(DEVICE),
        x_gnn.to(DEVICE), edge_index.to(DEVICE), batch_vec.to(DEVICE),
    )


def get_sf_logit(model, X_cnn, mask, x_gnn, edge_index, batch_vec, class_idx):
    """Get the logit for a specific superfamily class."""
    with torch.no_grad():
        _, sf_logits, _ = model(X_cnn, mask, x_gnn, edge_index, batch_vec)
    return sf_logits[0, class_idx].item()


def occlusion_saliency(
    seq, sf_id, cls_id, model, featurizer, fixed_length,
    window_size=200, stride=50, target_sf_id=None,
):
    """
    Compute occlusion-based saliency for a DNA sequence.

    Slides a window of N's across the sequence and measures the drop in the
    target superfamily logit.  Positive saliency = region important for the
    prediction.
    """
    if target_sf_id is None:
        # Use the PREDICTED superfamily as the target
        X_cnn, mask, x_gnn, ei, bv = prepare_single_sample(seq, sf_id, cls_id, featurizer, fixed_length)
        with torch.no_grad():
            _, sf_logits, _ = model(X_cnn, mask, x_gnn, ei, bv)
        target_sf_id = sf_logits[0].argmax().item()

    # Baseline logit
    X_cnn, mask, x_gnn, ei, bv = prepare_single_sample(seq, sf_id, cls_id, featurizer, fixed_length)
    baseline = get_sf_logit(model, X_cnn, mask, x_gnn, ei, bv, target_sf_id)

    seq_len = len(seq)
    starts = list(range(0, max(1, seq_len - window_size + 1), stride))
    saliency = np.zeros(seq_len, dtype=np.float32)
    counts = np.zeros(seq_len, dtype=np.float32)

    for st in starts:
        en = min(st + window_size, seq_len)
        occluded = seq[:st] + "N" * (en - st) + seq[en:]
        X_cnn_o, mask_o, x_gnn_o, ei_o, bv_o = prepare_single_sample(
            occluded, sf_id, cls_id, featurizer, fixed_length
        )
        logit_occ = get_sf_logit(model, X_cnn_o, mask_o, x_gnn_o, ei_o, bv_o, target_sf_id)
        drop = baseline - logit_occ  # positive = important
        saliency[st:en] += drop
        counts[st:en] += 1.0

    counts[counts == 0] = 1.0
    saliency /= counts
    return saliency, target_sf_id, baseline


# --- Select examples: top misclassified superfamily confusion pairs ---
misclass_sf = pd.read_csv("misclassified_superfamily_trials_v4.3.csv")

# Pick diverse examples from the top confusion pairs
top_pairs_for_saliency = [
    ("DNA/hAT", "DNA"),
    ("LTR/Gypsy", "LTR/Pao"),
    ("DNA", "DNA/hAT"),
    ("LINE/L1", "LINE/L1-Tx1"),
]

# Load the full FASTA to retrieve sequences for selected headers
FASTA_PATH = "../../../data/vgp/all_vgp_tes.fa"
FEAT_PATH = "../../../data/vgp/20260120_features_sf"

print("Loading FASTA for saliency samples...")
all_headers, all_sequences = read_fasta(FASTA_PATH)
seq_dict = dict(zip(all_headers, all_sequences))
print(f"  Loaded {len(seq_dict)} sequences")

# Collect examples
examples = []
for true_sf, pred_sf in top_pairs_for_saliency:
    mask = (misclass_sf["true_superfamily"] == true_sf) & (misclass_sf["pred_superfamily"] == pred_sf)
    subset = misclass_sf[mask]
    if len(subset) == 0:
        print(f"  No examples for {true_sf} → {pred_sf}, skipping")
        continue
    # Take the first example with a sequence available
    for _, row in subset.iterrows():
        hdr = row["header"]
        if hdr in seq_dict:
            seq = seq_dict[hdr]
            class_to_id = {c: i for i, c in enumerate(class_names)}
            examples.append({
                "header": hdr,
                "sequence": seq,
                "true_sf": true_sf,
                "pred_sf": pred_sf,
                "true_sf_id": superfamily_to_id.get(true_sf, 0),
                "pred_sf_id": superfamily_to_id.get(pred_sf, 0),
                "cls_id": class_to_id[true_sf.split("/")[0]],
            })
            break
    else:
        print(f"  No FASTA match for {true_sf} → {pred_sf}")

print(f"Selected {len(examples)} examples for saliency analysis")

# --- Compute saliency for each example ---
saliency_results = []
for ex in examples:
    print(f"  Computing saliency: {ex['true_sf']} → {ex['pred_sf']} "
          f"(len={len(ex['sequence'])})")

    # Saliency for PREDICTED class (what drives the wrong prediction)
    sal_pred, target_pred, base_pred = occlusion_saliency(
        ex["sequence"], ex["true_sf_id"], ex["cls_id"],
        model, featurizer, FIXED_LENGTH,
        window_size=200, stride=50,
        target_sf_id=ex["pred_sf_id"],
    )

    # Saliency for TRUE class (what should have driven the correct one)
    sal_true, target_true, base_true = occlusion_saliency(
        ex["sequence"], ex["true_sf_id"], ex["cls_id"],
        model, featurizer, FIXED_LENGTH,
        window_size=200, stride=50,
        target_sf_id=ex["true_sf_id"],
    )

    saliency_results.append({
        **ex,
        "sal_pred": sal_pred,
        "sal_true": sal_true,
        "baseline_pred": base_pred,
        "baseline_true": base_true,
    })

# --- Plot saliency maps ---
n_examples = len(saliency_results)
if n_examples == 0:
    print("No saliency examples to plot.")
    sys.exit(0)

fig, axes = plt.subplots(n_examples, 1, figsize=(16, 3.5 * n_examples), squeeze=False)
fig.suptitle(
    "Occlusion Saliency Maps — Misclassified TE Sequences",
    fontsize=14, fontweight="bold", y=1.01,
)

for idx, res in enumerate(saliency_results):
    ax = axes[idx, 0]
    seq_len = len(res["sequence"])
    x_pos = np.arange(seq_len)

    # Smooth for readability
    kernel_size = 500
    kernel = np.ones(kernel_size) / kernel_size

    sal_true_smooth = np.convolve(res["sal_true"], kernel, mode="same")
    sal_pred_smooth = np.convolve(res["sal_pred"], kernel, mode="same")

    # Integrated drop = sum over positions (proportional to total causal mass).
    int_true = float(np.sum(res["sal_true"]))
    int_pred = float(np.sum(res["sal_pred"]))
    b_true = float(res["baseline_true"])
    b_pred = float(res["baseline_pred"])

    ax.fill_between(x_pos, sal_true_smooth, alpha=0.3, color="#2196F3",
                    label=f"True: {res['true_sf']}  (baseline logit={b_true:+.2f}, Σdrop={int_true:+.2f})")
    ax.plot(x_pos, sal_true_smooth, color="#2196F3", linewidth=0.8)
    ax.fill_between(x_pos, sal_pred_smooth, alpha=0.3, color="#E53935",
                    label=f"Pred: {res['pred_sf']}  (baseline logit={b_pred:+.2f}, Σdrop={int_pred:+.2f})")
    ax.plot(x_pos, sal_pred_smooth, color="#E53935", linewidth=0.8)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Saliency\n(logit drop)")
    margin = b_pred - b_true
    ax.set_title(
        f"{res['header'].split('#')[0]}  |  True: {res['true_sf']}  →  Pred: {res['pred_sf']}  "
        f"(len={seq_len:,}, decision margin pred−true={margin:+.2f})",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)

    if idx == n_examples - 1:
        ax.set_xlabel("Sequence Position (bp)")

plt.tight_layout()
plt.savefig("saliency_analysis_v4.3.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: saliency_analysis_v4.3.png")

print("\n✅ All figures generated.")
