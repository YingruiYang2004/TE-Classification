"""
Evaluate hybrid_v4.2_epoch28.pt on the species-disjoint held-out test split,
plus train/val (fold 0) for an overfitting diagnostic.

Mirrors the preprocessing of vgp_hybrid_v4.2.ipynb (same FASTA, label file,
same min_class_count, same GroupShuffleSplit/StratifiedGroupKFold seeds), but
runs only inference (no training).

Outputs:
  - eval_epoch28.log   (stdout)
  - eval_epoch28.json  (metrics dict)
  - eval_epoch28_confusion.png  (test-split SF confusion)

Run from the v4.2 cluster session directory:
    cd data_analysis/vgp_model_split_fix/v4.2/cluster\\ session
    ../../../../.venv/bin/python eval_epoch28.py
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "vgp_hybrid_v4.2.ipynb"
CHECKPOINT = HERE / "hybrid_v4.2_epoch28.pt"

# Match notebook config (Cell 1)
FASTA_PATH = HERE / "../../../../data/vgp/all_vgp_tes.fa"
LABEL_PATH = HERE / "../../../../data/vgp/20260120_features_sf"
TEST_SIZE = 0.2
N_FOLDS = 5
RANDOM_STATE = 42
MIN_CLASS_COUNT = 100
KEEP_CLASSES = ('DNA', 'LTR', 'LINE')

OUT_LOG = HERE / "eval_epoch28.log"
OUT_JSON = HERE / "eval_epoch28.json"
OUT_PNG = HERE / "eval_epoch28_confusion.png"


# ----------------------------------------------------------------------------
# Tee stdout to log file
# ----------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


_log_fh = open(OUT_LOG, "w")
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)


# ----------------------------------------------------------------------------
# Load helper / model definitions from the notebook
# ----------------------------------------------------------------------------
print(f"Loading definitions from {NOTEBOOK}")

with open(NOTEBOOK) as f:
    nb = json.load(f)

# Cells to import (everything BEFORE the training driver):
#   0  imports
#   3  read_fasta + load_multiclass_labels + compute_class_weights
#   5  KmerWindowFeaturizer + build_chain_edge_index
#   7  HybridDataset + collate_hybrid + ENCODE
#   9  CNN building blocks
#  10  CNNTower
#  12  GNN building blocks (incl. KmerGNNTower)
#  14  Cross-modal fusion
#  16  HybridTEClassifierV4
#  18  Loss functions (not strictly needed but harmless)
# Cell 1 defines DEVICE/FIXED_LENGTH which the other cells reference, so we
# also exec cell 1.
NEEDED = [0, 1, 3, 5, 7, 9, 10, 12, 14, 16, 18]

# Register a real module so @dataclass can resolve cls.__module__ via sys.modules.
import types as _types
_helper_mod = _types.ModuleType("v4_2_eval_helpers")
sys.modules["v4_2_eval_helpers"] = _helper_mod
ns: dict = _helper_mod.__dict__
ns["__name__"] = "v4_2_eval_helpers"
for ci in NEEDED:
    cell = nb["cells"][ci]
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    try:
        exec(compile(src, f"<v4.2 cell {ci}>", "exec"), ns)
    except Exception as e:
        print(f"FAILED to exec cell {ci}: {e}")
        raise

print("  notebook helpers loaded.")

# Bind pieces we need
read_fasta = ns["read_fasta"]
load_multiclass_labels = ns["load_multiclass_labels"]
KmerWindowFeaturizer = ns["KmerWindowFeaturizer"]
HybridDataset = ns["HybridDataset"]
collate_hybrid = ns["collate_hybrid"]
HybridTEClassifierV4 = ns["HybridTEClassifierV4"]
FIXED_LENGTH = ns["FIXED_LENGTH"]
resolve_device = ns["resolve_device"]


# ----------------------------------------------------------------------------
# Load checkpoint
# ----------------------------------------------------------------------------
DEVICE = resolve_device()
print(f"Device: {DEVICE}")
print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
arch = ckpt["arch"]
sf_names: list[str] = ckpt["superfamily_names"]
sf_to_id: dict[str, int] = ckpt["superfamily_to_id"]
print(f"  epoch={ckpt['epoch']}  score={ckpt['score']:.4f}")
print(f"  num_superfamilies={len(sf_names)}")
print(f"  arch keys={list(arch.keys())}")

# Build model from arch (matches notebook construction)
model = HybridTEClassifierV4(
    num_classes=arch["num_classes"],
    num_superfamilies=arch["num_superfamilies"],
    cnn_width=arch["cnn_width"],
    motif_kernels=tuple(arch["motif_kernels"]),
    context_dilations=tuple(arch["context_dilations"]),
    rc_mode=arch["rc_mode"],
    gnn_in_dim=arch["gnn_in_dim"],
    gnn_hidden=arch["gnn_hidden"],
    gnn_layers=arch["gnn_layers"],
    fusion_dim=arch["fusion_dim"],
    num_heads=arch["num_heads"],
    dropout=arch.get("dropout", 0.15),
).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("  model loaded.")

class_names = list(KEEP_CLASSES)
n_classes = len(class_names)
n_superfamilies = len(sf_names)


# ----------------------------------------------------------------------------
# Load full dataset
# ----------------------------------------------------------------------------
print("\n=== Loading FASTA + labels ===")
headers, sequences = read_fasta(str(FASTA_PATH))
label_dict, class_dict = load_multiclass_labels(str(LABEL_PATH), keep_classes=KEEP_CLASSES)

# Match headers, only keep superfamilies the model knows about
all_h, all_s, all_cls, all_sf = [], [], [], []
for h, s in zip(headers, sequences):
    if h not in label_dict:
        continue
    tag = label_dict[h]
    if tag not in sf_to_id:
        continue
    all_h.append(h)
    all_s.append(s)
    all_cls.append(class_dict[h])
    all_sf.append(sf_to_id[tag])

del headers, sequences
import gc; gc.collect()

all_cls = np.array(all_cls, dtype=np.int64)
all_sf = np.array(all_sf, dtype=np.int64)
print(f"Matched {len(all_h)} sequences against the {n_superfamilies} known superfamilies")
for cid, cn in enumerate(class_names):
    print(f"  {cn}: {(all_cls == cid).sum()}")


# ----------------------------------------------------------------------------
# Reproduce the v4.2 splits (species-grouped)
# ----------------------------------------------------------------------------
def _species_from_header(h: str) -> str:
    return h.lstrip(">").split("#", 1)[0].rsplit("-", 1)[1]

all_species = np.array([_species_from_header(h) for h in all_h])

print(f"\n=== Reproducing splits (test_size={TEST_SIZE}, n_folds={N_FOLDS}, "
      f"random_state={RANDOM_STATE}) ===")

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
idx_trainval, idx_test = next(gss.split(
    np.arange(len(all_h)), y=all_sf, groups=all_species
))
assert set(all_species[idx_trainval]).isdisjoint(set(all_species[idx_test])), \
    "Species leak trainval/test"

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_splits = list(sgkf.split(
    np.arange(len(idx_trainval)),
    y=all_sf[idx_trainval],
    groups=all_species[idx_trainval],
))
# Use fold 0 as a representative train/val split (matches v4.2 notebook
# overfitting analysis for epoch 47).
idx_train_in_tv, idx_val_in_tv = fold_splits[0]
idx_train = idx_trainval[idx_train_in_tv]
idx_val = idx_trainval[idx_val_in_tv]

print(f"  trainval={len(idx_trainval)}  test={len(idx_test)}")
print(f"  fold0 train={len(idx_train)}  val={len(idx_val)}")
print(f"  unique species: trainval={len(set(all_species[idx_trainval]))} "
      f"test={len(set(all_species[idx_test]))} "
      f"train(f0)={len(set(all_species[idx_train]))} "
      f"val(f0)={len(set(all_species[idx_val]))}")


# ----------------------------------------------------------------------------
# Pre-compute k-mer features for everything in idx_train ∪ idx_val ∪ idx_test
# ----------------------------------------------------------------------------
print("\n=== K-mer featurization ===")
featurizer = KmerWindowFeaturizer(
    k=arch.get("kmer_k", 7),
    dim=arch.get("kmer_dim", arch["gnn_in_dim"] - 1),
    window=arch.get("kmer_window", 512),
    stride=arch.get("kmer_stride", 256),
    add_pos=True, l2_normalize=True,
)
needed = sorted(set(idx_train.tolist()) | set(idx_val.tolist()) | set(idx_test.tolist()))
print(f"  featurizing {len(needed)} sequences (full pool would be {len(all_h)})")
kmer_cache: dict[int, np.ndarray] = {}
t0 = time.time()
for k, i in enumerate(needed, 1):
    X, _ = featurizer.featurize_sequence(all_s[i])
    kmer_cache[i] = X
    if k % max(1, len(needed) // 20) == 0 or k == len(needed):
        print(f"    [{k}/{len(needed)}] elapsed {time.time()-t0:.1f}s")


# ----------------------------------------------------------------------------
# Evaluate on each split
# ----------------------------------------------------------------------------
def make_loader(indices: np.ndarray) -> DataLoader:
    sub_h = [all_h[i] for i in indices]
    sub_s = [all_s[i] for i in indices]
    sub_cls = all_cls[indices]
    sub_sf = all_sf[indices]
    sub_kmer = [kmer_cache[i] for i in indices]
    ds = HybridDataset(
        headers=sub_h, sequences=sub_s,
        binary_labels=sub_sf, class_labels=sub_cls,
        kmer_features=sub_kmer, fixed_length=FIXED_LENGTH,
    )
    return DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=0,
        collate_fn=partial(collate_hybrid, fixed_length=FIXED_LENGTH),
    )


@torch.no_grad()
def evaluate(name: str, indices: np.ndarray) -> dict:
    print(f"\n=== Eval: {name} (n={len(indices)}) ===")
    loader = make_loader(indices)
    sf_true_all, sf_pred_all = [], []
    cls_true_all, cls_pred_all = [], []
    t0 = time.time()
    for bi, batch in enumerate(loader, 1):
        _, X_cnn, mask, Y_sf, Y_cls, x_gnn, edge_index, batch_vec = batch
        X_cnn = X_cnn.to(DEVICE)
        mask = mask.to(DEVICE)
        x_gnn = x_gnn.to(DEVICE)
        edge_index = edge_index.to(DEVICE)
        batch_vec = batch_vec.to(DEVICE)
        cls_logits, sf_logits, _ = model(X_cnn, mask, x_gnn, edge_index, batch_vec)
        sf_pred_all.extend(sf_logits.argmax(dim=1).cpu().tolist())
        sf_true_all.extend(Y_sf.tolist())
        cls_pred_all.extend(cls_logits.argmax(dim=1).cpu().tolist())
        cls_true_all.extend(Y_cls.tolist())
        if bi % 20 == 0:
            print(f"    batch {bi}/{len(loader)}  elapsed {time.time()-t0:.1f}s")

    sf_true = np.array(sf_true_all)
    sf_pred = np.array(sf_pred_all)
    cls_true = np.array(cls_true_all)
    cls_pred = np.array(cls_pred_all)

    out = {
        "n": int(len(indices)),
        "class_acc": float(accuracy_score(cls_true, cls_pred)),
        "class_balanced_acc": float(balanced_accuracy_score(cls_true, cls_pred)),
        "class_f1_macro": float(f1_score(cls_true, cls_pred, average="macro", zero_division=0)),
        "class_f1_weighted": float(f1_score(cls_true, cls_pred, average="weighted", zero_division=0)),
        "sf_acc": float(accuracy_score(sf_true, sf_pred)),
        "sf_balanced_acc": float(balanced_accuracy_score(sf_true, sf_pred)),
        "sf_f1_macro": float(f1_score(sf_true, sf_pred, average="macro", zero_division=0)),
        "sf_f1_weighted": float(f1_score(sf_true, sf_pred, average="weighted", zero_division=0)),
    }
    out["combined_score"] = 0.5 * out["class_f1_macro"] + 0.5 * out["sf_f1_macro"]

    print(f"  class : acc={out['class_acc']:.4f}  bal_acc={out['class_balanced_acc']:.4f}  "
          f"macroF1={out['class_f1_macro']:.4f}  weightedF1={out['class_f1_weighted']:.4f}")
    print(f"  sf    : acc={out['sf_acc']:.4f}  bal_acc={out['sf_balanced_acc']:.4f}  "
          f"macroF1={out['sf_f1_macro']:.4f}  weightedF1={out['sf_f1_weighted']:.4f}")
    print(f"  combined (0.5*cls_macroF1 + 0.5*sf_macroF1) = {out['combined_score']:.4f}")

    print("\n  Top-level class report:")
    print(classification_report(cls_true, cls_pred, target_names=class_names,
                                zero_division=0, digits=4))
    return out, sf_true, sf_pred, cls_true, cls_pred


results = {
    "checkpoint": str(CHECKPOINT),
    "checkpoint_epoch": int(ckpt["epoch"]),
    "checkpoint_internal_score": float(ckpt["score"]),
    "test_size": TEST_SIZE, "n_folds": N_FOLDS, "random_state": RANDOM_STATE,
    "splits": {
        "train_f0": int(len(idx_train)),
        "val_f0": int(len(idx_val)),
        "test": int(len(idx_test)),
    },
    "metrics": {},
}

for name, idxs in [("train_fold0", idx_train), ("val_fold0", idx_val), ("test", idx_test)]:
    metrics, sf_true, sf_pred, cls_true, cls_pred = evaluate(name, idxs)
    results["metrics"][name] = metrics
    if name == "test":
        # Save full SF report + confusion matrix only for test split
        results["test_sf_classification_report"] = classification_report(
            sf_true, sf_pred, target_names=sf_names, zero_division=0,
            output_dict=True, digits=4,
        )
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            cm = confusion_matrix(sf_true, sf_pred, labels=list(range(n_superfamilies)))
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sums > 0, cm / np.clip(row_sums, 1, None), 0.0)
            fig, ax = plt.subplots(figsize=(max(8, n_superfamilies * 0.5),
                                            max(7, n_superfamilies * 0.5)))
            im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")
            ax.set_xticks(range(n_superfamilies))
            ax.set_yticks(range(n_superfamilies))
            ax.set_xticklabels(sf_names, rotation=90, fontsize=7)
            ax.set_yticklabels(sf_names, fontsize=7)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"v4.2 epoch28 — held-out test SF confusion (row-normalised)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
            print(f"\nSaved confusion matrix → {OUT_PNG}")
        except Exception as e:
            print(f"(skip confusion plot: {e})")

# Overfitting summary
print("\n=== Overfitting summary (sf macro-F1) ===")
m = results["metrics"]
print(f"  train_fold0 = {m['train_fold0']['sf_f1_macro']:.4f}")
print(f"  val_fold0   = {m['val_fold0']['sf_f1_macro']:.4f}  "
      f"(checkpoint internal val sf_F1 ≈ {ckpt['history']['val_sf_f1'][ckpt['epoch']-1]:.4f})")
print(f"  test        = {m['test']['sf_f1_macro']:.4f}")
print(f"  Δ(train-test) = {m['train_fold0']['sf_f1_macro'] - m['test']['sf_f1_macro']:+.4f}")
print(f"  Δ(val-test)   = {m['val_fold0']['sf_f1_macro'] - m['test']['sf_f1_macro']:+.4f}")

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote metrics JSON → {OUT_JSON}")
print("DONE.")
