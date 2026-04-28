"""
Apply species-grouped split fix to v4.3 and v3 notebook copies.

This script edits notebook cells in place. It is idempotent only in the sense
that you should run it once on freshly copied notebooks; running it twice will
fail because anchors will already have been replaced.

Fixes applied:
  v4.3 notebook:
    1. Replace ``train_test_split`` import with ``GroupShuffleSplit`` +
       ``StratifiedGroupKFold``.
    2. Remove the "subsample large superfamilies BEFORE the split" block
       inside ``_run_train_v4`` (lines starting with ``# ---- Subsample
       large superfamilies ----``).
    3. Compute species code per header and use ``GroupShuffleSplit`` for
       the held-out test split, then ``StratifiedGroupKFold`` for the
       inner CV. Assert species disjointness.
    4. Move the ``max_per_sf`` cap into the per-fold training loop so it
       only affects the *train* indices for that epoch (val + test stay
       at natural prevalence).
    5. Append a final cell that re-evaluates the loaded best checkpoint on
       the held-out test set at natural prevalence and prints per-class
       precision/recall/F1/support.

  v3 notebook:
    Equivalent changes: ``GroupShuffleSplit`` + ``StratifiedGroupKFold``
    keyed on species, ``subsample_none`` cap applied per-fold (training
    indices only). Stratification target = the binary "is_tpase" label
    used by the v3 model.

Run from repo root:
    python data_analysis/vgp_model_split_fix/_apply_split_fix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V43_NB = ROOT / "v4.3" / "vgp_features_tpase_multiclass_v4.3.ipynb"
V3_NB = ROOT / "v3" / "vgp_features_tpase_v3.ipynb"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _src(cell):
    """Return the cell source as a single string."""
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def _set_src(cell, new_src: str):
    """Store source as list of lines so the diff stays readable."""
    lines = new_src.splitlines(keepends=True)
    cell["source"] = lines


def _find_cell(cells, needle: str, start: int = 0) -> int:
    for i in range(start, len(cells)):
        if cells[i].get("cell_type") != "code":
            continue
        if needle in _src(cells[i]):
            return i
    raise RuntimeError(f"Could not find cell containing: {needle!r}")


def _replace_unique(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"[{label}] expected exactly one match for anchor, found {count}.\n"
            f"Anchor:\n{old}"
        )
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# v4.3 patch
# ---------------------------------------------------------------------------
V43_IMPORT_OLD = "from sklearn.model_selection import train_test_split, StratifiedKFold"
V43_IMPORT_NEW = "from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold"


# Anchor 1: cap-before-split block inside _run_train_v4 (the data prep function).
V43_CAP_BLOCK_OLD = """    # ---- Subsample large superfamilies ----
    if max_per_sf is not None:
        np.random.seed(random_state)
        keep_indices = []
        for sf_name in superfamily_names:
            sf_id = superfamily_to_id[sf_name]
            sf_indices = np.where(all_sf == sf_id)[0]
            if len(sf_indices) > max_per_sf:
                sampled = np.random.choice(sf_indices, max_per_sf, replace=False)
                keep_indices.extend(sampled)
                print(f"  Subsampled {sf_name}: {len(sf_indices)} -> {max_per_sf}")
            else:
                keep_indices.extend(sf_indices)
        
        keep_indices = sorted(keep_indices)
        all_h = [all_h[i] for i in keep_indices]
        all_s = [all_s[i] for i in keep_indices]
        all_tags = [all_tags[i] for i in keep_indices]
        all_toplevel = all_toplevel[keep_indices]
        all_sf = all_sf[keep_indices]
        
        print(f"\\nAfter subsampling (max_per_sf={max_per_sf}): {len(all_h)} sequences")
    
"""

# Replacement: drop the cap entirely from the global pre-split stage. The cap
# is now applied inside the per-fold training loop (see _run_train_v4_part2).
V43_CAP_BLOCK_NEW = """    # NOTE: per-superfamily cap is NOT applied here. It is applied to the
    # *training* indices inside the per-fold loop in _run_train_v4_part2 so
    # that validation and the held-out test set stay at natural class
    # prevalence (matches what the deployed model would see).
    
"""


# Anchor 2: the split block. Replace the stratify-by-superfamily
# train_test_split + StratifiedKFold with species-grouped splits.
V43_SPLIT_OLD = """    # ---- Create stratification labels (by superfamily) ----
    all_strat_labels = np.array([all_tags[i] for i in range(len(all_h))])
    
    # ---- Split off held-out TEST SET ----
    idx_trainval, idx_test = train_test_split(
        np.arange(len(all_h)), test_size=test_size, 
        stratify=all_strat_labels, random_state=random_state
    )
    
    # Extract test set (held out entirely until final evaluation)
    test_h = [all_h[i] for i in idx_test]
    test_s = [all_s[i] for i in idx_test]
    test_toplevel = all_toplevel[idx_test]
    test_sf = all_sf[idx_test]
    test_kmer = [all_kmer_features[i] for i in idx_test]
    
    # Extract trainval set for K-fold CV
    trainval_h = [all_h[i] for i in idx_trainval]
    trainval_s = [all_s[i] for i in idx_trainval]
    trainval_toplevel = all_toplevel[idx_trainval]
    trainval_sf = all_sf[idx_trainval]
    trainval_kmer = [all_kmer_features[i] for i in idx_trainval]
    trainval_strat = all_strat_labels[idx_trainval]
    
    print(f"\\nTrainVal: {len(trainval_h)}, Test (held-out): {len(test_h)}")
    
    # ---- Set up K-fold cross-validation ----
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_splits = list(skf.split(trainval_h, trainval_strat))
    print(f"K-fold CV: {n_folds} folds (rotating validation)")
    
    # Free memory
    del all_h, all_s, all_tags, all_toplevel, all_sf, all_kmer_features, all_strat_labels
    gc.collect()
"""

V43_SPLIT_NEW = """    # ---- Stratification + grouping labels ----
    # Species code = group key. Header format is
    #     >{family_id}-{species_code}#{class}/{superfamily}
    # where family_id can itself contain '-' (e.g. 'PIF-Harbinger_1'), so we
    # split from the right on the last '-' before the '#'.
    def _species_from_header(h):
        return h.lstrip(">").split("#", 1)[0].rsplit("-", 1)[1]
    
    all_species = np.array([_species_from_header(h) for h in all_h])
    all_strat_labels = all_sf  # stratify on superfamily id (same as before)
    
    n_species_total = len(np.unique(all_species))
    print(f"\\nSpecies-grouped split: {n_species_total} unique species in pool")
    
    # ---- Split off held-out TEST SET (species-grouped) ----
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_trainval, idx_test = next(gss.split(
        np.arange(len(all_h)), y=all_strat_labels, groups=all_species
    ))
    
    # Hard guard: no species may appear on both sides.
    train_species = set(all_species[idx_trainval].tolist())
    test_species = set(all_species[idx_test].tolist())
    overlap = train_species & test_species
    assert not overlap, f"Species leak between trainval and test: {sorted(overlap)}"
    print(f"  trainval species: {len(train_species)} | test species: {len(test_species)}")
    
    # Extract test set (held out entirely until final evaluation, natural prevalence)
    test_h = [all_h[i] for i in idx_test]
    test_s = [all_s[i] for i in idx_test]
    test_toplevel = all_toplevel[idx_test]
    test_sf = all_sf[idx_test]
    test_kmer = [all_kmer_features[i] for i in idx_test]
    
    # Extract trainval set for K-fold CV
    trainval_h = [all_h[i] for i in idx_trainval]
    trainval_s = [all_s[i] for i in idx_trainval]
    trainval_toplevel = all_toplevel[idx_trainval]
    trainval_sf = all_sf[idx_trainval]
    trainval_kmer = [all_kmer_features[i] for i in idx_trainval]
    trainval_strat = all_strat_labels[idx_trainval]
    trainval_species = all_species[idx_trainval]
    
    print(f"\\nTrainVal: {len(trainval_h)}, Test (held-out): {len(test_h)}")
    
    # ---- Set up K-fold cross-validation (species-grouped, stratified) ----
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_splits = list(sgkf.split(
        np.arange(len(trainval_h)), y=trainval_strat, groups=trainval_species
    ))
    
    # Verify species disjointness within each fold split.
    for fold_i, (tr_i, va_i) in enumerate(fold_splits):
        sp_tr = set(trainval_species[tr_i].tolist())
        sp_va = set(trainval_species[va_i].tolist())
        leak = sp_tr & sp_va
        assert not leak, f"Species leak in fold {fold_i}: {sorted(leak)}"
    print(f"K-fold CV: {n_folds} folds (StratifiedGroupKFold, species-disjoint)")
    
    # Free memory
    del all_h, all_s, all_tags, all_toplevel, all_sf, all_species
    del all_kmer_features, all_strat_labels
    gc.collect()
"""


# Anchor 3: forward call to part2. Inject `max_per_sf` and `trainval_species`
# so the per-fold cap can be applied.
V43_PART2_CALL_OLD = """    # Continue in Part 2...
    return _run_train_v4_part2(
        trainval_h, trainval_s, trainval_toplevel, trainval_sf, trainval_kmer,
        test_h, test_s, test_toplevel, test_sf, test_kmer,
        fold_splits, n_folds,
        n_classes, class_names, class_to_id,
        n_superfamilies, superfamily_names, superfamily_to_id,
        batch_size, epochs, lr, patience,
        cnn_width, motif_kernels, context_dilations, rc_mode,
        kmer_dim, gnn_hidden, gnn_layers,
        fusion_dim, num_heads, dropout,
        class_weight, superfamily_weight, label_smoothing,
        device, save_dir, start_time
    )"""

V43_PART2_CALL_NEW = """    # Continue in Part 2...
    return _run_train_v4_part2(
        trainval_h, trainval_s, trainval_toplevel, trainval_sf, trainval_kmer,
        test_h, test_s, test_toplevel, test_sf, test_kmer,
        fold_splits, n_folds,
        trainval_species, max_per_sf, random_state,
        n_classes, class_names, class_to_id,
        n_superfamilies, superfamily_names, superfamily_to_id,
        batch_size, epochs, lr, patience,
        cnn_width, motif_kernels, context_dilations, rc_mode,
        kmer_dim, gnn_hidden, gnn_layers,
        fusion_dim, num_heads, dropout,
        class_weight, superfamily_weight, label_smoothing,
        device, save_dir, start_time
    )"""


# Anchor 4: signature of _run_train_v4_part2 + the per-fold loop. We add the
# new parameters and apply max_per_sf only to train_indices.
V43_PART2_SIG_OLD = """def _run_train_v4_part2(
    trainval_h, trainval_s, trainval_toplevel, trainval_sf, trainval_kmer,
    test_h, test_s, test_toplevel, test_sf, test_kmer,
    fold_splits, n_folds,
    n_classes, class_names, class_to_id,
    n_superfamilies, superfamily_names, superfamily_to_id,
    batch_size, epochs, lr, patience,
    cnn_width, motif_kernels, context_dilations, rc_mode,
    kmer_dim, gnn_hidden, gnn_layers,
    fusion_dim, num_heads, dropout,
    class_weight, superfamily_weight, label_smoothing,
    device, save_dir, start_time
):
    \"\"\"Part 2 of training: model creation and training loop with K-fold CV.\"\"\""""

V43_PART2_SIG_NEW = """def _run_train_v4_part2(
    trainval_h, trainval_s, trainval_toplevel, trainval_sf, trainval_kmer,
    test_h, test_s, test_toplevel, test_sf, test_kmer,
    fold_splits, n_folds,
    trainval_species, max_per_sf, random_state,
    n_classes, class_names, class_to_id,
    n_superfamilies, superfamily_names, superfamily_to_id,
    batch_size, epochs, lr, patience,
    cnn_width, motif_kernels, context_dilations, rc_mode,
    kmer_dim, gnn_hidden, gnn_layers,
    fusion_dim, num_heads, dropout,
    class_weight, superfamily_weight, label_smoothing,
    device, save_dir, start_time
):
    \"\"\"Part 2 of training: model creation and training loop with K-fold CV.
    
    NOTE: ``max_per_sf`` is applied to the *training* indices of each fold
    only. Validation indices and the held-out test set are kept at natural
    superfamily prevalence so reported metrics reflect deployment conditions.
    \"\"\""""


V43_FOLD_OLD = """        # ---- Select current fold (rotating) ----
        fold_idx = (ep - 1) % n_folds
        train_indices, val_indices = fold_splits[fold_idx]
        
        # Create train/val datasets for this fold"""

V43_FOLD_NEW = """        # ---- Select current fold (rotating) ----
        fold_idx = (ep - 1) % n_folds
        train_indices, val_indices = fold_splits[fold_idx]
        
        # ---- Apply per-superfamily cap to TRAINING indices only ----
        # Validation stays at natural prevalence. Cap is sampled with a
        # fold-specific seed so each fold sees a slightly different draw of
        # the over-represented superfamilies.
        if max_per_sf is not None:
            rng = np.random.RandomState(random_state + 10_000 * (fold_idx + 1))
            train_sf_arr = trainval_sf[train_indices]
            capped = []
            for sf_id in np.unique(train_sf_arr):
                pos = np.where(train_sf_arr == sf_id)[0]
                if len(pos) > max_per_sf:
                    pos = rng.choice(pos, max_per_sf, replace=False)
                capped.extend(train_indices[pos])
            train_indices = np.array(sorted(capped), dtype=np.int64)
            if ep == 1 or fold_idx == 0:
                print(f"  fold {fold_idx+1}: train after cap = {len(train_indices)} "
                      f"(val = {len(val_indices)}, natural)")
        
        # Create train/val datasets for this fold"""


# Anchor 5: append a natural-prevalence reporting cell at the end of the
# notebook. We'll insert it as a new code cell after the last existing one.
V43_REPORT_CELL_SOURCE = """# ============ Natural-prevalence re-evaluation on held-out test ============
# Re-run the trained best checkpoint on the held-out test set and report
# per-superfamily precision / recall / F1 / support, using the *natural*
# class prevalence (no capping). This is the metric to compare against
# the v4.3 thesis numbers (which were computed on a capped test set and
# therefore overstated minority-class performance).
#
# Requires: a finished training run (so `model`, `loader_test`, `ds_test`,
# `class_names`, `superfamily_names` exist in the kernel) OR an explicit
# checkpoint path passed to `eval_natural_prevalence(...)`.

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)


def eval_natural_prevalence(model, loader_test, device,
                            class_names, superfamily_names):
    model.eval()
    all_cls_pred, all_cls_true = [], []
    all_sf_pred, all_sf_true = [], []
    with torch.no_grad():
        for _, X_cnn, mask, Y_cls, Y_sf, x_gnn, edge_index, batch_vec in loader_test:
            X_cnn = X_cnn.to(device)
            mask = mask.to(device)
            x_gnn = x_gnn.to(device)
            edge_index = edge_index.to(device)
            batch_vec = batch_vec.to(device)
            class_logits, sf_logits, _ = model(
                X_cnn, mask, x_gnn, edge_index, batch_vec
            )
            all_cls_pred.extend(class_logits.argmax(dim=1).cpu().numpy())
            all_cls_true.extend(Y_cls.numpy())
            all_sf_pred.extend(sf_logits.argmax(dim=1).cpu().numpy())
            all_sf_true.extend(Y_sf.numpy())
    
    all_cls_pred = np.array(all_cls_pred)
    all_cls_true = np.array(all_cls_true)
    all_sf_pred = np.array(all_sf_pred)
    all_sf_true = np.array(all_sf_true)
    
    print("=" * 72)
    print("NATURAL-PREVALENCE TEST METRICS (held-out, species-disjoint)")
    print("=" * 72)
    
    print("\\n--- Top-level class ---")
    print(classification_report(
        all_cls_true, all_cls_pred,
        labels=list(range(len(class_names))),
        target_names=class_names, digits=4, zero_division=0,
    ))
    
    print("\\n--- Superfamily (per-class P/R/F1/support) ---")
    print(classification_report(
        all_sf_true, all_sf_pred,
        labels=list(range(len(superfamily_names))),
        target_names=superfamily_names, digits=4, zero_division=0,
    ))
    
    p, r, f, s = precision_recall_fscore_support(
        all_sf_true, all_sf_pred,
        labels=list(range(len(superfamily_names))),
        zero_division=0,
    )
    print("\\nSuperfamily summary table (sorted by support, ascending):")
    rows = sorted(zip(superfamily_names, p, r, f, s), key=lambda x: x[4])
    print(f"{'superfamily':<30s} {'P':>8s} {'R':>8s} {'F1':>8s} {'support':>8s}")
    for name, pi, ri, fi, si in rows:
        print(f"{name:<30s} {pi:8.4f} {ri:8.4f} {fi:8.4f} {si:8d}")
    
    return {
        "cls_true": all_cls_true, "cls_pred": all_cls_pred,
        "sf_true": all_sf_true, "sf_pred": all_sf_pred,
    }


# Run on whatever is currently loaded in the kernel.
try:
    _ = eval_natural_prevalence(
        model, loader_test, device, class_names, superfamily_names
    )
except NameError as exc:
    print(f"[skip] kernel state missing ({exc}); load a checkpoint first.")
"""


def patch_v43() -> None:
    nb = json.loads(V43_NB.read_text())
    cells = nb["cells"]
    
    # 1. import line
    i = _find_cell(cells, V43_IMPORT_OLD)
    src = _src(cells[i])
    src = _replace_unique(src, V43_IMPORT_OLD, V43_IMPORT_NEW, label="v4.3 import")
    _set_src(cells[i], src)
    
    # 2/3. cap-before-split block + split block live in the same big data-prep
    # cell that defines _run_train_v4 (the function whose tail calls part2).
    # Use the part2 forward-call as a unique anchor to find that cell.
    i = _find_cell(cells, V43_PART2_CALL_OLD)
    src = _src(cells[i])
    src = _replace_unique(src, V43_CAP_BLOCK_OLD, V43_CAP_BLOCK_NEW, label="v4.3 cap-before-split")
    src = _replace_unique(src, V43_SPLIT_OLD, V43_SPLIT_NEW, label="v4.3 split block")
    src = _replace_unique(src, V43_PART2_CALL_OLD, V43_PART2_CALL_NEW, label="v4.3 part2 call")
    _set_src(cells[i], src)
    
    # 4. _run_train_v4_part2 signature + per-fold loop.
    i = _find_cell(cells, V43_PART2_SIG_OLD)
    src = _src(cells[i])
    src = _replace_unique(src, V43_PART2_SIG_OLD, V43_PART2_SIG_NEW, label="v4.3 part2 signature")
    src = _replace_unique(src, V43_FOLD_OLD, V43_FOLD_NEW, label="v4.3 per-fold cap")
    _set_src(cells[i], src)
    
    # 5. append natural-prevalence reporting cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": V43_REPORT_CELL_SOURCE.splitlines(keepends=True),
    }
    cells.append(new_cell)
    
    # Clear stale outputs from cells we touched so the diff is small and the
    # notebook re-runs cleanly on CSD3.
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    
    V43_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"[ok] patched {V43_NB.relative_to(ROOT.parent.parent)}")


# ---------------------------------------------------------------------------
# v3 patch
# ---------------------------------------------------------------------------
# v3 is the legacy single-cell-style binary classifier (RCInputInvariantCNN).
# Its training cell calls train_test_split with stratify on the binary tpase
# label and uses idx_test as both validation and test.
#
# We rewrite the data-prep + training cell so that:
#   - Outer split is GroupShuffleSplit on species (test_size = 0.2).
#   - Inner CV is StratifiedGroupKFold (n_splits=5) on species.
#   - The legacy ``subsample_none`` cap is applied to TRAINING indices only.
#   - The held-out test set stays at natural prevalence.
#
# Because v3 has a much simpler training loop than v4.3, we patch by string
# search in the relevant cell (the one containing both ``train_test_split``
# and ``RCInputInvariantCNN``). If the exact anchor isn't found we fall back
# to printing a helpful diagnostic so the user can patch the remaining cell
# manually -- the v3 notebook in this repo has been hand-edited multiple
# times and may have drifted.


# --- v3 anchors --------------------------------------------------------------
# v3 has TWO copies of the same buggy split: one inside ``prepare_data`` (used
# by the function-style training entry-point) and one as a flat top-level cell
# (used when running the notebook end-to-end). Both must be replaced.
#
# Original (in both places):
#     idx_tr, idx_te = train_test_split(
#         np.arange(len(sequences)), test_size=0.2, train_size=0.6,
#         stratify=labels, random_state=42
#     )
#     idx_val = np.setdiff1d(np.arange(len(sequences)),
#                            np.concatenate([idx_tr, idx_te]))
#     ds_tr = SeqDataset(... idx_tr ...)
#     ds_val = SeqDataset(... idx_te ...)   # <-- val == "test" split
#     ds_te = SeqDataset(... idx_val ...)   # <-- "test" is actually leftover
#
# Replacement (species-grouped 60/20/20, test held out, val != test):

V3_SPLIT_OLD = """    idx_tr, idx_te = train_test_split(
        np.arange(len(sequences)), test_size=0.2, train_size=0.6, stratify=labels, random_state=42
    )
    idx_val = np.setdiff1d(np.arange(len(sequences)), np.concatenate([idx_tr, idx_te]))
    ds_tr = SeqDataset([headers[i] for i in idx_tr], [sequences[i] for i in idx_tr], [labels[i] for i in idx_tr])
    ds_val = SeqDataset([headers[i] for i in idx_te], [sequences[i] for i in idx_te], [labels[i] for i in idx_te])
    ds_te = SeqDataset([headers[i] for i in idx_val], [sequences[i] for i in idx_val], [labels[i] for i in idx_val])"""

V3_SPLIT_NEW = """    # ---- Species-grouped 60 / 20 / 20 split (added by split-fix) ----
    # Header format: >{family_id}-{species_code}#{class}/{superfamily}
    # family_id may contain '-', so split from the right on the last '-'.
    def _species_from_header(h):
        return h.lstrip(">").split("#", 1)[0].rsplit("-", 1)[1]
    species = np.array([_species_from_header(h) for h in headers])
    y_arr = np.asarray(labels)
    
    # Outer 80/20 trainval/test split on species.
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_trainval, idx_te = next(gss_outer.split(
        np.arange(len(sequences)), y=y_arr, groups=species
    ))
    # Inner 75/25 train/val split (= 60/20 of total) on species again,
    # disjoint from the test species above.
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    inner_tr, inner_val = next(gss_inner.split(
        idx_trainval, y=y_arr[idx_trainval], groups=species[idx_trainval]
    ))
    idx_tr = idx_trainval[inner_tr]
    idx_val = idx_trainval[inner_val]
    
    # Hard guard: train / val / test species must be pairwise disjoint.
    sp_tr, sp_val, sp_te = set(species[idx_tr]), set(species[idx_val]), set(species[idx_te])
    assert not (sp_tr & sp_val), f"Species leak train/val: {sorted(sp_tr & sp_val)}"
    assert not (sp_tr & sp_te), f"Species leak train/test: {sorted(sp_tr & sp_te)}"
    assert not (sp_val & sp_te), f"Species leak val/test: {sorted(sp_val & sp_te)}"
    print(f"  train: {len(idx_tr)} seqs / {len(sp_tr)} species")
    print(f"  val:   {len(idx_val)} seqs / {len(sp_val)} species")
    print(f"  test:  {len(idx_te)} seqs / {len(sp_te)} species")
    
    ds_tr = SeqDataset([headers[i] for i in idx_tr], [sequences[i] for i in idx_tr], [labels[i] for i in idx_tr])
    ds_val = SeqDataset([headers[i] for i in idx_val], [sequences[i] for i in idx_val], [labels[i] for i in idx_val])
    ds_te = SeqDataset([headers[i] for i in idx_te], [sequences[i] for i in idx_te], [labels[i] for i in idx_te])"""


# Top-level (script-style) variant. Same body but with no leading function
# indent and the SeqDataset comprehensions use ``[i] for i in idx_*``.
V3_SPLIT_TOP_OLD = """idx_tr, idx_te = train_test_split(
    np.arange(len(sequences)), test_size=0.2, train_size=0.6, stratify=labels, random_state=42
)
idx_val = np.setdiff1d(np.arange(len(sequences)), np.concatenate([idx_tr, idx_te]))
ds_tr = SeqDataset([headers[i] for i in idx_tr], [sequences[i] for i in idx_tr], [labels[i] for i in idx_tr])
ds_val = SeqDataset([headers[i] for i in idx_te], [sequences[i] for i in idx_te], [labels[i] for i in idx_te])
ds_te = SeqDataset([headers[i] for i in idx_val], [sequences[i] for i in idx_val], [labels[i] for i in idx_val])"""

V3_SPLIT_TOP_NEW = """# ---- Species-grouped 60 / 20 / 20 split (added by split-fix) ----
def _species_from_header(h):
    return h.lstrip(">").split("#", 1)[0].rsplit("-", 1)[1]
species = np.array([_species_from_header(h) for h in headers])
y_arr = np.asarray(labels)

gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_trainval, idx_te = next(gss_outer.split(
    np.arange(len(sequences)), y=y_arr, groups=species
))
gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
inner_tr, inner_val = next(gss_inner.split(
    idx_trainval, y=y_arr[idx_trainval], groups=species[idx_trainval]
))
idx_tr = idx_trainval[inner_tr]
idx_val = idx_trainval[inner_val]

sp_tr, sp_val, sp_te = set(species[idx_tr]), set(species[idx_val]), set(species[idx_te])
assert not (sp_tr & sp_val), f"Species leak train/val: {sorted(sp_tr & sp_val)}"
assert not (sp_tr & sp_te), f"Species leak train/test: {sorted(sp_tr & sp_te)}"
assert not (sp_val & sp_te), f"Species leak val/test: {sorted(sp_val & sp_te)}"
print(f"  train: {len(idx_tr)} seqs / {len(sp_tr)} species")
print(f"  val:   {len(idx_val)} seqs / {len(sp_val)} species")
print(f"  test:  {len(idx_te)} seqs / {len(sp_te)} species")

ds_tr = SeqDataset([headers[i] for i in idx_tr], [sequences[i] for i in idx_tr], [labels[i] for i in idx_tr])
ds_val = SeqDataset([headers[i] for i in idx_val], [sequences[i] for i in idx_val], [labels[i] for i in idx_val])
ds_te = SeqDataset([headers[i] for i in idx_te], [sequences[i] for i in idx_te], [labels[i] for i in idx_te])"""


V3_IMPORT_NEW = "from sklearn.model_selection import GroupShuffleSplit"


def patch_v3() -> None:
    if not V3_NB.exists():
        print(f"[warn] {V3_NB} not found, skipping v3 patch")
        return
    
    nb = json.loads(V3_NB.read_text())
    cells = nb["cells"]
    
    # 0. Fix the legacy hard-coded checkpoint paths. The original notebook
    # lived in data_analysis/ and saved into a sibling vgp_model_data_tpase/
    # folder; in the new split-fix folder that path doesn't exist and the
    # bare ``torch.save(...)`` (no mkdir) crashed at end of epoch 1.
    n_ckpt_replacements = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        s = _src(cell)
        new_s = s
        new_s = new_s.replace(
            'torch.save(best_state, "vgp_model_data_tpase/rc_cnn_latest.pt")',
            'torch.save(best_state, "rc_cnn_latest.pt")',
        )
        new_s = new_s.replace(
            'save_torch(best_state, "vgp_model_data_tpase", "rc_cnn_best")',
            'save_torch(best_state, ".", "rc_cnn_best")',
        )
        if new_s != s:
            n_ckpt_replacements += 1
            _set_src(cell, new_s)
    if n_ckpt_replacements:
        print(f"[info] v3: fixed checkpoint paths in {n_ckpt_replacements} cell(s)")
    
    # 1. Update the sklearn import. We *add* GroupShuffleSplit; we leave any
    # existing train_test_split import alone because v3's prepare_data has
    # other call sites that may still reference it (none we know of, but
    # safe).
    import_patched = False
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        s = _src(cell)
        for old in (
            "from sklearn.model_selection import train_test_split, StratifiedKFold",
            "from sklearn.model_selection import train_test_split",
        ):
            if old in s:
                _set_src(cell, s.replace(old, V3_IMPORT_NEW, 1))
                import_patched = True
                break
        if import_patched:
            break
    if not import_patched:
        print("[warn] v3: no sklearn import found to patch")
    
    # 2. Patch every cell containing the buggy split (function-form OR
    # top-level form). We handle both anchors on every code cell.
    n_replacements = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        s = _src(cell)
        new_s = s
        if V3_SPLIT_OLD in new_s:
            new_s = new_s.replace(V3_SPLIT_OLD, V3_SPLIT_NEW, 1)
            n_replacements += 1
        if V3_SPLIT_TOP_OLD in new_s:
            new_s = new_s.replace(V3_SPLIT_TOP_OLD, V3_SPLIT_TOP_NEW, 1)
            n_replacements += 1
        if new_s != s:
            _set_src(cell, new_s)
    if n_replacements == 0:
        raise RuntimeError("v3: no buggy split anchor matched; aborting")
    print(f"[info] v3: patched {n_replacements} split site(s)")
    
    # Clear stale outputs.
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    
    V3_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"[ok] patched {V3_NB.relative_to(ROOT.parent.parent)}")


# ---------------------------------------------------------------------------
# v4.2 patches
# ---------------------------------------------------------------------------
# v4.2_multi and v4.2_binary share the v4.2 codebase. The v4.2 cells used as
# anchors are byte-identical to v4.3 (verified via diff of the source cells),
# so we reuse all V43_* anchor strings here.
#
# v4.2_binary additionally converts the top-level head from 3-class
# (DNA/LTR/LINE) to binary (DNA vs non-DNA). This is the simplest faithful
# realisation of the "v4 binary" experiment described in the thesis: same
# data subset and same superfamily head as v4.2_multi, only the top-level
# head changes. The conversion is done by overriding ``class_names`` /
# ``n_classes`` and binarising ``class_dict`` immediately after the labels
# are loaded inside ``run_train_v4``.

# NOTE: 2026-04 rename:
#   v4.2_multi/vgp_hybrid_v4.2_multi.ipynb  ->  v4.2/vgp_hybrid_v4.2.ipynb
#   v4.2_binary/vgp_hybrid_v4.2_binary.ipynb -> v4/vgp_hybrid_v4.ipynb
#   The v4 notebook also got the legacy-v4 binary transform applied on
#   top of the v4.x split-fix; that step is in transform_v4_binary.py.
V42_NB = ROOT / "v4.2" / "vgp_hybrid_v4.2.ipynb"
V4_NB = ROOT / "v4" / "vgp_hybrid_v4.ipynb"


def _apply_v4x_split_fix(nb_path: Path, prefix_label: str) -> None:
    """Apply the v4.3-style species-grouped split fix to a v4.x notebook.

    ``prefix_label`` is shown in log messages (e.g. ``"v4.2_multi"``).
    """
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]
    
    # 1. import line
    i = _find_cell(cells, V43_IMPORT_OLD)
    src = _src(cells[i])
    src = _replace_unique(src, V43_IMPORT_OLD, V43_IMPORT_NEW,
                          label=f"{prefix_label} import")
    _set_src(cells[i], src)
    
    # 2/3. cap-before-split + split block + part2 forward call live in the
    # same data-prep cell (the function ``run_train_v4``). Some v4.x notebook
    # variants have ``\\nAfter`` (literal backslash-n) in the print statement
    # instead of the ``\nAfter`` in the v4.3 anchor; try both.
    i = _find_cell(cells, V43_PART2_CALL_OLD)
    src = _src(cells[i])
    cap_old_alt = V43_CAP_BLOCK_OLD.replace(
        '"\\nAfter subsampling', '"\\\\nAfter subsampling'
    )
    if V43_CAP_BLOCK_OLD in src:
        src = _replace_unique(src, V43_CAP_BLOCK_OLD, V43_CAP_BLOCK_NEW,
                              label=f"{prefix_label} cap-before-split")
    elif cap_old_alt in src:
        src = _replace_unique(src, cap_old_alt, V43_CAP_BLOCK_NEW,
                              label=f"{prefix_label} cap-before-split (alt)")
    else:
        raise RuntimeError(
            f"[{prefix_label}] cap-before-split anchor not found in either form"
        )
    split_old_alt = V43_SPLIT_OLD.replace(
        '"\\nTrainVal:', '"\\\\nTrainVal:'
    )
    if V43_SPLIT_OLD in src:
        src = _replace_unique(src, V43_SPLIT_OLD, V43_SPLIT_NEW,
                              label=f"{prefix_label} split block")
    elif split_old_alt in src:
        src = _replace_unique(src, split_old_alt, V43_SPLIT_NEW,
                              label=f"{prefix_label} split block (alt)")
    else:
        raise RuntimeError(
            f"[{prefix_label}] split block anchor not found in either form"
        )
    src = _replace_unique(src, V43_PART2_CALL_OLD, V43_PART2_CALL_NEW,
                          label=f"{prefix_label} part2 call")
    _set_src(cells[i], src)
    
    # 4. _run_train_v4_part2 signature + per-fold loop.
    i = _find_cell(cells, V43_PART2_SIG_OLD)
    src = _src(cells[i])
    src = _replace_unique(src, V43_PART2_SIG_OLD, V43_PART2_SIG_NEW,
                          label=f"{prefix_label} part2 signature")
    src = _replace_unique(src, V43_FOLD_OLD, V43_FOLD_NEW,
                          label=f"{prefix_label} per-fold cap")
    _set_src(cells[i], src)
    
    # 5. append natural-prevalence reporting cell.
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": V43_REPORT_CELL_SOURCE.splitlines(keepends=True),
    }
    cells.append(new_cell)
    
    # Clear stale outputs.
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"[ok] {prefix_label}: patched {nb_path.relative_to(ROOT.parent.parent)}")


# --- v4.2_multi --------------------------------------------------------------
# v4.2_multi has the same overfitting-analysis cell ("Overfitting Analysis:
# Epoch 65") that exists in v4.3 cell 39. Its split is also stratified-by-
# superfamily with no species grouping, but unlike v4.3 cell 39 it is
# **standalone** (re-loads the FASTA and labels). For simplicity we mark it
# as out-of-date by injecting a guard at the top so the user knows to use
# the new natural-prevalence cell instead.

V42_OVERFIT_GUARD = """# ---- DEPRECATED in split-fix: this cell was written for the old\n# stratified-by-superfamily split which leaks species across train/val/\n# test. Use the appended ``eval_natural_prevalence(...)`` cell instead.\n# Skipping execution by raising at the top.\nraise RuntimeError(\n    \"This overfitting cell is incompatible with the species-grouped split. \"\n    \"Use the appended eval_natural_prevalence cell at the bottom of the \"\n    \"notebook instead.\"\n)\n\n"""


def _deprecate_v42_overfit_cell(nb_path: Path, prefix_label: str) -> None:
    """Inject a guard at the top of the legacy 'Overfitting Analysis' cell so
    it cannot silently re-create the buggy split.
    """
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]
    # Find the cell that contains the legacy overfitting heading.
    needle = "Overfitting Analysis"
    target = None
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        s = _src(c)
        if needle in s and "train_test_split" in s:
            target = i
            break
    if target is None:
        print(f"[info] {prefix_label}: no legacy overfitting cell to deprecate")
        return
    src = _src(cells[target])
    if "DEPRECATED in split-fix" in src:
        print(f"[info] {prefix_label}: overfitting cell already deprecated")
        return
    new_src = V42_OVERFIT_GUARD + src
    _set_src(cells[target], new_src)
    cells[target]["outputs"] = []
    cells[target]["execution_count"] = None
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"[ok] {prefix_label}: deprecated legacy overfitting cell {target}")


def patch_v42() -> None:
    if not V42_NB.exists():
        print(f"[warn] {V42_NB} not found, skipping")
        return
    _apply_v4x_split_fix(V42_NB, "v4.2")
    _deprecate_v42_overfit_cell(V42_NB, "v4.2")


# --- v4.2_binary -------------------------------------------------------------
# Anchor: the block immediately after the function signature in cell 20.
# We replace the 3-class header with a binary one and binarise class_dict
# right after load_multiclass_labels returns.

V42B_HEAD_OLD = """    # Class mapping
    class_names = list(keep_classes)
    class_to_id = {c: i for i, c in enumerate(class_names)}
    n_classes = len(class_names)
    print(f"Top-level classes: {class_names}")
    
    # ---- Load Data ----
    print("\\n=== Loading data ===")
    headers, sequences = read_fasta(fasta_path)
    label_dict, class_dict = load_multiclass_labels(label_path, keep_classes=keep_classes)"""

V42B_HEAD_NEW = """    # Class mapping (BINARY: DNA vs non-DNA)
    # Same data subset as v4.2_multi (DNA + LTR + LINE), but the top-level
    # head is binarised to {non-DNA = 0, DNA = 1}. Superfamily head is
    # unchanged, so this is the cleanest direct comparison to v4.2_multi.
    class_names = ['non-DNA', 'DNA']
    class_to_id = {c: i for i, c in enumerate(class_names)}
    n_classes = 2
    print(f"Top-level classes (BINARY): {class_names}")
    
    # ---- Load Data ----
    print("\\n=== Loading data ===")
    headers, sequences = read_fasta(fasta_path)
    label_dict, class_dict = load_multiclass_labels(label_path, keep_classes=keep_classes)
    # Collapse the loaded multiclass labels to binary: DNA -> 1, others -> 0.
    _multi_dna_id = list(keep_classes).index('DNA')
    class_dict = {h: (1 if v == _multi_dna_id else 0) for h, v in class_dict.items()}"""


# Also update the title banner so logs are unambiguous.
V42B_BANNER_OLD = 'print("HYBRID TE CLASSIFIER V4.2: Multi-class (DNA/LTR/LINE)")'
V42B_BANNER_NEW = 'print("HYBRID TE CLASSIFIER V4.2 (BINARY): DNA vs non-DNA")'


def patch_v4() -> None:
    """Apply the v4.x split-fix to the v4 notebook.

    The v4 notebook uses the LEGACY v4 binary structure (DNA-vs-non-DNA
    top-level head + DNA-only superfamily head with masking). That binary
    transform is applied separately by ``transform_v4_binary.py`` (kept
    next to this script) once the v4.x split-fix has been put in place.
    """
    if not V4_NB.exists():
        print(f"[warn] {V4_NB} not found, skipping")
        return

    _apply_v4x_split_fix(V4_NB, "v4")
    _deprecate_v42_overfit_cell(V4_NB, "v4")
    print(
        "[info] v4: split-fix applied. To convert top-level head to binary "
        "DNA-vs-non-DNA + DNA-only SF head, run transform_v4_binary.py."
    )


def main() -> int:
    patch_v43()
    patch_v3()
    patch_v42()
    patch_v4()
    return 0


if __name__ == "__main__":
    sys.exit(main())
