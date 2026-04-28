"""Transform v4.2_binary -> v4 (legacy v4 structure).

Changes the renamed v4 notebook to:
  - Drop multi-class top-level filter (KEEP_CLASSES); load ALL data.
  - Top-level head: binary (DNA=1 vs everything-else=0).
  - Superfamily head: DNA superfamilies only; non-DNA samples get a
    sentinel label and the SF loss/metrics are masked to DNA samples.
  - Checkpoint prefix: hybrid_v4.2 -> hybrid_v4.
  - Deprecate the legacy 3-class k-mer separability cells.

Idempotent: re-running on an already-transformed notebook is a no-op
(every str_replace must match exactly once or zero times).
"""
import json
from pathlib import Path

NB_PATH = Path('/Users/alexyang/Documents/Part III System Biology/TE Classification/data_analysis/vgp_model_split_fix/v4/vgp_hybrid_v4.ipynb')

nb = json.loads(NB_PATH.read_text())
cells = nb['cells']

def src(i):
    c = cells[i]
    return ''.join(c['source']) if isinstance(c['source'], list) else c['source']

def set_src(i, s):
    cells[i]['source'] = s.splitlines(keepends=True)
    cells[i]['outputs'] = []
    cells[i]['execution_count'] = None

def replace_in(i, old, new, required=True):
    s = src(i)
    n = s.count(old)
    if n == 0:
        if required:
            raise SystemExit(f"cell {i}: anchor not found:\n  {old[:120]!r}")
        return False
    if n > 1:
        raise SystemExit(f"cell {i}: anchor matched {n}x; tighten anchor")
    set_src(i, s.replace(old, new, 1))
    return True

# ---- Cell 1: configuration ----
replace_in(1,
    "# Top-level class configuration (replaces binary DNA+/None)\n"
    "KEEP_CLASSES = ['DNA', 'LTR', 'LINE']  # Drop None, SINE, PLE, RC\n"
    "CLASS_NAMES = KEEP_CLASSES             # 3-class head: DNA vs LTR vs LINE\n"
    "N_CLASSES = len(CLASS_NAMES)",
    "# Top-level class configuration (BINARY: legacy v4 structure)\n"
    "# Top-level head distinguishes DNA transposons from everything else\n"
    "# (LTR, LINE, SINE, PLE, RC, None). The superfamily head is trained\n"
    "# only on DNA samples; non-DNA samples are masked out of the SF loss.\n"
    "CLASS_NAMES = ['non-DNA', 'DNA']\n"
    "N_CLASSES = len(CLASS_NAMES)\n"
    "DNA_CLASS_ID = 1                       # sentinel: positive class index",
)

# ---- Cell 3: add binary loader (alongside existing load_multiclass_labels) ----
addon = '''


def load_binary_dna_labels(label_path):
    """Load ALL labels and binarise to DNA-vs-non-DNA.

    Returns
    -------
    label_dict : dict[str, str]
        header -> raw superfamily tag (kept for SF mapping later).
    class_dict : dict[str, int]
        header -> 0 (non-DNA) or 1 (DNA), based on the top-level prefix
        before the first '/' in the tag.

    Unlike ``load_multiclass_labels``, this loader keeps every entry
    in the label file (LINE, LTR, SINE, PLE, RC, None, ...) so that
    the model sees full natural prevalence on the negative side.
    """
    from collections import Counter
    label_path = Path(label_path)
    label_dict, class_dict = {}, {}
    top_counts = Counter()
    with label_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            header = parts[0].lstrip(">")
            tag = parts[1]
            top_class = tag.split("/")[0]
            label_dict[header] = tag
            class_dict[header] = 1 if top_class == "DNA" else 0
            top_counts[top_class] += 1
    print(f"Loaded {len(label_dict)} sequences (binary DNA vs non-DNA)")
    print("\\nTop-level class distribution:")
    for cls, count in sorted(top_counts.items(), key=lambda x: -x[1]):
        marker = "DNA -> 1" if cls == "DNA" else "non-DNA -> 0"
        print(f"  {cls:<10s} {count:>8d}   ({marker})")
    n_dna = sum(1 for v in class_dict.values() if v == 1)
    print(f"\\nBinary totals: DNA={n_dna}, non-DNA={len(class_dict) - n_dna}")
    return label_dict, class_dict
'''
set_src(3, src(3).rstrip() + "\n" + addon)

# ---- Cell 20: rewrite run_train_v4 data prep ----
new_c20 = '''# ============ Training Function (Part 1: Data Preparation) ============

def run_train_v4(
    fasta_path,
    label_path,
    # Training params
    batch_size: int = 16,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 10,
    # CNN params
    cnn_width: int = 128,
    motif_kernels: Tuple[int, ...] = (7, 15, 21),
    context_dilations: Tuple[int, ...] = (1, 2, 4, 8),
    rc_mode: str = "late",
    # GNN params
    kmer_k: int = 7,
    kmer_dim: int = 2048,
    kmer_window: int = 512,
    kmer_stride: int = 256,
    gnn_hidden: int = 128,
    gnn_layers: int = 3,
    # Fusion params
    fusion_dim: int = 256,
    num_heads: int = 4,
    # Loss params
    dropout: float = 0.15,
    class_weight: float = 1.0,
    superfamily_weight: float = 1.0,
    label_smoothing: float = 0.1,
    # Data params
    min_class_count: int = 100,
    max_per_sf: int = 3000,
    test_size: float = 0.2,
    n_folds: int = 5,
    random_state: int = 42,
    # Other
    device = None,
    save_dir: str = ".",
):
    """Train Hybrid V4 model: BINARY DNA-vs-non-DNA + DNA superfamily.

    Structure (legacy v4):
      - Top-level head: 2 classes  -> non-DNA (0), DNA (1)
      - Superfamily head: DNA superfamilies only; loss/metrics masked
        to DNA samples in the training/validation/test loops.
      - Includes ALL non-DNA classes (LINE, LTR, SINE, PLE, RC, None)
        as negatives so the model sees the natural prevalence at
        deployment time.
    """
    import time
    start_time = time.time()

    device = resolve_device(device)
    print(f"Using device: {device}")
    print(f"\\n{'='*60}")
    print("HYBRID TE CLASSIFIER V4 (BINARY): DNA vs non-DNA + DNA superfamily")
    print(f"{'='*60}")

    class_names = ['non-DNA', 'DNA']
    class_to_id = {c: i for i, c in enumerate(class_names)}
    n_classes = 2
    print(f"Top-level classes (BINARY): {class_names}")

    # ---- Load Data (ALL labels, binarised) ----
    print("\\n=== Loading data ===")
    headers, sequences = read_fasta(fasta_path)
    label_dict, class_dict = load_binary_dna_labels(label_path)

    all_h, all_s, all_tags, all_toplevel = [], [], [], []
    for h, s in zip(headers, sequences):
        if h not in label_dict:
            continue
        all_h.append(h)
        all_s.append(s)
        all_tags.append(label_dict[h])
        all_toplevel.append(class_dict[h])

    del headers, sequences
    gc.collect()
    print(f"Matched {len(all_h)} sequences")

    # ---- Build superfamily mapping FROM DNA SAMPLES ONLY ----
    dna_tags = [t for t, top in zip(all_tags, all_toplevel) if top == 1]
    tag_counts = Counter(dna_tags)
    keep_superfamilies = {t for t, c in tag_counts.items() if c >= min_class_count}
    superfamily_names = sorted(keep_superfamilies)
    superfamily_to_id = {t: i for i, t in enumerate(superfamily_names)}
    n_superfamilies = len(superfamily_names)
    print(f"\\nDNA superfamilies kept ({n_superfamilies}, min_count={min_class_count}):")
    for sf in superfamily_names:
        print(f"  {sf}: {tag_counts[sf]}")

    # ---- Filter: keep ALL non-DNA + DNA-with-kept-SF ----
    SF_SENTINEL = 0   # arbitrary valid index; non-DNA SF labels are MASKED
    filtered_h, filtered_s, filtered_tags, filtered_toplevel, filtered_sf = [], [], [], [], []
    for h, s, tag, toplevel in zip(all_h, all_s, all_tags, all_toplevel):
        if toplevel == 0:
            # non-DNA -> always keep, sentinel SF id
            filtered_h.append(h)
            filtered_s.append(s)
            filtered_tags.append(tag)
            filtered_toplevel.append(0)
            filtered_sf.append(SF_SENTINEL)
        elif tag in superfamily_to_id:
            # DNA with kept SF
            filtered_h.append(h)
            filtered_s.append(s)
            filtered_tags.append(tag)
            filtered_toplevel.append(1)
            filtered_sf.append(superfamily_to_id[tag])

    all_h = filtered_h
    all_s = filtered_s
    all_tags = filtered_tags
    all_toplevel = np.array(filtered_toplevel, dtype=np.int64)
    all_sf = np.array(filtered_sf, dtype=np.int64)

    del filtered_h, filtered_s, filtered_tags, filtered_toplevel, filtered_sf
    gc.collect()

    n_dna = int((all_toplevel == 1).sum())
    n_neg = int((all_toplevel == 0).sum())
    print(f"\\nAfter filtering: {len(all_h)} sequences  (DNA={n_dna}, non-DNA={n_neg})")

    # NOTE: per-superfamily cap is applied to the *training* indices inside
    # the per-fold loop in _run_train_v4_part2 so that validation and the
    # held-out test set stay at natural class prevalence.

    # ---- Pre-compute K-mer Features ----
    print("\\n=== Pre-computing k-mer features ===")
    featurizer = KmerWindowFeaturizer(
        k=kmer_k, dim=kmer_dim, window=kmer_window, stride=kmer_stride,
        add_pos=True, l2_normalize=True
    )
    all_kmer_features = []
    for seq in tqdm(all_s, desc="Featurizing", leave=False):
        X, _ = featurizer.featurize_sequence(seq)
        all_kmer_features.append(X)
    print(f"K-mer features computed: {len(all_kmer_features)} sequences")

    # ---- Stratification + grouping labels ----
    # Header format: >{family_id}-{species_code}#{class}/{superfamily}
    # family_id can contain '-', so split from the right.
    def _species_from_header(h):
        return h.lstrip(">").split("#", 1)[0].rsplit("-", 1)[1]

    all_species = np.array([_species_from_header(h) for h in all_h])

    # Stratify on (top_level, sf_id) so both the binary head and SF
    # distribution are preserved across folds. For non-DNA samples
    # the sf_id is a sentinel so they collapse into one stratum.
    all_strat_labels = all_toplevel * (n_superfamilies + 1) + all_sf

    n_species_total = len(np.unique(all_species))
    print(f"\\nSpecies-grouped split: {n_species_total} unique species in pool")

    # ---- Held-out TEST split (species-grouped) ----
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_trainval, idx_test = next(gss.split(
        np.arange(len(all_h)), y=all_strat_labels, groups=all_species
    ))
    train_species = set(all_species[idx_trainval].tolist())
    test_species = set(all_species[idx_test].tolist())
    overlap = train_species & test_species
    assert not overlap, f"Species leak between trainval and test: {sorted(overlap)}"
    print(f"  trainval species: {len(train_species)} | test species: {len(test_species)}")

    test_h = [all_h[i] for i in idx_test]
    test_s = [all_s[i] for i in idx_test]
    test_toplevel = all_toplevel[idx_test]
    test_sf = all_sf[idx_test]
    test_kmer = [all_kmer_features[i] for i in idx_test]

    trainval_h = [all_h[i] for i in idx_trainval]
    trainval_s = [all_s[i] for i in idx_trainval]
    trainval_toplevel = all_toplevel[idx_trainval]
    trainval_sf = all_sf[idx_trainval]
    trainval_kmer = [all_kmer_features[i] for i in idx_trainval]
    trainval_strat = all_strat_labels[idx_trainval]
    trainval_species = all_species[idx_trainval]

    print(f"\\nTrainVal: {len(trainval_h)}, Test (held-out): {len(test_h)}")

    # ---- K-fold CV (species-grouped, stratified) ----
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_splits = list(sgkf.split(
        np.arange(len(trainval_h)), y=trainval_strat, groups=trainval_species
    ))
    for fold_i, (tr_i, va_i) in enumerate(fold_splits):
        sp_tr = set(trainval_species[tr_i].tolist())
        sp_va = set(trainval_species[va_i].tolist())
        leak = sp_tr & sp_va
        assert not leak, f"Species leak in fold {fold_i}: {sorted(leak)}"
    print(f"K-fold CV: {n_folds} folds (StratifiedGroupKFold, species-disjoint)")

    del all_h, all_s, all_tags, all_toplevel, all_sf, all_species
    del all_kmer_features, all_strat_labels
    gc.collect()

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
    )
'''
set_src(20, new_c20)

# ---- Cell 21: training loop -- mask SF loss/metrics to DNA samples ----
# Several surgical edits, all on the same cell.
c21 = src(21)

# (a) Checkpoint prefix
c21 = c21.replace(
    'ckpt_manager = TopKCheckpointManagerV4(save_dir, prefix="hybrid_v4.2", k=5)',
    'ckpt_manager = TopKCheckpointManagerV4(save_dir, prefix="hybrid_v4", k=5)',
)

# (b) Comment update for SF weights -- compute on DNA samples only
old_sf_weights = (
    '    # Superfamily loss - weighted by inverse sqrt frequency  \n'
    '    sf_weights = compute_class_weights(trainval_sf, n_superfamilies, mode="inv_sqrt")\n'
    '    sf_weights_t = torch.tensor(sf_weights, dtype=torch.float32, device=device)'
)
new_sf_weights = (
    '    # Superfamily loss - weighted by inverse sqrt frequency.\n'
    '    # IMPORTANT: compute weights on DNA samples ONLY, since the SF\n'
    '    # head is supervised only on DNA samples (non-DNA samples are\n'
    '    # masked out of the SF loss in the training loop below).\n'
    '    _dna_train_mask = (trainval_toplevel == 1)\n'
    '    sf_weights = compute_class_weights(\n'
    '        trainval_sf[_dna_train_mask], n_superfamilies, mode="inv_sqrt"\n'
    '    )\n'
    '    sf_weights_t = torch.tensor(sf_weights, dtype=torch.float32, device=device)'
)
assert old_sf_weights in c21, "sf_weights anchor not found"
c21 = c21.replace(old_sf_weights, new_sf_weights, 1)

# (c) Cap loop should cap based on DNA SF only (non-DNA all share sentinel id)
# Find the cap block and replace.
old_cap = (
    '        if max_per_sf is not None:\n'
    '            rng = np.random.RandomState(random_state + 10_000 * (fold_idx + 1))\n'
    '            train_sf_arr = trainval_sf[train_indices]\n'
    '            capped = []\n'
    '            for sf_id in np.unique(train_sf_arr):\n'
    '                pos = np.where(train_sf_arr == sf_id)[0]\n'
    '                if len(pos) > max_per_sf:\n'
    '                    pos = rng.choice(pos, max_per_sf, replace=False)\n'
    '                capped.extend(train_indices[pos])\n'
    '            train_indices = np.array(sorted(capped), dtype=np.int64)\n'
    '            if ep == 1 or fold_idx == 0:\n'
    '                print(f"  fold {fold_idx+1}: train after cap = {len(train_indices)} "\n'
    '                      f"(val = {len(val_indices)}, natural)")'
)
new_cap = (
    '        if max_per_sf is not None:\n'
    '            # Cap DNA superfamilies on the training side only.\n'
    '            # Non-DNA samples (toplevel==0) all share the sentinel\n'
    '            # SF id and are NOT capped here; they are capped via\n'
    '            # max_negatives below if requested.\n'
    '            rng = np.random.RandomState(random_state + 10_000 * (fold_idx + 1))\n'
    '            train_top_arr = trainval_toplevel[train_indices]\n'
    '            train_sf_arr = trainval_sf[train_indices]\n'
    '            capped = []\n'
    '            # Non-DNA: keep all\n'
    '            non_dna_pos = np.where(train_top_arr == 0)[0]\n'
    '            capped.extend(train_indices[non_dna_pos])\n'
    '            # DNA: cap per superfamily\n'
    '            dna_pos = np.where(train_top_arr == 1)[0]\n'
    '            for sf_id in np.unique(train_sf_arr[dna_pos]):\n'
    '                pos = dna_pos[train_sf_arr[dna_pos] == sf_id]\n'
    '                if len(pos) > max_per_sf:\n'
    '                    pos = rng.choice(pos, max_per_sf, replace=False)\n'
    '                capped.extend(train_indices[pos])\n'
    '            train_indices = np.array(sorted(capped), dtype=np.int64)\n'
    '            if ep == 1 or fold_idx == 0:\n'
    '                _n_dna_after = int((trainval_toplevel[train_indices] == 1).sum())\n'
    '                _n_neg_after = int((trainval_toplevel[train_indices] == 0).sum())\n'
    '                print(f"  fold {fold_idx+1}: train after cap = {len(train_indices)} "\n'
    '                      f"(DNA={_n_dna_after}, non-DNA={_n_neg_after}; "\n'
    '                      f"val = {len(val_indices)}, natural)")'
)
assert old_cap in c21, "cap-block anchor not found"
c21 = c21.replace(old_cap, new_cap, 1)

# (d) Loss in training loop: mask SF loss to DNA samples
old_loss = (
    '            # Class loss (all samples - DNA vs LTR vs LINE)\n'
    '            cls_loss = class_loss_fn(class_logits, Y_cls)\n'
    '            \n'
    '            # Superfamily loss (all samples have valid superfamily labels now)\n'
    '            sf_loss = superfamily_loss_fn(sf_logits, Y_sf)\n'
    '            \n'
    '            loss = class_weight * cls_loss + superfamily_weight * sf_loss'
)
new_loss = (
    '            # Class loss (binary: DNA vs non-DNA, all samples)\n'
    '            cls_loss = class_loss_fn(class_logits, Y_cls)\n'
    '\n'
    '            # Superfamily loss (DNA samples ONLY; non-DNA are masked out)\n'
    '            dna_mask = (Y_cls == 1)\n'
    '            if dna_mask.any():\n'
    '                sf_loss = superfamily_loss_fn(sf_logits[dna_mask], Y_sf[dna_mask])\n'
    '            else:\n'
    '                sf_loss = torch.zeros((), device=class_logits.device)\n'
    '\n'
    '            loss = class_weight * cls_loss + superfamily_weight * sf_loss'
)
assert old_loss in c21, "training-loss anchor not found"
c21 = c21.replace(old_loss, new_loss, 1)

# (e) Validation: keep cls metrics over all samples, but SF metrics over DNA only
old_val = (
    '        all_cls_pred = np.array(all_cls_pred)\n'
    '        all_cls_true = np.array(all_cls_true)\n'
    '        all_sf_pred = np.array(all_sf_pred)\n'
    '        all_sf_true = np.array(all_sf_true)\n'
    '        \n'
    '        cls_acc = accuracy_score(all_cls_true, all_cls_pred)\n'
    '        cls_f1 = f1_score(all_cls_true, all_cls_pred, average="macro", zero_division=0)\n'
    '        sf_acc = accuracy_score(all_sf_true, all_sf_pred)\n'
    '        sf_f1 = f1_score(all_sf_true, all_sf_pred, average="macro", zero_division=0)'
)
new_val = (
    '        all_cls_pred = np.array(all_cls_pred)\n'
    '        all_cls_true = np.array(all_cls_true)\n'
    '        all_sf_pred = np.array(all_sf_pred)\n'
    '        all_sf_true = np.array(all_sf_true)\n'
    '\n'
    '        cls_acc = accuracy_score(all_cls_true, all_cls_pred)\n'
    '        cls_f1 = f1_score(all_cls_true, all_cls_pred, average="macro", zero_division=0)\n'
    '\n'
    '        # SF metrics: only over DNA samples (non-DNA are sentinel-labelled)\n'
    '        _val_dna_mask = (all_cls_true == 1)\n'
    '        if _val_dna_mask.any():\n'
    '            sf_acc = accuracy_score(all_sf_true[_val_dna_mask], all_sf_pred[_val_dna_mask])\n'
    '            sf_f1 = f1_score(\n'
    '                all_sf_true[_val_dna_mask], all_sf_pred[_val_dna_mask],\n'
    '                average="macro", zero_division=0,\n'
    '            )\n'
    '        else:\n'
    '            sf_acc = 0.0\n'
    '            sf_f1 = 0.0'
)
assert old_val in c21, "val-metrics anchor not found"
c21 = c21.replace(old_val, new_val, 1)

set_src(21, c21)

# ---- Cell 22: Final eval part 3 -- mask SF on test set ----
c22 = src(22)
old_22 = (
    '    print("\\n--- Class Classification (DNA vs LTR vs LINE) ---")\n'
    '    print(classification_report(all_cls_true, all_cls_pred, target_names=class_names, zero_division=0))\n'
    '    \n'
    '    print("\\n--- Superfamily Classification ---")\n'
    '    print(classification_report(all_sf_true, all_sf_pred, target_names=superfamily_names, zero_division=0))'
)
new_22 = (
    '    print("\\n--- Top-level (binary: DNA vs non-DNA) ---")\n'
    '    print(classification_report(all_cls_true, all_cls_pred, target_names=class_names, zero_division=0))\n'
    '\n'
    '    print("\\n--- DNA Superfamily Classification (DNA samples only) ---")\n'
    '    _test_dna = (all_cls_true == 1)\n'
    '    if _test_dna.any():\n'
    '        print(classification_report(\n'
    '            all_sf_true[_test_dna], all_sf_pred[_test_dna],\n'
    '            target_names=superfamily_names, zero_division=0,\n'
    '        ))\n'
    '    else:\n'
    '        print("  (no DNA samples in held-out test set)")'
)
assert old_22 in c22, "cell22 anchor not found"
set_src(22, c22.replace(old_22, new_22, 1))

# ---- Cell 24: drop keep_classes/min_class_count rename, update title ----
c24 = src(24)
c24 = c24.replace(
    "# ============ Train the Hybrid V4.2 Model ============\n"
    "# V4.2 Changes:\n"
    "# - Multi-class top-level head (DNA vs LTR vs LINE) instead of binary\n"
    "# - Drops None, SINE, PLE, RC classes\n"
    "# - Subsamples large superfamilies to MAX_PER_SF",
    "# ============ Train the Hybrid V4 Model (BINARY) ============\n"
    "# V4 structure (legacy):\n"
    "# - Binary top-level head: DNA vs non-DNA\n"
    "# - DNA superfamily head (masked to DNA samples in loss/metrics)\n"
    "# - Includes ALL classes (LINE, LTR, SINE, PLE, RC, None) on the\n"
    "#   non-DNA side at natural prevalence\n"
    "# - Subsamples large DNA superfamilies to MAX_PER_SF on the\n"
    "#   training side only",
)
c24 = c24.replace(
    "    # Data params\n"
    "    keep_classes=KEEP_CLASSES,\n"
    "    min_class_count=MIN_CLASS_COUNT,\n"
    "    max_per_sf=MAX_PER_SF,\n",
    "    # Data params\n"
    "    min_class_count=MIN_CLASS_COUNT,\n"
    "    max_per_sf=MAX_PER_SF,\n",
)
set_src(24, c24)

# ---- Cell 26: plot title text ----
c26 = src(26)
c26 = c26.replace(
    'fig.suptitle("Hybrid V4 Training History (CNN + K-mer GNN)"',
    'fig.suptitle("Hybrid V4 Training History (Binary DNA-vs-non-DNA + DNA SF)"',
)
c26 = c26.replace(
    'ax2.set_title("Class Classification (DNA vs LTR vs LINE)")',
    'ax2.set_title("Top-level: DNA vs non-DNA")',
)
c26 = c26.replace(
    'ax3.set_title("Superfamily Classification")',
    'ax3.set_title("DNA Superfamily (DNA samples only)")',
)
set_src(26, c26)

# ---- Cell 27: confusion matrix titles + filename unchanged but update label ----
c27 = src(27)
c27 = c27.replace(
    'ax1.set_title("Class Classification Confusion Matrix")',
    'ax1.set_title("Binary (DNA vs non-DNA) Confusion Matrix")',
)
set_src(27, c27)

# ---- Cell 29: final summary text ----
c29 = src(29)
c29 = c29.replace(
    "HYBRID V4.2 MODEL SUMMARY (Multi-class)",
    "HYBRID V4 MODEL SUMMARY (Binary DNA-vs-non-DNA)",
)
c29 = c29.replace(
    'print(f"\\nClass Classification (DNA vs LTR vs LINE):")',
    'print(f"\\nTop-level (DNA vs non-DNA):")',
)
c29 = c29.replace(
    'print(f"\\nSuperfamily Classification:")',
    'print(f"\\nDNA Superfamily Classification (DNA samples only):")',
)
set_src(29, c29)

# ---- Cells 31, 32, 33, 35: deprecate (multi-class k-mer separability) ----
DEPRECATION_HEADER = (
    "# ---- DEPRECATED in v4 (binary): this cell was a 3-class k-mer\n"
    "# separability check for the v4.2 multi-class model. The v4 binary\n"
    "# model has a different label structure, so this analysis no longer\n"
    "# applies. Skipping execution by raising at the top.\n"
    "raise RuntimeError(\n"
    "    \"Multi-class k-mer separability is not applicable to the v4 \"\n"
    "    \"binary model. See the v4.2 notebook for the multi-class \"\n"
    "    \"version of this analysis.\"\n"
    ")\n\n"
)
for i in (31, 32, 33, 35):
    s = src(i)
    if not s.startswith("# ---- DEPRECATED in v4"):
        set_src(i, DEPRECATION_HEADER + s)

# ---- Cell 34: gate-weight analysis -- update glob ----
c34 = src(34)
c34 = c34.replace(
    'ckpt_files = sorted(glob.glob("hybrid_v4.2_epoch*.pt"))',
    'ckpt_files = sorted(glob.glob("hybrid_v4_epoch*.pt"))',
)
c34 = c34.replace(
    'print("No checkpoint files found (hybrid_v4.2_epoch*.pt)")',
    'print("No checkpoint files found (hybrid_v4_epoch*.pt)")',
)
set_src(34, c34)

# ---- Cell 37: already deprecated; leave as-is (its body references v4.2) ----

# ---- Cell 38: natural-prevalence eval -- mask SF metrics ----
c38 = src(38)
old_38 = (
    '    print("\\n--- Top-level class ---")\n'
    '    print(classification_report(\n'
    '        all_cls_true, all_cls_pred,\n'
    '        labels=list(range(len(class_names))),\n'
    '        target_names=class_names, digits=4, zero_division=0,\n'
    '    ))\n'
    '    \n'
    '    print("\\n--- Superfamily (per-class P/R/F1/support) ---")\n'
    '    print(classification_report(\n'
    '        all_sf_true, all_sf_pred,\n'
    '        labels=list(range(len(superfamily_names))),\n'
    '        target_names=superfamily_names, digits=4, zero_division=0,\n'
    '    ))\n'
    '    \n'
    '    p, r, f, s = precision_recall_fscore_support(\n'
    '        all_sf_true, all_sf_pred,\n'
    '        labels=list(range(len(superfamily_names))),\n'
    '        zero_division=0,\n'
    '    )'
)
new_38 = (
    '    print("\\n--- Top-level class ---")\n'
    '    print(classification_report(\n'
    '        all_cls_true, all_cls_pred,\n'
    '        labels=list(range(len(class_names))),\n'
    '        target_names=class_names, digits=4, zero_division=0,\n'
    '    ))\n'
    '\n'
    '    # Superfamily metrics: DNA samples only (non-DNA carry sentinel SF)\n'
    '    _dna = (all_cls_true == 1)\n'
    '    if not _dna.any():\n'
    '        print("\\n[skip] no DNA samples in test set; cannot report SF metrics.")\n'
    '        return {\n'
    '            "cls_true": all_cls_true, "cls_pred": all_cls_pred,\n'
    '            "sf_true": all_sf_true, "sf_pred": all_sf_pred,\n'
    '        }\n'
    '    sf_t = all_sf_true[_dna]\n'
    '    sf_p = all_sf_pred[_dna]\n'
    '\n'
    '    print("\\n--- DNA Superfamily (per-class P/R/F1/support, DNA samples only) ---")\n'
    '    print(classification_report(\n'
    '        sf_t, sf_p,\n'
    '        labels=list(range(len(superfamily_names))),\n'
    '        target_names=superfamily_names, digits=4, zero_division=0,\n'
    '    ))\n'
    '\n'
    '    p, r, f, s = precision_recall_fscore_support(\n'
    '        sf_t, sf_p,\n'
    '        labels=list(range(len(superfamily_names))),\n'
    '        zero_division=0,\n'
    '    )'
)
assert old_38 in c38, "cell38 SF anchor not found"
set_src(38, c38.replace(old_38, new_38, 1))

# ---- Save ----
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"[ok] transformed {NB_PATH.name}")
