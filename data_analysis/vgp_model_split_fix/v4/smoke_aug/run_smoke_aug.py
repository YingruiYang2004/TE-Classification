"""Smoke test: sequence augmentation for cross-species v4 hybrid SF head.

This is the smallest runnable demonstration of **strategy 1** from
``../cross_species_strategies.md``. It trains the existing v4 hybrid
architecture on a 5000-sample species-disjoint subset for 3 epochs,
with a training-time augmentation pipeline:

    - random reverse complement (p=0.5)
    - random crop within FIXED_LENGTH (always when seq longer than canvas)
    - low-rate point mutations (p=0.005 per base, A/C/G/T uniform)
    - random window dropout in the GNN graph (5% per window, run-level p=0.5)
    - SF-head label smoothing (0.1)
    - SF-head dropout raised to 0.3 (vs DROPOUT default)

A control run is available with ``--no-aug``: identical schedule but
augmentation disabled. The two runs are written side by side so you
can compare val SF macro F1 + per-species hAT recall + mean gate
weights.

Usage::

    cd data_analysis/vgp_model_split_fix/v4/smoke_aug
    python run_smoke_aug.py            # aug arm
    python run_smoke_aug.py --no-aug   # control arm
    # or both in one go:
    python run_smoke_aug.py --both

Each arm writes ``smoke_<aug|noaug>.log`` next to this script and the
final-epoch metrics to ``smoke_summary.json``. With the smoke params
on Apple Silicon MPS, each arm takes ~30-60 minutes.

To scale up to a full run on CSD3, change the SMOKE TEST PARAMS block
near the top: ``EPOCHS=20-30, SUBSET_SIZE=None, BATCH_SIZE=64`` and
submit via the existing slurm template
``models/slurm_submit_hybrid_v5.sh`` (adjust the script path).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------ #
# SMOKE TEST PARAMS  -- change these for full-scale run.              #
# ------------------------------------------------------------------ #
EPOCHS         = 3
SUBSET_SIZE    = 5000        # set to None for full data
BATCH_SIZE     = 16
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
SF_DROPOUT     = 0.3         # raised from baseline 0.1
LABEL_SMOOTHING_SF = 0.1
P_RC           = 0.5
P_MUT          = 0.005
P_WINDOW_DROP_RATE = 0.05
P_WINDOW_DROP_RUN  = 0.5
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
VAL_SIZE       = 0.25        # of trainval -> 60/20/20 overall
# ------------------------------------------------------------------ #

THIS = Path(__file__).resolve().parent
V4_DIR = THIS.parent
NB = V4_DIR / "cluster session" / "vgp_hybrid_v4_gpu.ipynb"
WORKSPACE = V4_DIR.parents[2]
FASTA_PATH = str(WORKSPACE / "data" / "vgp" / "all_vgp_tes.fa")
LABEL_PATH = str(WORKSPACE / "data" / "vgp" / "20260120_features_sf")

# Definition cells from the GPU notebook (identical to eval_epoch8.py)
DEFINITION_CELLS = [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 23, 39]

COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def _exec_notebook_defs() -> dict:
    nb = json.load(open(NB))
    import types
    mod = types.ModuleType("__nbexec__")
    mod.__file__ = str(NB)
    sys.modules["__nbexec__"] = mod
    ns: dict = mod.__dict__
    ns["__name__"] = "__nbexec__"
    ns["__file__"] = str(NB)
    for i in DEFINITION_CELLS:
        src = "".join(nb["cells"][i]["source"])
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    return ns


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _species_from_header(h: str) -> str:
    return h.lstrip(">").split("#", 1)[0].rsplit("-", 1)[1]


# ------------------------------------------------------------------ #
# Augmentation primitives                                             #
# ------------------------------------------------------------------ #

def reverse_complement(seq: str) -> str:
    return "".join(COMP.get(c, "N") for c in reversed(seq.upper()))


def augment_sequence(seq: str, rng: np.random.Generator,
                     *, p_rc: float, p_mut: float, crop_to: int) -> str:
    """Apply RC + random crop + point mutations. Order matters.

    The crop is `crop_to`-aligned: if `len(seq) > crop_to` we sample a
    contiguous window of size `crop_to`; otherwise we leave it (the
    collate function will canvas-pad).
    """
    s = seq.upper()
    if rng.random() < p_rc:
        s = reverse_complement(s)
    if len(s) > crop_to:
        start = int(rng.integers(0, len(s) - crop_to + 1))
        s = s[start:start + crop_to]
    if p_mut > 0 and len(s) > 0:
        b = bytearray(s, "ascii")
        n_mut = int(rng.binomial(len(b), p_mut))
        if n_mut > 0:
            positions = rng.choice(len(b), n_mut, replace=False)
            choices = b"ACGT"
            for pos in positions:
                cur = b[pos]
                # pick a non-cur base; rng.choice over a 4-char list, reject
                while True:
                    nb_ = int(choices[int(rng.integers(0, 4))])
                    if nb_ != cur:
                        b[pos] = nb_
                        break
        s = b.decode("ascii")
    return s


# ------------------------------------------------------------------ #
# Augmenting dataset wrapper                                          #
# ------------------------------------------------------------------ #

class AugHybridDataset(Dataset):
    """Like HybridDataset but re-featurizes on the fly when augmenting.

    `kmer_features_pre` holds the precomputed (no-aug) k-mer arrays
    used in eval / when `augment=False`. When `augment=True` we draw a
    fresh augmented sequence and call `featurizer.featurize_sequence`
    each call, so this is only cheap on small subsets (which is the
    point of the smoke test).
    """

    def __init__(self, base_dataset, *, augment: bool, featurizer,
                 fixed_length: int, encode_table, p_rc: float,
                 p_mut: float, seed: int = 0):
        self.base = base_dataset
        self.augment = augment
        self.featurizer = featurizer
        self.fixed_length = fixed_length
        self.ENCODE = encode_table
        self.p_rc = p_rc
        self.p_mut = p_mut
        # Per-sample seed to keep workers reproducible-ish even at num_workers=0
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Tap base dataset for header/labels
        # base.__getitem__ returns: header, seq_idx, top, sf, start, end, length, kmer
        header = self.base.headers[idx]
        seq = self.base.sequences[idx]
        top = int(self.base.binary_labels[idx])
        sf = int(self.base.class_labels[idx])

        if self.augment:
            seq2 = augment_sequence(
                seq, self._rng,
                p_rc=self.p_rc, p_mut=self.p_mut, crop_to=self.fixed_length,
            )
            kmer_feat, _ = self.featurizer.featurize_sequence(seq2)
            kmer_feat = kmer_feat.astype(np.float32, copy=False)
        else:
            seq2 = seq[: self.fixed_length] if len(seq) > self.fixed_length else seq
            kmer_feat = np.asarray(self.base.kmer_features[idx], dtype=np.float32)

        seq_bytes = seq2.encode("ascii", "ignore")
        seq_idx = self.ENCODE[np.frombuffer(seq_bytes, dtype=np.uint8)]
        seq_len = len(seq2)

        max_start = max(0, self.fixed_length - seq_len)
        start_pos = int(self._rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end_pos = start_pos + seq_len

        return (header, seq_idx, top, sf, start_pos, end_pos, seq_len, kmer_feat)


# ------------------------------------------------------------------ #
# Window-dropout wrapper around collate_hybrid                         #
# ------------------------------------------------------------------ #

def make_collate_with_window_dropout(base_collate, *, augment: bool,
                                     p_drop: float, p_run: float,
                                     fixed_length: int):
    """Wrap collate_hybrid: drop a random subset of GNN windows per batch.

    Easier than rebuilding edge_index ourselves: we drop windows from
    each *sample's* k-mer feature tensor before the base collate
    builds the chain graph. That way the chain graph automatically has
    one fewer node where we drop, and the base collate's edge-index
    construction stays intact.
    """
    rng = np.random.default_rng(0xC0FFEE)

    def _collate(batch):
        if augment and rng.random() < p_run:
            new_batch = []
            for (header, seq_idx, top, sf, start, end, length, kmer) in batch:
                n = kmer.shape[0]
                if n > 1 and p_drop > 0:
                    keep = rng.random(n) > p_drop
                    if not keep.any():
                        keep[int(rng.integers(0, n))] = True
                    kmer = kmer[keep]
                new_batch.append((header, seq_idx, top, sf, start, end, length, kmer))
            batch = new_batch
        return base_collate(batch, fixed_length=fixed_length)

    return _collate


# ------------------------------------------------------------------ #
# Data prep (mirrors cell 21, plus train/val/test split)              #
# ------------------------------------------------------------------ #

def prepare_smoke_data(ns: dict, device: torch.device, *, subset_size, random_state):
    read_fasta = ns["read_fasta"]
    load_binary_dna_labels = ns["load_binary_dna_labels"]
    KmerWindowFeaturizer = ns["KmerWindowFeaturizer"]
    KmerWindowFeaturizerGPU = ns["KmerWindowFeaturizerGPU"]
    HybridDataset = ns["HybridDataset"]
    MIN_CLASS_COUNT = ns["MIN_CLASS_COUNT"]
    MAX_PER_SF = ns["MAX_PER_SF"]
    KMER_K = ns["KMER_K"]
    KMER_DIM = ns["KMER_DIM"]
    KMER_WINDOW = ns["KMER_WINDOW"]
    KMER_STRIDE = ns["KMER_STRIDE"]

    print("=== Loading data ===", flush=True)
    headers, sequences = read_fasta(FASTA_PATH)
    label_dict, class_dict = load_binary_dna_labels(LABEL_PATH)

    all_h, all_s, all_tags, all_top = [], [], [], []
    for h, s in zip(headers, sequences):
        if h not in label_dict:
            continue
        all_h.append(h); all_s.append(s)
        all_tags.append(label_dict[h]); all_top.append(class_dict[h])
    del headers, sequences; gc.collect()

    # SF mapping
    dna_tags = [t for t, top in zip(all_tags, all_top) if top == 1]
    keep_sf = {t for t, c in Counter(dna_tags).items() if c >= MIN_CLASS_COUNT}
    superfamily_names = sorted(keep_sf)
    superfamily_to_id = {t: i for i, t in enumerate(superfamily_names)}

    SF_SENTINEL = 0
    fh, fs, ft, ftop, fsf = [], [], [], [], []
    for h, s, tag, top in zip(all_h, all_s, all_tags, all_top):
        if top == 0:
            fh.append(h); fs.append(s); ft.append(tag); ftop.append(0); fsf.append(SF_SENTINEL)
        elif tag in superfamily_to_id:
            fh.append(h); fs.append(s); ft.append(tag); ftop.append(1); fsf.append(superfamily_to_id[tag])
    all_h, all_s, all_tags = fh, fs, ft
    all_top = np.array(ftop, dtype=np.int64)
    all_sf = np.array(fsf, dtype=np.int64)

    # Global non-DNA cap (matches eval_epoch8.py)
    if MAX_PER_SF is not None:
        rng_cap = np.random.RandomState(random_state)
        nd_idx = np.where(all_top == 0)[0]
        d_idx = np.where(all_top == 1)[0]
        by_tag = defaultdict(list)
        for i in nd_idx:
            by_tag[all_tags[i]].append(int(i))
        keep_nd = []
        for tag, idxs in by_tag.items():
            if len(idxs) > MAX_PER_SF:
                idxs = rng_cap.choice(idxs, MAX_PER_SF, replace=False).tolist()
            keep_nd.extend(idxs)
        keep = sorted(d_idx.tolist() + keep_nd)
        all_h = [all_h[i] for i in keep]
        all_s = [all_s[i] for i in keep]
        all_tags = [all_tags[i] for i in keep]
        all_top = all_top[keep]
        all_sf = all_sf[keep]
    print(f"After cap: {len(all_h)} (DNA={int((all_top==1).sum())}, non-DNA={int((all_top==0).sum())})")

    # ---- Subset (random, *before* species split, so species coverage shrinks too)
    if subset_size is not None and subset_size < len(all_h):
        rng_sub = np.random.default_rng(random_state)
        sel = np.sort(rng_sub.choice(len(all_h), subset_size, replace=False))
        all_h = [all_h[i] for i in sel]
        all_s = [all_s[i] for i in sel]
        all_top = all_top[sel]
        all_sf = all_sf[sel]
        print(f"Subset: {len(all_h)} sequences", flush=True)

    species = np.array([_species_from_header(h) for h in all_h])
    print(f"Species in subset: {len(set(species))}")

    # Featurize ALL (used for val/test always; for train when augment=False)
    print("=== K-mer featurization ===", flush=True)
    if device.type in ("cuda", "mps"):
        feat = KmerWindowFeaturizerGPU(
            k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW, stride=KMER_STRIDE,
            add_pos=True, l2_normalize=True, device=device,
        )
    else:
        feat = KmerWindowFeaturizer(
            k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW, stride=KMER_STRIDE,
            add_pos=True, l2_normalize=True,
        )
    all_kmer = []
    n_total = len(all_s)
    pe = max(500, n_total // 10)
    t0 = time.time()
    for i, seq in enumerate(all_s, 1):
        X, _ = feat.featurize_sequence(seq)
        all_kmer.append(X.astype(np.float16, copy=False))
        if i % pe == 0 or i == n_total:
            print(f"  [{i}/{n_total}] {(time.time()-t0):.1f}s", flush=True)

    # CPU featurizer for on-the-fly augmentation in training (avoids GPU contention).
    feat_train = KmerWindowFeaturizer(
        k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW, stride=KMER_STRIDE,
        add_pos=True, l2_normalize=True,
    )

    # Split: 80/20 species-disjoint outer (test), 75/25 inner (train/val).
    strat = all_top * (len(superfamily_names) + 1) + all_sf
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=random_state)
    idx_trainval, idx_te = next(gss_outer.split(np.arange(len(all_h)), y=strat, groups=species))
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=random_state)
    inner_tr, inner_val = next(gss_inner.split(
        idx_trainval, y=strat[idx_trainval], groups=species[idx_trainval]))
    idx_tr = idx_trainval[inner_tr]
    idx_val = idx_trainval[inner_val]

    sp_tr, sp_val, sp_te = set(species[idx_tr]), set(species[idx_val]), set(species[idx_te])
    assert not (sp_tr & sp_val), "leak train/val"
    assert not (sp_tr & sp_te), "leak train/test"
    assert not (sp_val & sp_te), "leak val/test"
    print(f"Splits: train={len(idx_tr)} ({len(sp_tr)}sp) | val={len(idx_val)} ({len(sp_val)}sp) | test={len(idx_te)} ({len(sp_te)}sp)")

    def _make(idxs):
        return HybridDataset(
            [all_h[i] for i in idxs],
            [all_s[i] for i in idxs],
            all_top[idxs],
            all_sf[idxs],
            [all_kmer[i] for i in idxs],
        )
    ds_tr = _make(idx_tr)
    ds_val = _make(idx_val)
    ds_te = _make(idx_te)

    return (ds_tr, ds_val, ds_te,
            superfamily_names, feat_train,
            species[idx_te])


# ------------------------------------------------------------------ #
# Training loop                                                       #
# ------------------------------------------------------------------ #

def build_model(ns: dict, n_classes: int, n_sf: int, device, sf_dropout: float):
    HybridTEClassifierV4 = ns["HybridTEClassifierV4"]
    CNN_WIDTH = ns["CNN_WIDTH"]
    MOTIF_KERNELS = ns["MOTIF_KERNELS"]
    CONTEXT_DILATIONS = ns["CONTEXT_DILATIONS"]
    RC_FUSION_MODE = ns["RC_FUSION_MODE"]
    KMER_DIM = ns["KMER_DIM"]
    GNN_HIDDEN = ns["GNN_HIDDEN"]
    GNN_LAYERS = ns["GNN_LAYERS"]
    FUSION_DIM = ns["FUSION_DIM"]
    NUM_HEADS = ns["NUM_HEADS"]
    DROPOUT = ns["DROPOUT"]

    model = HybridTEClassifierV4(
        num_classes=n_classes,
        num_superfamilies=n_sf,
        cnn_width=CNN_WIDTH,
        motif_kernels=tuple(MOTIF_KERNELS),
        context_dilations=tuple(CONTEXT_DILATIONS),
        rc_mode=RC_FUSION_MODE,
        gnn_in_dim=KMER_DIM + 1,  # +1 for position channel
        gnn_hidden=GNN_HIDDEN,
        gnn_layers=GNN_LAYERS,
        fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(device)

    # Bump dropout on the SF head (named `superfamily_head` in v4)
    for m in model.superfamily_head.modules():
        if isinstance(m, nn.Dropout):
            m.p = sf_dropout
    return model


def evaluate(model, loader, device, n_sf, superfamily_names):
    model.eval()
    cls_p, cls_t, sf_p, sf_t, gw = [], [], [], [], []
    with torch.no_grad():
        for _, X_cnn, mask, Y_cls, Y_sf, x_gnn, edge_index, batch_vec in loader:
            X_cnn = X_cnn.to(device); mask = mask.to(device)
            x_gnn = x_gnn.to(device); edge_index = edge_index.to(device)
            batch_vec = batch_vec.to(device)
            class_logits, sf_logits, g = model(X_cnn, mask, x_gnn, edge_index, batch_vec)
            cls_p.append(class_logits.argmax(1).cpu().numpy())
            cls_t.append(Y_cls.numpy())
            sf_p.append(sf_logits.argmax(1).cpu().numpy())
            sf_t.append(Y_sf.numpy())
            gw.append(g.cpu().numpy())
    cls_p = np.concatenate(cls_p); cls_t = np.concatenate(cls_t)
    sf_p = np.concatenate(sf_p); sf_t = np.concatenate(sf_t)
    gw = np.concatenate(gw, axis=0)

    cls_macro = f1_score(cls_t, cls_p, average="macro", zero_division=0)
    dna = (cls_t == 1)
    if dna.any():
        p, r, f, s = precision_recall_fscore_support(
            sf_t[dna], sf_p[dna],
            labels=list(range(n_sf)), zero_division=0,
        )
        sf_macro = float(np.mean(f))
        per_sf = {n: {"P": float(pi), "R": float(ri), "F1": float(fi), "n": int(si)}
                  for n, pi, ri, fi, si in zip(superfamily_names, p, r, f, s)}
    else:
        sf_macro = float("nan")
        per_sf = {}

    gate_mean = gw.mean(0).tolist()  # [w_cnn, w_gnn]
    return {
        "cls_macro_f1": float(cls_macro),
        "sf_macro_f1": sf_macro,
        "per_sf": per_sf,
        "gate_mean": gate_mean,
        "n": int(len(cls_p)),
    }


def run_one_arm(ns, *, augment: bool, log_path: Path, ckpt_dir: Path):
    device = _resolve_device()
    print(f"[arm augment={augment}] device={device}", flush=True)

    (ds_tr, ds_val, ds_te,
     superfamily_names, feat_train,
     test_species) = prepare_smoke_data(ns, device, subset_size=SUBSET_SIZE,
                                        random_state=RANDOM_STATE)

    ENCODE = ns["ENCODE"]
    FIXED_LENGTH = ns["FIXED_LENGTH"]
    collate_hybrid = ns["collate_hybrid"]

    train_view = AugHybridDataset(
        ds_tr, augment=augment, featurizer=feat_train,
        fixed_length=FIXED_LENGTH, encode_table=ENCODE,
        p_rc=P_RC, p_mut=P_MUT, seed=RANDOM_STATE,
    )
    collate_train = make_collate_with_window_dropout(
        collate_hybrid, augment=augment,
        p_drop=P_WINDOW_DROP_RATE, p_run=P_WINDOW_DROP_RUN,
        fixed_length=FIXED_LENGTH,
    )

    loader_tr = DataLoader(train_view, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_train, num_workers=0)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_hybrid, num_workers=0)
    loader_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_hybrid, num_workers=0)

    n_classes = 2  # non-DNA, DNA
    n_sf = len(superfamily_names)

    # Class weights (from train labels)
    tr_top = ds_tr.binary_labels
    cls_w = torch.tensor(
        [len(tr_top) / (2 * max((tr_top == c).sum(), 1)) for c in range(n_classes)],
        dtype=torch.float32, device=device,
    )
    tr_sf_dna = ds_tr.class_labels[ds_tr.binary_labels == 1]
    sf_w = torch.tensor(
        [len(tr_sf_dna) / (n_sf * max((tr_sf_dna == c).sum(), 1)) for c in range(n_sf)],
        dtype=torch.float32, device=device,
    )

    model = build_model(ns, n_classes, n_sf, device, SF_DROPOUT)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    cls_loss_fn = nn.CrossEntropyLoss(weight=cls_w)
    sf_loss_fn = nn.CrossEntropyLoss(weight=sf_w, label_smoothing=LABEL_SMOOTHING_SF)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    log_lines = []

    def _log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    _log(f"\n=== Training (augment={augment}, epochs={EPOCHS}) ===")
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running = 0.0; n_seen = 0
        for _, X_cnn, mask, Y_cls, Y_sf, x_gnn, edge_index, batch_vec in loader_tr:
            X_cnn = X_cnn.to(device); mask = mask.to(device)
            x_gnn = x_gnn.to(device); edge_index = edge_index.to(device)
            batch_vec = batch_vec.to(device)
            Y_cls = Y_cls.to(device); Y_sf = Y_sf.to(device)

            class_logits, sf_logits, _ = model(X_cnn, mask, x_gnn, edge_index, batch_vec)
            loss_cls = cls_loss_fn(class_logits, Y_cls)
            dna_mask = (Y_cls == 1)
            if dna_mask.any():
                loss_sf = sf_loss_fn(sf_logits[dna_mask], Y_sf[dna_mask])
            else:
                loss_sf = torch.zeros((), device=device)
            loss = loss_cls + loss_sf

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = X_cnn.size(0)
            running += float(loss.item()) * bs; n_seen += bs

        train_loss = running / max(n_seen, 1)
        val_metrics = evaluate(model, loader_val, device, n_sf, superfamily_names)
        elapsed = time.time() - t0

        msg = (f"  epoch {ep}/{EPOCHS} | loss {train_loss:.4f} "
               f"| val cls F1 {val_metrics['cls_macro_f1']:.4f} "
               f"| val SF macro F1 {val_metrics['sf_macro_f1']:.4f} "
               f"| gate (cnn,gnn)=({val_metrics['gate_mean'][0]:.2f},"
               f"{val_metrics['gate_mean'][1]:.2f}) "
               f"| {elapsed:.0f}s")
        _log(msg)
        if "hAT" in val_metrics["per_sf"]:
            h = val_metrics["per_sf"]["hAT"]
            _log(f"      hAT  P={h['P']:.3f} R={h['R']:.3f} F1={h['F1']:.3f} n={h['n']}")
        if "TcMar-Tc1" in val_metrics["per_sf"]:
            t = val_metrics["per_sf"]["TcMar-Tc1"]
            _log(f"      Tc1  P={t['P']:.3f} R={t['R']:.3f} F1={t['F1']:.3f} n={t['n']}")
        history.append({"epoch": ep, "train_loss": train_loss, **val_metrics})

    # Test eval
    test_metrics = evaluate(model, loader_te, device, n_sf, superfamily_names)
    _log("\n=== Held-out test (species-disjoint) ===")
    _log(f"  cls macro F1: {test_metrics['cls_macro_f1']:.4f}")
    _log(f"  SF  macro F1: {test_metrics['sf_macro_f1']:.4f}")
    _log(f"  gate mean   : (cnn={test_metrics['gate_mean'][0]:.3f}, "
         f"gnn={test_metrics['gate_mean'][1]:.3f})")
    for name, m in test_metrics["per_sf"].items():
        _log(f"  {name:<20s} P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} n={m['n']}")

    log_path.write_text("\n".join(log_lines) + "\n")

    ckpt = {
        "augment": augment,
        "epochs": EPOCHS,
        "subset_size": SUBSET_SIZE,
        "history": history,
        "test": test_metrics,
        "superfamily_names": superfamily_names,
    }
    torch.save(ckpt, ckpt_dir / f"smoke_{('aug' if augment else 'noaug')}.pt")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-aug", action="store_true",
                   help="Run the control arm only (no augmentation).")
    g.add_argument("--both", action="store_true",
                   help="Run both arms back-to-back.")
    args = ap.parse_args()

    sys.path.insert(0, str(V4_DIR / "cluster session"))
    os.chdir(V4_DIR / "cluster session")  # so relative imports/paths in the nb defs work
    ns = _exec_notebook_defs()
    os.chdir(THIS)

    summary = {}
    arms = []
    if args.both:
        arms = [True, False]
    elif args.no_aug:
        arms = [False]
    else:
        arms = [True]

    for augment in arms:
        log_path = THIS / f"smoke_{('aug' if augment else 'noaug')}.log"
        result = run_one_arm(ns, augment=augment, log_path=log_path, ckpt_dir=THIS)
        summary[("aug" if augment else "noaug")] = {
            "test_cls_macro_f1": result["test"]["cls_macro_f1"],
            "test_sf_macro_f1": result["test"]["sf_macro_f1"],
            "test_gate_mean": result["test"]["gate_mean"],
            "epochs": EPOCHS,
            "subset_size": SUBSET_SIZE,
        }

    if args.both and "aug" in summary and "noaug" in summary:
        d_sf = summary["aug"]["test_sf_macro_f1"] - summary["noaug"]["test_sf_macro_f1"]
        d_cls = summary["aug"]["test_cls_macro_f1"] - summary["noaug"]["test_cls_macro_f1"]
        summary["delta"] = {"sf_macro_f1": d_sf, "cls_macro_f1": d_cls}
        print(f"\nA/B delta (aug - noaug):  SF macro F1 = {d_sf:+.4f},  cls macro F1 = {d_cls:+.4f}")

    (THIS / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {THIS / 'smoke_summary.json'}")


if __name__ == "__main__":
    main()
