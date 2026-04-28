#!/usr/bin/env python3
"""
DANN Smoke Test for Species-Disjoint SF Generalisation
=======================================================
Implements Domain-Adversarial Neural Network (DANN) training on the Hybrid V4
TE classifier to reduce species-specific memorisation in the superfamily head.

The gradient-reversal layer (GRL) on a species-classifier head forces the fused
embedding to be species-agnostic, targeting the gate-collapse from 0.40/0.60 to
0.91/0.09 (CNN/GNN) observed on held-out species in the baseline model.

Usage:
    python dann_smoke_test.py              # full run on MPS/CPU
    python dann_smoke_test.py --device cuda  # override device

# ============================================================================
# SMOKE TEST PARAMS — edit these for a fast sanity check
EPOCHS = 3
subset_size = 5000
# To scale up: EPOCHS=25, subset_size=None
# ============================================================================

Expected outputs (saved to OUTPUT_DIR, default ./dann_smoke_results/):
    dann_epoch{N}.pt    — checkpoint after each epoch
    dann_metrics.json   — per-epoch metrics dict
    dann_species_f1.csv — per-species hAT precision/recall on held-out test
"""

# ============ SMOKE TEST PARAMS ============
EPOCHS = 3
SUBSET_SIZE = 5000   # set to None to use full dataset
# ===========================================

import argparse
import gc
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score

print(f"PyTorch {torch.__version__}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Paths — adjust to your local checkout.
# Defaults assume this script lives at:
#   <repo>/data_analysis/vgp_model_split_fix/v4/dann_smoke_test.py
_THIS_DIR  = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
_REPO_ROOT = _THIS_DIR.parents[2]   # TE-Classification/
FASTA_PATH = _REPO_ROOT / "data_analysis" / "all_vgp_tes.fa"
LABEL_PATH = _REPO_ROOT / "data_analysis" / "20251215-features-tpase"

OUTPUT_DIR = Path("./dann_smoke_results")

# Model architecture (Hybrid V4 — matches species-disjoint training notebook)
FIXED_LENGTH  = 20_000
MIN_CLASS_COUNT = 100
KMER_K    = 7
KMER_DIM  = 2048
KMER_WIN  = 512
KMER_STR  = 256
GNN_HID   = 128
GNN_LAY   = 3
CNN_W     = 128
MOTIF_K   = (7, 15, 21)
CTX_DILS  = (1, 2, 4, 8)
FUSE_DIM  = 256
FUSE_HEADS = 4
DROPOUT   = 0.15

BATCH_SIZE = 16
LR         = 1e-3
DANN_ALPHA = 0.10     # weight on species adversarial loss
LAM_MAX    = 1.0      # GRL lambda ceiling
LABEL_SMOOTH = 0.1

RANDOM_STATE = 42
TEST_SIZE    = 0.20   # fraction of species for held-out test


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_fasta(path):
    headers, sequences = [], []
    h, buf = None, []
    with open(path, "r") as f:
        for line in f:
            if not line:
                continue
            if line[0] == ">":
                if h is not None:
                    sequences.append("".join(buf).upper())
                    buf = []
                h = line[1:].strip()
                headers.append(h)
            else:
                buf.append(line.strip())
    if h is not None:
        sequences.append("".join(buf).upper())
    return headers, sequences


def load_hierarchical_labels(label_path):
    label_dict, binary_dict = {}, {}
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            header = parts[0].lstrip(">")
            tag = parts[1]
            label_dict[header] = tag
            binary_dict[header] = 0 if tag == "None" else 1
    return label_dict, binary_dict


def extract_genome_id(header: str) -> str:
    """e.g. 'hAT_1-aAnoBae#DNA/hAT' -> 'aAnoBae'"""
    return header.split("#")[0].rsplit("-", 1)[-1]


def compute_class_weights(y_ids, n_classes):
    counts = np.bincount(np.asarray(y_ids, dtype=np.int64), minlength=n_classes).astype(np.float64)
    w = 1.0 / np.sqrt(counts + 1e-6)
    return (w / w.mean()).astype(np.float32)


# ---------------------------------------------------------------------------
# K-mer featurizer
# ---------------------------------------------------------------------------

_ASCII_MAP = np.full(256, 4, dtype=np.uint8)
for _ch, _val in [("A", 0), ("C", 1), ("G", 2), ("T", 3),
                  ("a", 0), ("c", 1), ("g", 2), ("t", 3)]:
    _ASCII_MAP[ord(_ch)] = _val
_COMP = np.array([3, 2, 1, 0], dtype=np.uint8)


def _kmer_code(arr, rc=False):
    code = 0
    src = arr[::-1] if rc else arr
    if rc:
        for v in src:
            code = (code << 2) | int(_COMP[v])
    else:
        for v in src:
            code = (code << 2) | int(v)
    return code


def _canonical(arr):
    c1 = _kmer_code(arr, rc=False)
    c2 = _kmer_code(arr, rc=True)
    return c1 if c1 < c2 else c2


def _hash(x, dim):
    z = (x * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z ^= (z >> 33)
    z = (z * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF
    z ^= (z >> 29)
    return int(z % dim)


def featurize(seq: str, k=KMER_K, dim=KMER_DIM, win=KMER_WIN, stride=KMER_STR) -> np.ndarray:
    arr = _ASCII_MAP[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
    L = int(arr.size)
    out_dim = dim + 1
    if L == 0:
        return np.zeros((1, out_dim), dtype=np.float32)
    starts = (np.array([0]) if L <= win
              else np.arange(0, L - win + 1, stride, dtype=np.int64))
    if starts.size == 0:
        starts = np.array([0])
    X = np.zeros((starts.size, out_dim), dtype=np.float32)
    for wi, st in enumerate(starts):
        en = min(st + win, L)
        sub = arr[st:en]
        counts = np.zeros(dim, dtype=np.float32)
        total = 0
        for i in range(sub.size - k + 1):
            kmer = sub[i:i + k]
            if np.any(kmer == 4):
                continue
            j = _hash(_canonical(kmer), dim)
            counts[j] += 1.0
            total += 1
        if total > 0:
            counts /= total
        nrm = np.linalg.norm(counts)
        if nrm > 0:
            counts /= nrm
        X[wi, :-1] = counts
        X[wi, -1] = (st + en) / 2.0 / max(1.0, L)
    return X


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

ENCODE = np.full(256, 4, dtype=np.int64)
for _ch, _i in zip(b"ACGTNacgtn", [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]):
    ENCODE[_ch] = _i
REV_COMP = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)


def _chain_edges(n):
    if n <= 0:
        return torch.zeros((2, 0), dtype=torch.long)
    self_l = np.arange(n, dtype=np.int64)
    if n == 1:
        return torch.from_numpy(np.stack([self_l, self_l]))
    src = np.arange(n - 1, dtype=np.int64)
    dst = np.arange(1, n, dtype=np.int64)
    s = np.concatenate([src, dst, self_l])
    d = np.concatenate([dst, src, self_l])
    return torch.from_numpy(np.stack([s, d]))


class HybridDataset(Dataset):
    def __init__(self, headers, sequences, binary_labels, class_labels,
                 kmer_features, species_ids, fixed_length=FIXED_LENGTH, train=True):
        self.headers = list(headers)
        self.sequences = list(sequences)
        self.binary_labels = np.asarray(binary_labels, dtype=np.int64)
        self.class_labels = np.asarray(class_labels, dtype=np.int64)
        self.kmer_features = kmer_features
        self.species_ids = np.asarray(species_ids, dtype=np.int64)
        self.fixed_length = fixed_length
        self.train = train  # True -> random canvas placement; False -> deterministic (left-align)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_idx = ENCODE[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
        max_start = max(0, self.fixed_length - len(seq))
        if self.train and max_start > 0:
            start_pos = np.random.randint(0, max_start + 1)
        else:
            start_pos = 0  # deterministic for eval: always left-align
        return (
            self.headers[idx],
            seq_idx,
            int(self.binary_labels[idx]),
            int(self.class_labels[idx]),
            int(self.species_ids[idx]),
            start_pos,
            len(seq),
            self.kmer_features[idx],
        )


def collate_hybrid(batch, fixed_length=FIXED_LENGTH):
    headers, seq_idxs, bin_labs, cls_labs, sp_ids, starts, lengths, kmers = zip(*batch)
    B = len(batch)
    X_cnn = torch.zeros((B, 5, fixed_length), dtype=torch.float32)
    mask = torch.zeros((B, fixed_length), dtype=torch.bool)
    for i, (si, st, sl) in enumerate(zip(seq_idxs, starts, lengths)):
        al = min(sl, fixed_length - st)
        if al > 0:
            idx = torch.from_numpy(si[:al].astype(np.int64))
            pos = torch.arange(al, dtype=torch.long) + st
            X_cnn[i, idx, pos] = 1.0
            mask[i, st:st + al] = idx != 4
    Y_bin = torch.tensor(bin_labs, dtype=torch.long)
    Y_cls = torch.tensor(cls_labs, dtype=torch.long)
    Y_sp  = torch.tensor(sp_ids, dtype=torch.long)
    xs, eis, bvecs = [], [], []
    off = 0
    for gi, kf in enumerate(kmers):
        x = torch.from_numpy(kf).float()
        n = x.size(0)
        ei = _chain_edges(n)
        xs.append(x)
        eis.append(ei + off)
        bvecs.append(torch.full((n,), gi, dtype=torch.long))
        off += n
    x_gnn = torch.cat(xs, 0)
    edge_index = torch.cat(eis, 1) if eis else torch.zeros((2, 0), dtype=torch.long)
    batch_vec = torch.cat(bvecs, 0)
    return list(headers), X_cnn, mask, Y_bin, Y_cls, Y_sp, x_gnn, edge_index, batch_vec


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=9, d=1, drop=0.1):
        super().__init__()
        pad = (k // 2) * d
        self.conv = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=d)
        self.bn = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)

    def forward(self, x):
        y = self.drop(F.gelu(self.bn(self.conv(x))))
        return y + self.proj(x)


class MaskedMaxPool(nn.Module):
    def __init__(self, k=2, s=2):
        super().__init__()
        self.k = k
        self.s = s

    def forward(self, x, mask):
        if mask is not None:
            x = x * mask.unsqueeze(1).float() + (~mask.unsqueeze(1)) * (-1e9)
        xp = F.max_pool1d(x, self.k, self.s)
        if mask is None:
            return xp, None
        mp = F.max_pool1d(mask.float().unsqueeze(1), self.k, self.s).squeeze(1) > 0
        return xp, mp


def masked_avg(z, mask):
    if mask is None:
        return z.mean(-1)
    m = mask.unsqueeze(1).float()
    return (z * m).sum(-1) / m.sum(-1).clamp_min(1.0)


class CNNTower(nn.Module):
    def __init__(self, w=CNN_W, mk=MOTIF_K, cd=CTX_DILS, drop=DROPOUT):
        super().__init__()
        self.out_dim = w
        self.motifs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(5, w, k, padding=k // 2),
                          nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(drop))
            for k in mk
        ])
        self.mix = nn.Sequential(
            nn.Conv1d(w * len(mk), w, 1), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(drop))
        self.ctx = nn.ModuleList([ConvBlock(w, w, 9, d, drop) for d in cd])
        self.pool = MaskedMaxPool()

    def encode(self, x, mask):
        z = self.mix(torch.cat([c(x) for c in self.motifs], 1))
        m = mask
        for b in self.ctx:
            z = b(z); z, m = self.pool(z, m)
        return masked_avg(z, m)

    def forward(self, x, mask):
        f = self.encode(x, mask)
        xrc = x.index_select(1, REV_COMP.to(x.device)).flip(-1)
        mrc = None if mask is None else mask.flip(-1)
        return 0.5 * (f + self.encode(xrc, mrc))


def scatter_mean(x, idx, dim_size):
    out = torch.zeros(dim_size, x.size(1), device=x.device, dtype=x.dtype)
    out.index_add_(0, idx, x)
    ones = torch.ones(idx.size(0), device=x.device, dtype=x.dtype)
    cnt = torch.zeros(dim_size, device=x.device, dtype=x.dtype)
    cnt.index_add_(0, idx, ones)
    return out / cnt.clamp_min(1).unsqueeze(1)


class GNNLayer(nn.Module):
    def __init__(self, in_d, out_d, drop=0.1):
        super().__init__()
        self.ls = nn.Linear(in_d, out_d)
        self.ln = nn.Linear(in_d, out_d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, ei):
        src, dst = ei[0], ei[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        ones = torch.ones(dst.size(0), device=x.device, dtype=x.dtype)
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, ones)
        agg = agg / deg.clamp_min(1).unsqueeze(1)
        return self.drop(F.relu(self.ls(x) + self.ln(agg)))


class GNNTower(nn.Module):
    def __init__(self, in_d=KMER_DIM + 1, hid=GNN_HID, n=GNN_LAY, drop=DROPOUT):
        super().__init__()
        self.out_dim = hid
        self.layers = nn.ModuleList()
        d = in_d
        for _ in range(n):
            self.layers.append(GNNLayer(d, hid, drop)); d = hid

    def forward(self, x, ei, bvec, B=None):
        for l in self.layers: x = l(x, ei)
        if B is None: B = int(bvec.max().item()) + 1 if bvec.numel() else 0
        return scatter_mean(x, bvec, B)


class FusionModule(nn.Module):
    def __init__(self, cd=CNN_W, gd=GNN_HID, fd=FUSE_DIM, nh=FUSE_HEADS, drop=DROPOUT):
        super().__init__()
        self.cp = nn.Linear(cd, fd); self.gp = nn.Linear(gd, fd)
        self.ln1 = nn.LayerNorm(fd); self.ln2 = nn.LayerNorm(fd)
        self.attn = nn.MultiheadAttention(fd, nh, dropout=drop, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(fd * 2, fd), nn.GELU(),
                                  nn.Linear(fd, 2), nn.Softmax(dim=-1))
        self.out = nn.Sequential(nn.Linear(fd, fd), nn.GELU(), nn.Dropout(drop))

    def forward(self, ce, ge):
        c = self.ln1(self.cp(ce)); g = self.ln2(self.gp(ge))
        comb = torch.stack([c, g], 1)
        ao, _ = self.attn(comb, comb, comb)
        ca, ga = ao[:, 0], ao[:, 1]
        gw = self.gate(torch.cat([ca, ga], -1))
        fused = gw[:, :1] * ca + gw[:, 1:] * ga
        return self.out(fused), gw


class HybridV4WithDANN(nn.Module):
    """Hybrid V4 with an optional DANN species-adversary branch."""

    def __init__(self, n_sf, n_species, drop=DROPOUT):
        super().__init__()
        self.cnn_tower = CNNTower(drop=drop)
        self.gnn_tower = GNNTower(drop=drop)
        self.fusion = FusionModule(drop=drop)
        self.binary_head = nn.Sequential(
            nn.Linear(FUSE_DIM, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 2))
        self.sf_head = nn.Sequential(
            nn.Linear(FUSE_DIM, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, n_sf))
        # Species adversary reads from fused through GRL
        self.species_adversary = SpeciesAdversary(FUSE_DIM, n_species, drop=0.3)

    def encode(self, x_cnn, mask, x_gnn, ei, bvec):
        B = x_cnn.size(0)
        ce = self.cnn_tower(x_cnn, mask)
        ge = self.gnn_tower(x_gnn, ei, bvec, B=B)
        fused, gw = self.fusion(ce, ge)
        return fused, gw

    def forward(self, x_cnn, mask, x_gnn, ei, bvec):
        fused, gw = self.encode(x_cnn, mask, x_gnn, ei, bvec)
        return self.binary_head(fused), self.sf_head(fused), gw, fused


# ---------------------------------------------------------------------------
# Gradient Reversal Layer + Species Adversary
# ---------------------------------------------------------------------------

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GradRevLayer(nn.Module):
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return _GRL.apply(x, self.lam)


class SpeciesAdversary(nn.Module):
    """Reads fused embedding through GRL, predicts species."""

    def __init__(self, fusion_dim: int, n_species: int, drop: float = 0.3):
        super().__init__()
        self.grl = GradRevLayer(lam=1.0)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(fusion_dim // 2, n_species),
        )

    def set_lam(self, lam: float):
        self.grl.lam = lam

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.head(self.grl(fused))


# ---------------------------------------------------------------------------
# Label-smoothing CE
# ---------------------------------------------------------------------------

class LabelSmoothCE(nn.Module):
    def __init__(self, s=LABEL_SMOOTH, weight=None):
        super().__init__()
        self.s = s
        self.register_buffer("w", weight if weight is not None else torch.empty(0))

    def forward(self, logits, targets):
        n = logits.size(-1)
        lp = F.log_softmax(logits, -1)
        oh = torch.zeros_like(lp).scatter_(1, targets.unsqueeze(1), 1)
        sm = (1 - self.s) * oh + self.s / n
        if self.w.numel() > 0:
            wt = self.w[targets].unsqueeze(1)
            return -(sm * lp * wt).sum(-1).mean()
        return -(sm * lp).sum(-1).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def dann_schedule(epoch: int, total_epochs: int) -> float:
    """DANN GRL lambda: 0 at epoch 1, ramps to LAM_MAX by epoch=total_epochs."""
    p = (epoch - 1) / max(1, total_epochs - 1)
    return LAM_MAX * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def run_training(
    device: torch.device,
    fasta_path: Path,
    label_path: Path,
    output_dir: Path,
    epochs: int = EPOCHS,
    subset_size: Optional[int] = SUBSET_SIZE,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"DANN Smoke Test  |  device={device}  |  epochs={epochs}")
    if subset_size:
        print(f"  subset_size={subset_size}  (set to None to run full dataset)")
    print(f"{'='*60}\n")

    # ---- Load data ----
    print("Loading FASTA...")
    headers, sequences = read_fasta(fasta_path)
    label_dict, binary_dict = load_hierarchical_labels(label_path)

    all_h, all_s, all_tags, all_bin = [], [], [], []
    for h, s in zip(headers, sequences):
        if h in label_dict:
            all_h.append(h); all_s.append(s)
            all_tags.append(label_dict[h]); all_bin.append(binary_dict[h])
    del headers, sequences
    print(f"Loaded {len(all_h)} sequences")

    # ---- Extract species IDs ----
    genome_ids = [extract_genome_id(h) for h in all_h]
    species_names = sorted(set(genome_ids))
    species_to_id = {s: i for i, s in enumerate(species_names)}
    sp_ids = np.array([species_to_id[g] for g in genome_ids], dtype=np.int64)
    n_species = len(species_names)
    print(f"Species: {n_species}")

    # ---- Build superfamily mapping ----
    tpase_tags = [t for t, b in zip(all_tags, all_bin) if b == 1]
    tag_counts = Counter(tpase_tags)
    sf_names = sorted(t for t, c in tag_counts.items() if c >= MIN_CLASS_COUNT)
    sf_to_id = {t: i for i, t in enumerate(sf_names)}
    n_sf = len(sf_names)
    print(f"Superfamilies ({n_sf}): {sf_names}")

    # ---- Filter ----
    fh, fs, fbin, fcls, fsp = [], [], [], [], []
    for h, s, tag, b, sp in zip(all_h, all_s, all_tags, all_bin, sp_ids):
        if b == 0:
            fh.append(h); fs.append(s); fbin.append(0); fcls.append(0); fsp.append(int(sp))
        elif tag in sf_to_id:
            fh.append(h); fs.append(s); fbin.append(1); fcls.append(sf_to_id[tag]); fsp.append(int(sp))
    del all_h, all_s, all_tags, all_bin, sp_ids
    gc.collect()
    print(f"After filtering: {len(fh)} sequences")

    # Optional subset for smoke test
    if subset_size is not None and len(fh) > subset_size:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(fh), subset_size, replace=False)
        fh = [fh[i] for i in idx]
        fs = [fs[i] for i in idx]
        fbin = [fbin[i] for i in idx]
        fcls = [fcls[i] for i in idx]
        fsp  = [fsp[i] for i in idx]
        print(f"Subsampled to {len(fh)} sequences (smoke mode)")

    fbin = np.array(fbin, dtype=np.int64)
    fcls = np.array(fcls, dtype=np.int64)
    fsp  = np.array(fsp, dtype=np.int64)

    # ---- Species-disjoint split ----
    print("Building species-disjoint split...")
    # Group by genome id; GroupShuffleSplit ensures no species leaks between train/test
    genome_id_per_sample = [extract_genome_id(h) for h in fh]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    idx_tr, idx_te = next(gss.split(fh, groups=genome_id_per_sample))

    tr_h = [fh[i] for i in idx_tr]; te_h = [fh[i] for i in idx_te]
    tr_s = [fs[i] for i in idx_tr]; te_s = [fs[i] for i in idx_te]
    tr_bin = fbin[idx_tr]; te_bin = fbin[idx_te]
    tr_cls = fcls[idx_tr]; te_cls = fcls[idx_te]
    tr_sp  = fsp[idx_tr];  te_sp  = fsp[idx_te]

    tr_species = set(genome_id_per_sample[i] for i in idx_tr)
    te_species = set(genome_id_per_sample[i] for i in idx_te)
    assert len(tr_species & te_species) == 0, "Species leak detected!"
    print(f"Train: {len(tr_h)} seqs, {len(tr_species)} species")
    print(f"Test:  {len(te_h)} seqs, {len(te_species)} species (disjoint)")

    # ---- Precompute k-mer features ----
    print("Computing k-mer features (this takes a few minutes)...")
    tr_kmer = [featurize(s) for s in tqdm(tr_s, desc="train")]
    te_kmer = [featurize(s) for s in tqdm(te_s, desc="test")]

    # ---- Datasets / loaders ----
    ds_tr = HybridDataset(tr_h, tr_s, tr_bin, tr_cls, tr_kmer, tr_sp, train=True)
    ds_te = HybridDataset(te_h, te_s, te_bin, te_cls, te_kmer, te_sp, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, collate_fn=collate_hybrid)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0, collate_fn=collate_hybrid)

    # ---- Model ----
    torch.manual_seed(RANDOM_STATE)
    model = HybridV4WithDANN(n_sf=n_sf, n_species=n_species, drop=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ---- Loss functions ----
    n_pos = float(tr_bin.sum()); n_neg = float((tr_bin == 0).sum())
    bin_w = torch.tensor([n_pos / (n_pos + n_neg), n_neg / (n_pos + n_neg)],
                         dtype=torch.float32, device=device)
    bin_loss_fn = nn.CrossEntropyLoss(weight=bin_w)

    tpase_mask = tr_bin == 1
    sf_w = torch.tensor(compute_class_weights(tr_cls[tpase_mask], n_sf),
                        dtype=torch.float32, device=device)
    sf_loss_fn = LabelSmoothCE(s=LABEL_SMOOTH, weight=sf_w)

    sp_loss_fn = nn.CrossEntropyLoss()

    # ---- Optimizer ----
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=LR * 0.01)

    # ---- Training loop ----
    history = {"train_loss": [], "val_binary_f1": [], "val_sf_f1": [],
               "gate_cnn": [], "dann_lam": []}
    best_sf_f1 = -1.0
    print(f"\nStarting training ({epochs} epochs)...\n")

    for ep in range(1, epochs + 1):
        lam = dann_schedule(ep, epochs)
        model.species_adversary.set_lam(lam)
        model.train()
        run_loss = 0.0
        run_sp_loss = 0.0

        for batch in tqdm(dl_tr, desc=f"Ep {ep}/{epochs}", leave=False):
            hdr, X_cnn, mask, Y_bin, Y_cls, Y_sp, x_gnn, ei, bvec = batch
            X_cnn = X_cnn.to(device); mask = mask.to(device)
            Y_bin = Y_bin.to(device); Y_cls = Y_cls.to(device); Y_sp = Y_sp.to(device)
            x_gnn = x_gnn.to(device); ei = ei.to(device); bvec = bvec.to(device)

            bin_logits, sf_logits, gw, fused = model(X_cnn, mask, x_gnn, ei, bvec)

            # Task losses
            loss_bin = bin_loss_fn(bin_logits, Y_bin)
            tmask = Y_bin == 1
            loss_sf = sf_loss_fn(sf_logits[tmask], Y_cls[tmask]) if tmask.sum() > 0 \
                else torch.zeros(1, device=device).squeeze()

            # Species adversarial loss (gradient reversal through GRL)
            sp_logits = model.species_adversary(fused)
            loss_sp = sp_loss_fn(sp_logits, Y_sp)

            loss = loss_bin + loss_sf + DANN_ALPHA * loss_sp

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()
            run_sp_loss += loss_sp.item()

        sched.step()

        # ---- Evaluation ----
        model.eval()
        all_bp, all_bt, all_sp2, all_st, all_gc = [], [], [], [], []
        with torch.no_grad():
            for batch in dl_te:
                hdr, X_cnn, mask, Y_bin, Y_cls, _, x_gnn, ei, bvec = batch
                X_cnn = X_cnn.to(device); mask = mask.to(device)
                x_gnn = x_gnn.to(device); ei = ei.to(device); bvec = bvec.to(device)
                bl, sl, gw, _ = model(X_cnn, mask, x_gnn, ei, bvec)
                bp = bl.argmax(1).cpu().numpy()
                sp_pred = sl.argmax(1).cpu().numpy()
                all_bp.extend(bp); all_bt.extend(Y_bin.numpy())
                tmask = Y_bin == 1
                all_sp2.extend(sp_pred[tmask.numpy()])
                all_st.extend(Y_cls[tmask].numpy())
                all_gc.extend(gw[:, 0].cpu().numpy())

        all_bp = np.array(all_bp); all_bt = np.array(all_bt)
        all_sp2 = np.array(all_sp2); all_st = np.array(all_st)
        bin_f1 = f1_score(all_bt, all_bp, average="binary")
        sf_f1 = f1_score(all_st, all_sp2, average="macro", zero_division=0) if len(all_st) > 0 else 0
        gate_cnn = float(np.mean(all_gc))

        history["train_loss"].append(float(run_loss / len(dl_tr)))
        history["val_binary_f1"].append(float(bin_f1))
        history["val_sf_f1"].append(float(sf_f1))
        history["gate_cnn"].append(gate_cnn)
        history["dann_lam"].append(float(lam))

        print(f"Ep {ep:2d}: loss={run_loss/len(dl_tr):.4f}  "
              f"bin_f1={bin_f1:.3f}  sf_f1={sf_f1:.3f}  "
              f"gate_CNN={gate_cnn:.2f}  lam={lam:.3f}  "
              f"sp_loss={run_sp_loss/len(dl_tr):.4f}")

        # Save checkpoint
        if sf_f1 >= best_sf_f1:
            best_sf_f1 = sf_f1
            ckpt_path = output_dir / f"dann_epoch{ep}_sf{sf_f1:.3f}.pt"
            torch.save({
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": ep, "sf_f1": sf_f1, "bin_f1": bin_f1,
                "sf_names": sf_names, "sf_to_id": sf_to_id,
                "species_names": species_names, "species_to_id": species_to_id,
                "n_sf": n_sf, "n_species": n_species,
            }, ckpt_path)
            print(f"  => Checkpoint saved ({ckpt_path.name})")

    # ---- Save metrics ----
    metrics_path = output_dir / "dann_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # ---- Per-species SF breakdown on held-out test ----
    _compute_per_species_breakdown(
        model, dl_te, sf_names, device, output_dir, genome_id_per_sample, idx_te, te_h
    )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Best SF macro F1: {best_sf_f1:.4f}")
    print(f"\nResults in: {output_dir.resolve()}")
    return history


def _compute_per_species_breakdown(
    model, dl_te, sf_names, device, output_dir, genome_id_per_sample, idx_te, te_h
):
    """Write a CSV with per-species SF precision/recall and mean gate weight."""
    import csv
    hat_id = next((i for i, n in enumerate(sf_names) if "hAT" in n), None)

    model.eval()
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0, "gc": []})
    with torch.no_grad():
        for batch in dl_te:
            hdr, X_cnn, mask, Y_bin, Y_cls, _, x_gnn, ei, bvec = batch
            X_cnn = X_cnn.to(device); mask = mask.to(device)
            x_gnn = x_gnn.to(device); ei = ei.to(device); bvec = bvec.to(device)
            bl, sl, gw, _ = model(X_cnn, mask, x_gnn, ei, bvec)
            sp_pred = sl.argmax(1).cpu().numpy()
            bin_true = Y_bin.numpy()
            sf_true  = Y_cls.numpy()
            gate_cnn = gw[:, 0].cpu().numpy()

            for i, h in enumerate(hdr):
                g = extract_genome_id(h)
                stats[g]["n"] += 1
                stats[g]["gc"].append(float(gate_cnn[i]))
                if bin_true[i] == 1 and hat_id is not None:
                    if sf_true[i] == hat_id:
                        if sp_pred[i] == hat_id: stats[g]["tp"] += 1
                        else: stats[g]["fn"] += 1
                    elif sp_pred[i] == hat_id:
                        stats[g]["fp"] += 1

    csv_path = output_dir / "dann_species_f1.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "n_dna", "hat_prec", "hat_rec", "gate_cnn_mean"])
        for sp, d in sorted(stats.items(), key=lambda x: -x[1]["n"]):
            tp, fp, fn = d["tp"], d["fp"], d["fn"]
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            gc   = sum(d["gc"]) / len(d["gc"]) if d["gc"] else 0.0
            w.writerow([sp, d["n"], f"{prec:.3f}", f"{rec:.3f}", f"{gc:.3f}"])
    print(f"Per-species breakdown saved to {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DANN smoke test for VGP Hybrid V4")
    parser.add_argument("--device", default=None, help="Force device (cuda/mps/cpu)")
    parser.add_argument("--fasta", default=str(FASTA_PATH))
    parser.add_argument("--labels", default=str(LABEL_PATH))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--subset", type=int, default=SUBSET_SIZE,
                        help="Subsample N sequences for smoke test; 0 = full dataset")
    parser.add_argument("--dann-alpha", type=float, default=DANN_ALPHA,
                        help="Weight on species adversarial loss (default 0.10)")
    args = parser.parse_args()

    DANN_ALPHA = args.dann_alpha
    dev = resolve_device(args.device)
    subset = args.subset if args.subset and args.subset > 0 else None

    run_training(
        device=dev,
        fasta_path=Path(args.fasta),
        label_path=Path(args.labels),
        output_dir=Path(args.output),
        epochs=args.epochs,
        subset_size=subset,
    )
