"""Build 3 self-contained CUDA notebooks for the imbalance-handling ablation.

Run from this directory:
    python _generate_notebooks.py

Emits:
    binary_dna_natural_focal.ipynb
    binary_dna_natural_threshold_tuned.ipynb
    three_class_natural_weighted.ipynb

Architecture is the proven v4.3 trunk (lifted verbatim from
data_analysis/vgp_model_data_tpase_multi/v4.3/vgp_features_tpase_multiclass_v4.2.ipynb
and models/scripts/hybrid_v4_3_train.py). The only per-notebook differences
are: head shape, loss, sampling, and the final analysis block.
"""
from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in dedent(text).strip("\n").splitlines()],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in dedent(text).strip("\n").splitlines()],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Shared cell content (same in every notebook)
# ---------------------------------------------------------------------------

CELL_IMPORTS = """
# ============ Imports & Reproducibility ============
import os, gc, math, json, time, random, heapq
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"CUDA: {torch.cuda.get_device_name(0)} | {torch.cuda.device_count()} device(s)")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps"); print("MPS")
else:
    DEVICE = torch.device("cpu"); print("CPU")
"""

CELL_CONFIG_BASE = """
# ============ Paths & Config ============
# Update FASTA_PATH / LABEL_PATH if your repo is laid out differently.
ROOT = Path.cwd()
# Notebook lives in models/scripts/cuda_experiments/ — walk up to repo root.
for _ in range(5):
    if (ROOT / "data" / "vgp" / "all_vgp_tes.fa").exists():
        break
    ROOT = ROOT.parent
FASTA_PATH = ROOT / "data" / "vgp" / "all_vgp_tes.fa"
LABEL_PATH = ROOT / "data" / "vgp" / "20260120_features_sf"
SAVE_DIR = Path.cwd() / "results___NB_TAG__"
SAVE_DIR.mkdir(exist_ok=True, parents=True)
print(f"FASTA : {FASTA_PATH} ({'OK' if FASTA_PATH.exists() else 'MISSING'})")
print(f"LABEL : {LABEL_PATH} ({'OK' if LABEL_PATH.exists() else 'MISSING'})")
print(f"SAVE  : {SAVE_DIR}")

EXCLUDE_GENOMES = {"mOrnAna", "bTaeGut", "rAllMis"}
FIXED_LENGTH = 20_000
MIN_CLASS_COUNT = 100   # drop superfamilies with < this many samples
MAX_PER_SF = None       # NO cap — preserve natural prior

# Architecture (v4.3 trunk, ~8M params)
CNN_WIDTH         = 128
MOTIF_KERNELS     = (7, 15, 21)
CONTEXT_DILATIONS = (1, 2, 4, 8)
RC_FUSION_MODE    = "late"
KMER_K            = 7
KMER_DIM          = 2048
KMER_WINDOW       = 512
KMER_STRIDE       = 256
GNN_HIDDEN        = 128
GNN_LAYERS        = 3
FUSION_DIM        = 256
NUM_HEADS         = 4
DROPOUT           = 0.15

# Training
BATCH_SIZE        = 16
EPOCHS            = __EPOCHS__
LR                = 1e-3
PATIENCE          = 12
N_FOLDS           = 5
TEST_FRACTION     = 0.2
LABEL_SMOOTHING   = 0.1
"""

CELL_DATA_HELPERS = """
# ============ FASTA + label loaders ============
def read_fasta(path):
    headers, sequences, h, buf = [], [], None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
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


def _extract_genome_id(header: str) -> str:
    name_part = header.split("#")[0]
    return name_part.rsplit("-", 1)[-1]


def load_multiclass_labels(label_path, keep_classes):
    class_to_id = {c: i for i, c in enumerate(keep_classes)}
    label_dict, class_dict = {}, {}
    counts = Counter()
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            header, tag = parts[0].lstrip(">"), parts[1]
            top = tag.split("/")[0]
            if top in class_to_id:
                label_dict[header] = tag
                class_dict[header] = class_to_id[top]
                counts[tag] += 1
    return label_dict, class_dict, counts


def compute_class_weights(y, n, mode="inv_sqrt", eps=1e-6):
    c = np.bincount(np.asarray(y, dtype=np.int64), minlength=n).astype(np.float64)
    if mode == "none":     w = np.ones(n)
    elif mode == "inv":    w = 1.0 / (c + eps)
    elif mode == "inv_sqrt": w = 1.0 / np.sqrt(c + eps)
    else: raise ValueError(mode)
    return (w / (w.mean() + 1e-12)).astype(np.float32)
"""

CELL_KMER = """
# ============ K-mer Window Featurizer ============
_ASCII_MAP = np.full(256, 4, dtype=np.uint8)
for ch, v in [("A",0),("C",1),("G",2),("T",3),("a",0),("c",1),("g",2),("t",3)]:
    _ASCII_MAP[ord(ch)] = v
_COMP = np.array([3, 2, 1, 0], dtype=np.uint8)


def _code_fwd(arr):
    code = 0
    for v in arr: code = (code << 2) | int(v)
    return code

def _code_rc(arr):
    code = 0
    for v in arr[::-1]: code = (code << 2) | int(_COMP[v])
    return code

def _canon(arr):
    a, b = _code_fwd(arr), _code_rc(arr)
    return a if a < b else b

def _hash(x, dim):
    z = (x * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z ^= (z >> 33); z = (z * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF
    z ^= (z >> 29)
    return int(z % dim)


@dataclass
class KmerWindowFeaturizer:
    k: int = 7
    dim: int = 2048
    window: int = 512
    stride: int = 256
    add_pos: bool = True
    l2_normalize: bool = True

    def featurize_sequence(self, seq: str):
        arr = _ASCII_MAP[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
        L = int(arr.size)
        out_dim = self.dim + (1 if self.add_pos else 0)
        if L == 0:
            return np.zeros((1, out_dim), dtype=np.float32)
        starts = np.array([0]) if L <= self.window else np.arange(0, L - self.window + 1, self.stride)
        if starts.size == 0: starts = np.array([0])
        X = np.zeros((starts.size, out_dim), dtype=np.float32)
        for wi, st in enumerate(starts):
            en = min(st + self.window, L)
            sub = arr[st:en]
            counts = np.zeros(self.dim, dtype=np.float32)
            total = 0
            if sub.size >= self.k:
                for i in range(sub.size - self.k + 1):
                    kmer = sub[i:i+self.k]
                    if np.any(kmer == 4): continue
                    counts[_hash(_canon(kmer), self.dim)] += 1.0
                    total += 1
            if total > 0: counts /= float(total)
            if self.l2_normalize:
                n = np.linalg.norm(counts)
                if n > 0: counts /= n
            if self.add_pos:
                X[wi, :-1] = counts
                X[wi, -1] = (st + en) / 2.0 / max(1.0, float(L))
            else:
                X[wi] = counts
        return X
"""

CELL_DATASET = """
# ============ Dataset & Collate ============
ENCODE = np.full(256, 4, dtype=np.int64)
for _ch, _idx in zip(b"ACGTNacgtn", [0,1,2,3,4,0,1,2,3,4]):
    ENCODE[_ch] = _idx
REV_COMP = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)


def build_chain_edge_index(n: int) -> torch.Tensor:
    edges = []
    if n > 1:
        src = np.arange(n - 1, dtype=np.int64); dst = np.arange(1, n, dtype=np.int64)
        edges.append((src, dst)); edges.append((dst, src))
    idx = np.arange(n, dtype=np.int64)
    edges.append((idx, idx))
    s = np.concatenate([e[0] for e in edges]); d = np.concatenate([e[1] for e in edges])
    return torch.from_numpy(np.stack([s, d], axis=0))


class HybridDataset(Dataset):
    def __init__(self, headers, sequences, y_top, y_sf, kmer_features, fixed_length=FIXED_LENGTH):
        self.headers = list(headers); self.sequences = list(sequences)
        self.y_top = np.asarray(y_top, dtype=np.int64)
        self.y_sf  = np.asarray(y_sf,  dtype=np.int64)
        self.kmer_features = kmer_features
        self.fixed_length = fixed_length

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = len(seq)
        seq_idx = ENCODE[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
        max_start = max(0, self.fixed_length - seq_len)
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        return (self.headers[idx], seq_idx, int(self.y_top[idx]), int(self.y_sf[idx]),
                start, start + seq_len, seq_len, self.kmer_features[idx])


def collate_hybrid(batch, fixed_length=FIXED_LENGTH):
    (headers, seq_idxs, y_top, y_sf, starts, ends, lengths, kmer_feats) = zip(*batch)
    B = len(batch)
    X_cnn = torch.zeros((B, 5, fixed_length), dtype=torch.float32)
    mask  = torch.zeros((B, fixed_length), dtype=torch.bool)
    for i, (seq_idx, start, end, seq_len) in enumerate(zip(seq_idxs, starts, ends, lengths)):
        actual = min(seq_len, fixed_length - start)
        if actual > 0:
            idx = torch.from_numpy(seq_idx[:actual].astype(np.int64))
            pos = torch.arange(actual, dtype=torch.long) + start
            X_cnn[i, idx, pos] = 1.0
            mask[i, start:start+actual] = (idx != 4)
    Y_top = torch.tensor(y_top, dtype=torch.long)
    Y_sf  = torch.tensor(y_sf,  dtype=torch.long)
    xs, eis, batch_vecs = [], [], []
    offset = 0
    for gi, kf in enumerate(kmer_feats):
        x = torch.from_numpy(kf).to(torch.float32); n = x.size(0)
        ei = build_chain_edge_index(n)
        xs.append(x); eis.append(ei + offset)
        batch_vecs.append(torch.full((n,), gi, dtype=torch.int64))
        offset += n
    x_gnn = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros((2,0), dtype=torch.int64)
    batch_vec = torch.cat(batch_vecs, dim=0)
    return list(headers), X_cnn, mask, Y_top, Y_sf, x_gnn, edge_index, batch_vec
"""

CELL_CNN_TOWER = """
# ============ CNN Tower (v4.3) ============
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=9, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)
    def forward(self, x):
        y = F.gelu(self.bn(self.conv(x))); y = self.drop(y)
        return y + self.proj(x)


class MaskedMaxPool1d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
    def forward(self, x, mask):
        if mask is not None:
            m = mask.unsqueeze(1).float()
            x = x * m + (~mask.unsqueeze(1)) * (-1e9)
        x_p = F.max_pool1d(x, self.kernel_size, self.stride)
        if mask is None: return x_p, None
        m_p = F.max_pool1d(mask.float().unsqueeze(1), self.kernel_size, self.stride).squeeze(1) > 0
        return x_p, m_p


def masked_avg_pool(z, mask):
    if mask is None: return z.mean(-1)
    m = mask.unsqueeze(1).float()
    return (z * m).sum(-1) / m.sum(-1).clamp_min(1.0)


class RCFirstConv1d(nn.Module):
    def __init__(self, out_ch, kernel_size=15, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(5, out_ch, kernel_size, padding=pad)
        self.bn = nn.BatchNorm1d(out_ch); self.dp = nn.Dropout(dropout)
    def forward(self, x):
        y1 = self.conv(x)
        x_rc = x.flip(-1).index_select(1, REV_COMP.to(x.device))
        y2 = self.conv(x_rc).flip(-1)
        return self.dp(F.gelu(self.bn(torch.max(y1, y2))))


class CNNTower(nn.Module):
    def __init__(self, width=128, motif_kernels=(7,15,21),
                 context_dilations=(1,2,4,8), dropout=0.15, rc_mode="late"):
        super().__init__()
        self.rc_mode = rc_mode; self.out_dim = width
        if rc_mode == "early":
            self.motif_convs = nn.ModuleList([
                RCFirstConv1d(width, kernel_size=k, dropout=dropout) for k in motif_kernels
            ])
        else:
            self.motif_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(5, width, kernel_size=k, padding=k//2),
                    nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout),
                ) for k in motif_kernels
            ])
        self.mix = nn.Sequential(
            nn.Conv1d(width * len(motif_kernels), width, 1),
            nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout),
        )
        self.context_blocks = nn.ModuleList([
            ConvBlock(width, width, 9, dilation=d, dropout=dropout) for d in context_dilations
        ])
        self.pool = MaskedMaxPool1d(2, 2)

    @staticmethod
    def rc_transform(x, mask):
        x_rc = x.index_select(1, REV_COMP.to(x.device)).flip(-1)
        return x_rc, (None if mask is None else mask.flip(-1))

    def encode(self, x, mask):
        z = self.mix(torch.cat([c(x) for c in self.motif_convs], dim=1))
        m = mask
        for blk in self.context_blocks:
            z = blk(z); z, m = self.pool(z, m)
        return masked_avg_pool(z, m)

    def forward(self, x, mask):
        if self.rc_mode == "late":
            f = self.encode(x, mask)
            x_rc, m_rc = self.rc_transform(x, mask)
            r = self.encode(x_rc, m_rc)
            return 0.5 * (f + r)
        return self.encode(x, mask)
"""

CELL_GNN_FUSION = """
# ============ GNN Tower + Cross-Modal Attention Fusion (v4.3) ============
def scatter_mean(x, idx, dim_size):
    out = torch.zeros((dim_size, x.size(1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, idx, x)
    cnt = torch.bincount(idx, minlength=dim_size).clamp_min(1).to(x.device).to(x.dtype).unsqueeze(1)
    return out / cnt


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.lin_self  = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.dp = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x); agg.index_add_(0, dst, x[src])
        deg = torch.bincount(dst, minlength=x.size(0)).clamp_min(1).to(x.device).to(x.dtype).unsqueeze(1)
        agg = agg / deg
        return self.dp(F.relu(self.lin_self(x) + self.lin_neigh(agg)))


class GNNTower(nn.Module):
    def __init__(self, in_dim, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.out_dim = hidden
        d = in_dim; layers = []
        for _ in range(n_layers):
            layers.append(GraphSAGELayer(d, hidden, dropout)); d = hidden
        self.layers = nn.ModuleList(layers)
    def forward(self, x, edge_index, batch_vec):
        for L in self.layers: x = L(x, edge_index)
        B = int(batch_vec.max().item()) + 1 if batch_vec.numel() else 0
        return scatter_mean(x, batch_vec, B)


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, cnn_dim=128, gnn_dim=128, fusion_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
        self.ln1 = nn.LayerNorm(fusion_dim); self.ln2 = nn.LayerNorm(fusion_dim)
        self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim), nn.GELU(),
            nn.Linear(fusion_dim, 2), nn.Softmax(dim=-1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout),
        )
    def forward(self, cnn_e, gnn_e):
        c = self.ln1(self.cnn_proj(cnn_e)); g = self.ln2(self.gnn_proj(gnn_e))
        combined = torch.stack([c, g], dim=1)
        attn, _ = self.cross_attn(combined, combined, combined)
        c_a, g_a = attn[:, 0], attn[:, 1]
        gate_w = self.gate(torch.cat([c_a, g_a], dim=-1))
        fused = gate_w[:, 0:1] * c_a + gate_w[:, 1:2] * g_a
        return self.out_proj(fused), gate_w
"""

# Variant-specific model class (binary vs three-class)
CELL_MODEL_BINARY = """
# ============ Hybrid Classifier (binary head) ============
class HybridTEClassifier(nn.Module):
    \"\"\"Binary classifier: DNA (1) vs non-DNA (0). Single logit output.\"\"\"
    def __init__(self, num_superfamilies, cnn_width=128, motif_kernels=(7,15,21),
                 context_dilations=(1,2,4,8), rc_mode="late",
                 gnn_in_dim=2049, gnn_hidden=128, gnn_layers=3,
                 fusion_dim=256, num_heads=4, dropout=0.15):
        super().__init__()
        self.num_superfamilies = num_superfamilies
        self.cnn_tower = CNNTower(cnn_width, motif_kernels, context_dilations, dropout, rc_mode)
        self.gnn_tower = GNNTower(gnn_in_dim, gnn_hidden, gnn_layers, dropout)
        self.fusion = CrossModalAttentionFusion(cnn_width, gnn_hidden, fusion_dim, num_heads, dropout)
        self.binary_head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        # SF head retained for richer logging (optional auxiliary signal).
        self.sf_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_superfamilies),
        )
    def forward(self, x_cnn, mask, x_gnn, edge_index, batch_vec):
        c = self.cnn_tower(x_cnn, mask)
        g = self.gnn_tower(x_gnn, edge_index, batch_vec)
        fused, gate = self.fusion(c, g)
        return self.binary_head(fused).squeeze(-1), self.sf_head(fused), gate
"""

CELL_MODEL_THREE = """
# ============ Hybrid Classifier (3-class head) ============
class HybridTEClassifier(nn.Module):
    \"\"\"Three-class top-level classifier (DNA/LTR/LINE) + SF head (v4.3 trunk).\"\"\"
    def __init__(self, num_classes, num_superfamilies, cnn_width=128,
                 motif_kernels=(7,15,21), context_dilations=(1,2,4,8), rc_mode="late",
                 gnn_in_dim=2049, gnn_hidden=128, gnn_layers=3,
                 fusion_dim=256, num_heads=4, dropout=0.15):
        super().__init__()
        self.num_classes = num_classes; self.num_superfamilies = num_superfamilies
        self.cnn_tower = CNNTower(cnn_width, motif_kernels, context_dilations, dropout, rc_mode)
        self.gnn_tower = GNNTower(gnn_in_dim, gnn_hidden, gnn_layers, dropout)
        self.fusion = CrossModalAttentionFusion(cnn_width, gnn_hidden, fusion_dim, num_heads, dropout)
        self.class_head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.sf_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_superfamilies),
        )
    def forward(self, x_cnn, mask, x_gnn, edge_index, batch_vec):
        c = self.cnn_tower(x_cnn, mask)
        g = self.gnn_tower(x_gnn, edge_index, batch_vec)
        fused, gate = self.fusion(c, g)
        return self.class_head(fused), self.sf_head(fused), gate
"""

CELL_LOSS_FOCAL = """
# ============ Focal Loss + Label Smoothing ============
class BinaryFocalLoss(nn.Module):
    \"\"\"Binary focal loss with class-balanced alpha on the positive class.\"\"\"
    def __init__(self, alpha_pos: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha_pos; self.gamma = gamma
    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing, self.weight = smoothing, weight
    def forward(self, logits, targets):
        n = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        oh = torch.zeros_like(logp).scatter_(1, targets.unsqueeze(1), 1)
        sm = (1 - self.smoothing) * oh + self.smoothing / n
        if self.weight is not None:
            w = self.weight[targets].unsqueeze(1)
            return -(sm * logp * w).sum(-1).mean()
        return -(sm * logp).sum(-1).mean()
"""

CELL_LOSS_BCE = """
# ============ BCE with pos_weight + Label Smoothing ============
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing, self.weight = smoothing, weight
    def forward(self, logits, targets):
        n = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        oh = torch.zeros_like(logp).scatter_(1, targets.unsqueeze(1), 1)
        sm = (1 - self.smoothing) * oh + self.smoothing / n
        if self.weight is not None:
            w = self.weight[targets].unsqueeze(1)
            return -(sm * logp * w).sum(-1).mean()
        return -(sm * logp).sum(-1).mean()
"""

CELL_LOSS_CE = CELL_LOSS_BCE  # 3-class notebook only needs label-smoothing CE.

CELL_DATA_PREP_BINARY = """
# ============ Load + Filter Data (binary: DNA vs non-DNA) ============
KEEP_CLASSES = ["DNA", "LTR", "LINE"]   # all three loaded; binary collapse below

print("Reading FASTA…")
headers, sequences = read_fasta(FASTA_PATH)
print(f"  {len(headers):,} sequences")

label_dict, class_dict, sf_counts = load_multiclass_labels(LABEL_PATH, KEEP_CLASSES)
print(f"Labels for {len(label_dict):,} sequences across {len(sf_counts)} superfamilies")

all_h, all_s, all_tags, all_top = [], [], [], []
for h, s in zip(headers, sequences):
    if h not in label_dict: continue
    if _extract_genome_id(h) in EXCLUDE_GENOMES: continue
    all_h.append(h); all_s.append(s)
    all_tags.append(label_dict[h]); all_top.append(class_dict[h])
del headers, sequences; gc.collect()
print(f"After genome filter: {len(all_h):,}")

# Drop tiny superfamilies
tag_counts = Counter(all_tags)
keep_sf = {t for t, c in tag_counts.items() if c >= MIN_CLASS_COUNT}
sf_names = sorted(keep_sf)
sf_to_id = {t: i for i, t in enumerate(sf_names)}
n_sf = len(sf_names)

filt_h, filt_s, filt_tags, filt_top, filt_sf = [], [], [], [], []
for h, s, t, top in zip(all_h, all_s, all_tags, all_top):
    if t in sf_to_id:
        filt_h.append(h); filt_s.append(s); filt_tags.append(t)
        filt_top.append(top); filt_sf.append(sf_to_id[t])
all_h, all_s, all_tags = filt_h, filt_s, filt_tags
all_top = np.array(filt_top, dtype=np.int64)
all_sf  = np.array(filt_sf,  dtype=np.int64)
print(f"After SF filter: {len(all_h):,} | {n_sf} superfamilies")

# Binary label: DNA (top==0) -> 1, else 0  (positive = minority)
all_bin = (all_top == KEEP_CLASSES.index("DNA")).astype(np.int64)
n_pos = int(all_bin.sum()); n_neg = len(all_bin) - n_pos
print(f"Binary distribution — DNA(+): {n_pos} ({100*n_pos/len(all_bin):.1f}%) | other(−): {n_neg}")

# NO max_per_sf cap.

# Pre-compute k-mer features (once, for all splits)
print("Pre-computing k-mer features…")
featurizer = KmerWindowFeaturizer(k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW, stride=KMER_STRIDE)
all_kmer = [featurizer.featurize_sequence(s) for s in tqdm(all_s)]
print("Done.")

# Stratify by superfamily so all rare SFs survive every split
strat = np.array(all_tags)
idx_tv, idx_te = train_test_split(np.arange(len(all_h)), test_size=TEST_FRACTION,
                                   stratify=strat, random_state=SEED)
test_h   = [all_h[i]    for i in idx_te]
test_s   = [all_s[i]    for i in idx_te]
test_bin = all_bin[idx_te]; test_sf_y = all_sf[idx_te]
test_kmer = [all_kmer[i] for i in idx_te]
tv_h, tv_s = [all_h[i] for i in idx_tv], [all_s[i] for i in idx_tv]
tv_bin, tv_sf_y = all_bin[idx_tv], all_sf[idx_tv]
tv_kmer = [all_kmer[i] for i in idx_tv]
tv_strat = strat[idx_tv]
print(f"TrainVal: {len(tv_h):,}  Test: {len(test_h):,}")
print(f"  TrainVal pos: {int(tv_bin.sum())} ({100*tv_bin.mean():.1f}%)")
print(f"  Test pos    : {int(test_bin.sum())} ({100*test_bin.mean():.1f}%)")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_splits = list(skf.split(tv_h, tv_strat))
print(f"{N_FOLDS}-fold rotating CV ready.")
"""

CELL_DATA_PREP_THREE = """
# ============ Load + Filter Data (3-class DNA/LTR/LINE) ============
KEEP_CLASSES = ["DNA", "LTR", "LINE"]
class_to_id = {c: i for i, c in enumerate(KEEP_CLASSES)}
n_classes = len(KEEP_CLASSES)

print("Reading FASTA…")
headers, sequences = read_fasta(FASTA_PATH)
print(f"  {len(headers):,} sequences")

label_dict, class_dict, sf_counts = load_multiclass_labels(LABEL_PATH, KEEP_CLASSES)
print(f"Labels for {len(label_dict):,} sequences across {len(sf_counts)} superfamilies")

all_h, all_s, all_tags, all_top = [], [], [], []
for h, s in zip(headers, sequences):
    if h not in label_dict: continue
    if _extract_genome_id(h) in EXCLUDE_GENOMES: continue
    all_h.append(h); all_s.append(s)
    all_tags.append(label_dict[h]); all_top.append(class_dict[h])
del headers, sequences; gc.collect()
print(f"After genome filter: {len(all_h):,}")

tag_counts = Counter(all_tags)
keep_sf = {t for t, c in tag_counts.items() if c >= MIN_CLASS_COUNT}
sf_names = sorted(keep_sf)
sf_to_id = {t: i for i, t in enumerate(sf_names)}
n_sf = len(sf_names)

filt_h, filt_s, filt_tags, filt_top, filt_sf = [], [], [], [], []
for h, s, t, top in zip(all_h, all_s, all_tags, all_top):
    if t in sf_to_id:
        filt_h.append(h); filt_s.append(s); filt_tags.append(t)
        filt_top.append(top); filt_sf.append(sf_to_id[t])
all_h, all_s, all_tags = filt_h, filt_s, filt_tags
all_top = np.array(filt_top, dtype=np.int64)
all_sf  = np.array(filt_sf,  dtype=np.int64)
print(f"After SF filter: {len(all_h):,} | {n_sf} superfamilies")

print("Class distribution (NATURAL — no SF cap):")
for cid, cname in enumerate(KEEP_CLASSES):
    n = int((all_top == cid).sum())
    print(f"  {cname}: {n} ({100*n/len(all_top):.1f}%)")

print("Pre-computing k-mer features…")
featurizer = KmerWindowFeaturizer(k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW, stride=KMER_STRIDE)
all_kmer = [featurizer.featurize_sequence(s) for s in tqdm(all_s)]
print("Done.")

strat = np.array(all_tags)
idx_tv, idx_te = train_test_split(np.arange(len(all_h)), test_size=TEST_FRACTION,
                                   stratify=strat, random_state=SEED)
test_h    = [all_h[i] for i in idx_te]
test_s    = [all_s[i] for i in idx_te]
test_top  = all_top[idx_te]; test_sf_y = all_sf[idx_te]
test_kmer = [all_kmer[i] for i in idx_te]
tv_h, tv_s = [all_h[i] for i in idx_tv], [all_s[i] for i in idx_tv]
tv_top, tv_sf_y = all_top[idx_tv], all_sf[idx_tv]
tv_kmer = [all_kmer[i] for i in idx_tv]
tv_strat = strat[idx_tv]
print(f"TrainVal: {len(tv_h):,}  Test: {len(test_h):,}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_splits = list(skf.split(tv_h, tv_strat))
print(f"{N_FOLDS}-fold rotating CV ready.")
"""

CELL_CKPT = """
# ============ Top-K Checkpoint Manager ============
class TopKCheckpointManager:
    def __init__(self, save_dir, prefix, k=5):
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix; self.k = k
        self.heap = []          # (-score, epoch)
        self.paths = {}         # epoch -> Path
    def maybe_save(self, score, epoch, model, arch, history):
        neg = -score
        if len(self.heap) < self.k:
            self._save(score, epoch, model, arch, history)
            heapq.heappush(self.heap, (neg, epoch))
            return True
        if neg < self.heap[0][0]:
            _, worst = heapq.heappop(self.heap)
            p = self.paths.pop(worst, None)
            if p and p.exists(): p.unlink()
            self._save(score, epoch, model, arch, history)
            heapq.heappush(self.heap, (neg, epoch))
            return True
        return False
    def _save(self, score, epoch, model, arch, history):
        sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        path = self.save_dir / f"{self.prefix}_epoch{epoch}.pt"
        torch.save({"model_state_dict": sd, "arch": arch,
                    "history": dict(history), "epoch": epoch, "score": score}, path)
        self.paths[epoch] = path
    def get_best(self):
        if not self.heap: return None, None
        neg, ep = min(self.heap)
        ckpt = torch.load(self.paths[ep], map_location="cpu", weights_only=False)
        return ckpt, ep
    def all_saved(self):
        return sorted([(-n, e) for n, e in self.heap], reverse=True)
"""

# Variant-specific training loops
CELL_TRAIN_BINARY_FOCAL = """
# ============ Build Model + Loss + Optimizer (Focal + WeightedRandomSampler) ============
arch = dict(num_superfamilies=n_sf, cnn_width=CNN_WIDTH, motif_kernels=tuple(MOTIF_KERNELS),
            context_dilations=tuple(CONTEXT_DILATIONS), rc_mode=RC_FUSION_MODE,
            gnn_in_dim=KMER_DIM + 1, gnn_hidden=GNN_HIDDEN, gnn_layers=GNN_LAYERS,
            fusion_dim=FUSION_DIM, num_heads=NUM_HEADS, dropout=DROPOUT)
model = HybridTEClassifier(**arch).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# Focal loss with α biased to the positive (DNA) minority class.
ALPHA_POS = 0.75; GAMMA = 2.0
binary_loss_fn = BinaryFocalLoss(alpha_pos=ALPHA_POS, gamma=GAMMA)

sf_w = compute_class_weights(tv_sf_y, n_sf, mode="inv_sqrt")
sf_loss_fn = LabelSmoothingCrossEntropy(LABEL_SMOOTHING,
              weight=torch.tensor(sf_w, dtype=torch.float32, device=DEVICE))

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)

# Build per-sample weights for WeightedRandomSampler (1/sqrt class freq)
def make_sampler(y_bin, replacement=True):
    counts = np.bincount(y_bin, minlength=2)
    w_class = 1.0 / np.sqrt(counts.clip(min=1))
    w_sample = w_class[y_bin]
    return WeightedRandomSampler(torch.from_numpy(w_sample).double(),
                                 num_samples=len(y_bin), replacement=replacement)

ds_test = HybridDataset(test_h, test_s, test_bin, test_sf_y, test_kmer)
loader_test = DataLoader(ds_test, BATCH_SIZE, shuffle=False, num_workers=0,
                         collate_fn=collate_hybrid)
ckpt = TopKCheckpointManager(SAVE_DIR, "binary_focal", k=5)
"""

CELL_TRAIN_BINARY_BCE = """
# ============ Build Model + Loss + Optimizer (BCE pos_weight, no resampling) ============
arch = dict(num_superfamilies=n_sf, cnn_width=CNN_WIDTH, motif_kernels=tuple(MOTIF_KERNELS),
            context_dilations=tuple(CONTEXT_DILATIONS), rc_mode=RC_FUSION_MODE,
            gnn_in_dim=KMER_DIM + 1, gnn_hidden=GNN_HIDDEN, gnn_layers=GNN_LAYERS,
            fusion_dim=FUSION_DIM, num_heads=NUM_HEADS, dropout=DROPOUT)
model = HybridTEClassifier(**arch).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# BCE-with-logits using neg/pos as pos_weight  (no resampling — natural shuffle).
pos = float(tv_bin.sum()); neg = float(len(tv_bin) - pos)
pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=DEVICE)
print(f"pos_weight (neg/pos) = {pos_weight.item():.2f}")
binary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

sf_w = compute_class_weights(tv_sf_y, n_sf, mode="inv_sqrt")
sf_loss_fn = LabelSmoothingCrossEntropy(LABEL_SMOOTHING,
              weight=torch.tensor(sf_w, dtype=torch.float32, device=DEVICE))

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)

ds_test = HybridDataset(test_h, test_s, test_bin, test_sf_y, test_kmer)
loader_test = DataLoader(ds_test, BATCH_SIZE, shuffle=False, num_workers=0,
                         collate_fn=collate_hybrid)
ckpt = TopKCheckpointManager(SAVE_DIR, "binary_bce", k=5)
"""

CELL_TRAIN_THREE = """
# ============ Build Model + Loss + Optimizer (3-class weighted CE) ============
arch = dict(num_classes=n_classes, num_superfamilies=n_sf, cnn_width=CNN_WIDTH,
            motif_kernels=tuple(MOTIF_KERNELS), context_dilations=tuple(CONTEXT_DILATIONS),
            rc_mode=RC_FUSION_MODE, gnn_in_dim=KMER_DIM + 1, gnn_hidden=GNN_HIDDEN,
            gnn_layers=GNN_LAYERS, fusion_dim=FUSION_DIM, num_heads=NUM_HEADS, dropout=DROPOUT)
model = HybridTEClassifier(**arch).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

cls_w = compute_class_weights(tv_top, n_classes, mode="inv_sqrt")
cls_w_t = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
class_loss_fn = LabelSmoothingCrossEntropy(LABEL_SMOOTHING, weight=cls_w_t)
print(f"Class weights: {dict(zip(KEEP_CLASSES, cls_w.tolist()))}")

sf_w = compute_class_weights(tv_sf_y, n_sf, mode="inv_sqrt")
sf_loss_fn = LabelSmoothingCrossEntropy(LABEL_SMOOTHING,
              weight=torch.tensor(sf_w, dtype=torch.float32, device=DEVICE))

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)

ds_test = HybridDataset(test_h, test_s, test_top, test_sf_y, test_kmer)
loader_test = DataLoader(ds_test, BATCH_SIZE, shuffle=False, num_workers=0,
                         collate_fn=collate_hybrid)
ckpt = TopKCheckpointManager(SAVE_DIR, "three_class", k=5)
"""

# Common training loop, parameterised by the variant via inline if-statements.
CELL_LOOP_BINARY = """
# ============ Training Loop (rotating K-fold, binary) ============
SF_LOSS_WEIGHT = 0.5     # auxiliary
history = {"train_loss": [], "train_bin": [], "train_sf": [],
           "val_acc": [], "val_f1_macro": [], "val_recall_pos": [], "val_precision_pos": [],
           "val_auroc": [], "val_auprc": [],
           "gate_cnn": [], "gate_gnn": [], "fold_used": []}
best_score, bad = -math.inf, 0
USE_SAMPLER = __USE_SAMPLER__

for ep in range(1, EPOCHS + 1):
    fold = (ep - 1) % N_FOLDS
    tr_idx, va_idx = fold_splits[fold]

    ds_train = HybridDataset([tv_h[i] for i in tr_idx], [tv_s[i] for i in tr_idx],
                             tv_bin[tr_idx], tv_sf_y[tr_idx], [tv_kmer[i] for i in tr_idx])
    ds_val   = HybridDataset([tv_h[i] for i in va_idx], [tv_s[i] for i in va_idx],
                             tv_bin[va_idx], tv_sf_y[va_idx], [tv_kmer[i] for i in va_idx])

    if USE_SAMPLER:
        sampler = make_sampler(tv_bin[tr_idx])
        loader_tr = DataLoader(ds_train, BATCH_SIZE, sampler=sampler,
                               num_workers=0, collate_fn=collate_hybrid)
    else:
        loader_tr = DataLoader(ds_train, BATCH_SIZE, shuffle=True,
                               num_workers=0, collate_fn=collate_hybrid)
    loader_va = DataLoader(ds_val, BATCH_SIZE, shuffle=False,
                           num_workers=0, collate_fn=collate_hybrid)

    model.train()
    run_loss = run_bin = run_sf = 0.0; n_seen = 0
    pbar = tqdm(loader_tr, desc=f"Ep {ep}/{EPOCHS} fold {fold+1}", leave=False)
    for _, X, M, Yb, Ys, xg, ei, bv in pbar:
        X = X.to(DEVICE); M = M.to(DEVICE); Yb = Yb.to(DEVICE); Ys = Ys.to(DEVICE)
        xg = xg.to(DEVICE); ei = ei.to(DEVICE); bv = bv.to(DEVICE)
        bin_logit, sf_logit, gate = model(X, M, xg, ei, bv)
        l_bin = binary_loss_fn(bin_logit, Yb)
        l_sf  = sf_loss_fn(sf_logit, Ys)
        loss = l_bin + SF_LOSS_WEIGHT * l_sf
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bs = X.size(0); n_seen += bs
        run_loss += loss.item() * bs; run_bin += l_bin.item() * bs; run_sf += l_sf.item() * bs
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    scheduler.step()

    model.eval()
    yt, yp, yp_score, gates_c, gates_g = [], [], [], [], []
    with torch.no_grad():
        for _, X, M, Yb, Ys, xg, ei, bv in loader_va:
            X=X.to(DEVICE); M=M.to(DEVICE); xg=xg.to(DEVICE); ei=ei.to(DEVICE); bv=bv.to(DEVICE)
            bin_logit, _, gate = model(X, M, xg, ei, bv)
            prob = torch.sigmoid(bin_logit).cpu().numpy()
            yp_score.extend(prob)
            yp.extend((prob > 0.5).astype(int))
            yt.extend(Yb.numpy())
            gates_c.extend(gate[:,0].cpu().numpy()); gates_g.extend(gate[:,1].cpu().numpy())
    yt = np.array(yt); yp = np.array(yp); yp_score = np.array(yp_score)
    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro", zero_division=0)
    pr, rc, _, _ = precision_recall_fscore_support(yt, yp, labels=[1], zero_division=0)
    try:    auroc = roc_auc_score(yt, yp_score)
    except: auroc = float("nan")
    auprc = average_precision_score(yt, yp_score)

    history["train_loss"].append(run_loss / max(1, n_seen))
    history["train_bin"].append(run_bin / max(1, n_seen))
    history["train_sf"].append(run_sf / max(1, n_seen))
    history["val_acc"].append(acc); history["val_f1_macro"].append(f1m)
    history["val_recall_pos"].append(float(rc[0])); history["val_precision_pos"].append(float(pr[0]))
    history["val_auroc"].append(auroc); history["val_auprc"].append(auprc)
    history["gate_cnn"].append(float(np.mean(gates_c))); history["gate_gnn"].append(float(np.mean(gates_g)))
    history["fold_used"].append(fold + 1)

    score = 0.5 * f1m + 0.5 * auprc
    print(f"Ep {ep:2d} fold {fold+1}: loss {history['train_loss'][-1]:.3f}  "
          f"acc {acc:.3f}  F1m {f1m:.3f}  rec+ {rc[0]:.3f}  prec+ {pr[0]:.3f}  "
          f"AUROC {auroc:.3f}  AUPRC {auprc:.3f}  gate C/G {np.mean(gates_c):.2f}/{np.mean(gates_g):.2f}")

    ckpt.maybe_save(score, ep, model, arch, history)
    if score > best_score + 1e-4:
        best_score = score; bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stop at epoch {ep}.")
            break

with open(SAVE_DIR / "history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"\\nDone. Best CV score: {best_score:.4f}")
"""

CELL_LOOP_THREE = """
# ============ Training Loop (rotating K-fold, 3-class) ============
SF_LOSS_WEIGHT = 0.5
history = {"train_loss": [], "train_cls": [], "train_sf": [],
           "val_acc": [], "val_f1_macro": [],
           "val_per_class_f1": [],
           "val_sf_acc": [], "val_sf_f1": [],
           "gate_cnn": [], "gate_gnn": [], "fold_used": []}
best_score, bad = -math.inf, 0

for ep in range(1, EPOCHS + 1):
    fold = (ep - 1) % N_FOLDS
    tr_idx, va_idx = fold_splits[fold]

    ds_train = HybridDataset([tv_h[i] for i in tr_idx], [tv_s[i] for i in tr_idx],
                             tv_top[tr_idx], tv_sf_y[tr_idx], [tv_kmer[i] for i in tr_idx])
    ds_val   = HybridDataset([tv_h[i] for i in va_idx], [tv_s[i] for i in va_idx],
                             tv_top[va_idx], tv_sf_y[va_idx], [tv_kmer[i] for i in va_idx])
    loader_tr = DataLoader(ds_train, BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_hybrid)
    loader_va = DataLoader(ds_val,   BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_hybrid)

    model.train()
    run_loss = run_cls = run_sf = 0.0; n_seen = 0
    pbar = tqdm(loader_tr, desc=f"Ep {ep}/{EPOCHS} fold {fold+1}", leave=False)
    for _, X, M, Yc, Ys, xg, ei, bv in pbar:
        X=X.to(DEVICE); M=M.to(DEVICE); Yc=Yc.to(DEVICE); Ys=Ys.to(DEVICE)
        xg=xg.to(DEVICE); ei=ei.to(DEVICE); bv=bv.to(DEVICE)
        cls_logit, sf_logit, gate = model(X, M, xg, ei, bv)
        l_cls = class_loss_fn(cls_logit, Yc)
        l_sf  = sf_loss_fn(sf_logit, Ys)
        loss = l_cls + SF_LOSS_WEIGHT * l_sf
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bs = X.size(0); n_seen += bs
        run_loss += loss.item() * bs; run_cls += l_cls.item() * bs; run_sf += l_sf.item() * bs
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    scheduler.step()

    model.eval()
    yt_c, yp_c, yt_s, yp_s, gates_c, gates_g = [], [], [], [], [], []
    with torch.no_grad():
        for _, X, M, Yc, Ys, xg, ei, bv in loader_va:
            X=X.to(DEVICE); M=M.to(DEVICE); xg=xg.to(DEVICE); ei=ei.to(DEVICE); bv=bv.to(DEVICE)
            cls_logit, sf_logit, gate = model(X, M, xg, ei, bv)
            yp_c.extend(cls_logit.argmax(-1).cpu().numpy()); yt_c.extend(Yc.numpy())
            yp_s.extend(sf_logit.argmax(-1).cpu().numpy()); yt_s.extend(Ys.numpy())
            gates_c.extend(gate[:,0].cpu().numpy()); gates_g.extend(gate[:,1].cpu().numpy())
    yt_c, yp_c = np.array(yt_c), np.array(yp_c)
    yt_s, yp_s = np.array(yt_s), np.array(yp_s)
    acc = accuracy_score(yt_c, yp_c)
    f1m = f1_score(yt_c, yp_c, average="macro", zero_division=0)
    per_cls_f1 = f1_score(yt_c, yp_c, average=None, labels=list(range(n_classes)), zero_division=0)
    sf_acc = accuracy_score(yt_s, yp_s)
    sf_f1  = f1_score(yt_s, yp_s, average="macro", zero_division=0)

    history["train_loss"].append(run_loss / max(1, n_seen))
    history["train_cls"].append(run_cls / max(1, n_seen))
    history["train_sf"].append(run_sf / max(1, n_seen))
    history["val_acc"].append(acc); history["val_f1_macro"].append(f1m)
    history["val_per_class_f1"].append(per_cls_f1.tolist())
    history["val_sf_acc"].append(sf_acc); history["val_sf_f1"].append(sf_f1)
    history["gate_cnn"].append(float(np.mean(gates_c))); history["gate_gnn"].append(float(np.mean(gates_g)))
    history["fold_used"].append(fold + 1)

    score = 0.5 * f1m + 0.5 * sf_f1
    print(f"Ep {ep:2d} fold {fold+1}: loss {history['train_loss'][-1]:.3f}  "
          f"top acc {acc:.3f} F1m {f1m:.3f} | per-cls {[f'{v:.3f}' for v in per_cls_f1]} | "
          f"sf acc {sf_acc:.3f} F1m {sf_f1:.3f} | gate C/G {np.mean(gates_c):.2f}/{np.mean(gates_g):.2f}")

    ckpt.maybe_save(score, ep, model, arch, history)
    if score > best_score + 1e-4:
        best_score = score; bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stop at epoch {ep}.")
            break

with open(SAVE_DIR / "history.json", "w") as f:
    # per_class_f1 nested list is JSON-safe
    json.dump(history, f, indent=2)
print(f"\\nDone. Best CV score: {best_score:.4f}")
"""

CELL_TEST_BINARY = """
# ============ Held-Out Test Evaluation ============
best_ckpt, best_ep = ckpt.get_best()
if best_ckpt is None:
    raise RuntimeError("No checkpoints saved.")
model.load_state_dict(best_ckpt["model_state_dict"]); model.to(DEVICE).eval()
print(f"Loaded best epoch {best_ep} (CV score={best_ckpt['score']:.4f})")
print(f"Top-5 saved: {ckpt.all_saved()}")

yt, yp_score, gates_c, gates_g = [], [], [], []
with torch.no_grad():
    for _, X, M, Yb, Ys, xg, ei, bv in loader_test:
        X=X.to(DEVICE); M=M.to(DEVICE); xg=xg.to(DEVICE); ei=ei.to(DEVICE); bv=bv.to(DEVICE)
        bin_logit, _, gate = model(X, M, xg, ei, bv)
        yp_score.extend(torch.sigmoid(bin_logit).cpu().numpy())
        yt.extend(Yb.numpy())
        gates_c.extend(gate[:,0].cpu().numpy()); gates_g.extend(gate[:,1].cpu().numpy())
yt = np.array(yt); yp_score = np.array(yp_score)
np.save(SAVE_DIR / "test_y_true.npy", yt)
np.save(SAVE_DIR / "test_y_score.npy", yp_score)
print(f"Test set: {len(yt):,} | pos rate: {yt.mean():.3f}")
print(f"Test gate CNN/GNN: {np.mean(gates_c):.3f} / {np.mean(gates_g):.3f}")
"""

CELL_TEST_THREE = """
# ============ Held-Out Test Evaluation ============
best_ckpt, best_ep = ckpt.get_best()
if best_ckpt is None: raise RuntimeError("No checkpoints saved.")
model.load_state_dict(best_ckpt["model_state_dict"]); model.to(DEVICE).eval()
print(f"Loaded best epoch {best_ep} (CV score={best_ckpt['score']:.4f})")
print(f"Top-5 saved: {ckpt.all_saved()}")

yt_c, yp_c, yp_proba, yt_s, yp_s, gates_c, gates_g = [], [], [], [], [], [], []
with torch.no_grad():
    for _, X, M, Yc, Ys, xg, ei, bv in loader_test:
        X=X.to(DEVICE); M=M.to(DEVICE); xg=xg.to(DEVICE); ei=ei.to(DEVICE); bv=bv.to(DEVICE)
        cls_logit, sf_logit, gate = model(X, M, xg, ei, bv)
        proba = F.softmax(cls_logit, dim=-1).cpu().numpy()
        yp_proba.append(proba)
        yp_c.extend(proba.argmax(-1)); yt_c.extend(Yc.numpy())
        yp_s.extend(sf_logit.argmax(-1).cpu().numpy()); yt_s.extend(Ys.numpy())
        gates_c.extend(gate[:,0].cpu().numpy()); gates_g.extend(gate[:,1].cpu().numpy())
yt_c, yp_c = np.array(yt_c), np.array(yp_c)
yp_proba = np.concatenate(yp_proba, axis=0)
yt_s, yp_s = np.array(yt_s), np.array(yp_s)
np.save(SAVE_DIR / "test_y_true_top.npy", yt_c)
np.save(SAVE_DIR / "test_y_pred_top.npy", yp_c)
np.save(SAVE_DIR / "test_y_proba_top.npy", yp_proba)
np.save(SAVE_DIR / "test_y_true_sf.npy", yt_s)
np.save(SAVE_DIR / "test_y_pred_sf.npy", yp_s)
print(f"Test set: {len(yt_c):,}")
print(f"Test gate CNN/GNN: {np.mean(gates_c):.3f} / {np.mean(gates_g):.3f}")
"""

# ---------------------------------------------------------------------------
# Variant-specific analysis cells
# ---------------------------------------------------------------------------

ANALYSIS_FOCAL = """
# ============ ANALYSIS: Focal Loss + Weighted Sampler ============
# Hypothesis: focal loss + balanced sampling lifts DNA recall under the natural
# ~8/92 prior without trashing precision.

# 1) Training curves
hist = history
ep = range(1, len(hist["train_loss"]) + 1)
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0,0].plot(ep, hist["train_loss"], label="loss"); ax[0,0].plot(ep, hist["train_bin"], label="binary")
ax[0,0].plot(ep, hist["train_sf"], label="sf"); ax[0,0].set_title("Training losses"); ax[0,0].legend()
ax[0,1].plot(ep, hist["val_recall_pos"], label="recall+")
ax[0,1].plot(ep, hist["val_precision_pos"], label="precision+")
ax[0,1].plot(ep, hist["val_f1_macro"], label="F1 macro"); ax[0,1].set_title("Validation"); ax[0,1].legend()
ax[1,0].plot(ep, hist["val_auroc"], label="AUROC"); ax[1,0].plot(ep, hist["val_auprc"], label="AUPRC")
ax[1,0].set_title("Probabilistic metrics"); ax[1,0].legend()
ax[1,1].plot(ep, hist["gate_cnn"], label="CNN"); ax[1,1].plot(ep, hist["gate_gnn"], label="GNN")
ax[1,1].set_title("Fusion gate weights"); ax[1,1].legend()
plt.tight_layout(); plt.savefig(SAVE_DIR / "training_curves.png", dpi=120); plt.show()

# 2) Confusion matrix at default threshold (0.5)
yp = (yp_score > 0.5).astype(int)
cm = confusion_matrix(yt, yp, labels=[0, 1])
print("Test confusion matrix (rows=true, cols=pred; 0=other, 1=DNA):")
print(cm)
print()
print(classification_report(yt, yp, target_names=["non-DNA", "DNA"], zero_division=0, digits=4))

# 3) PR / ROC curves
prec, rec, thr_pr = precision_recall_curve(yt, yp_score)
fpr, tpr, thr_roc = roc_curve(yt, yp_score)
auprc = average_precision_score(yt, yp_score)
auroc = roc_auc_score(yt, yp_score)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(rec, prec, label=f"AUPRC={auprc:.3f}"); ax[0].axhline(yt.mean(), ls="--", c="grey", label="prior")
ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision"); ax[0].set_title("PR Curve"); ax[0].legend()
ax[1].plot(fpr, tpr, label=f"AUROC={auroc:.3f}"); ax[1].plot([0,1],[0,1], "--", c="grey")
ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].set_title("ROC Curve"); ax[1].legend()
plt.tight_layout(); plt.savefig(SAVE_DIR / "pr_roc.png", dpi=120); plt.show()

# 4) Threshold sweep — does focal already pick a sensible threshold, or do we still need tuning?
thresholds = np.linspace(0.05, 0.95, 19)
rows = []
for t in thresholds:
    yp_t = (yp_score > t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yt, yp_t, labels=[1], zero_division=0)
    rows.append((t, float(p[0]), float(r[0]), float(f[0])))
import pandas as pd
df_thr = pd.DataFrame(rows, columns=["threshold", "precision+", "recall+", "F1+"])
print("\\nThreshold sweep (positive class):")
print(df_thr.to_string(index=False))
opt_idx = int(df_thr["F1+"].idxmax())
print(f"\\nOptimal-F1 threshold: {df_thr.iloc[opt_idx]['threshold']:.2f}  "
      f"-> P={df_thr.iloc[opt_idx]['precision+']:.3f}  R={df_thr.iloc[opt_idx]['recall+']:.3f}  "
      f"F1={df_thr.iloc[opt_idx]['F1+']:.3f}")
df_thr.to_csv(SAVE_DIR / "threshold_sweep.csv", index=False)

# 5) Calibration
from sklearn.calibration import calibration_curve
frac_pos, mean_pred = calibration_curve(yt, yp_score, n_bins=10, strategy="quantile")
plt.figure(figsize=(5,5)); plt.plot(mean_pred, frac_pos, "o-", label="model")
plt.plot([0,1],[0,1], "--", c="grey", label="perfect")
plt.xlabel("Mean predicted prob"); plt.ylabel("Fraction positive"); plt.title("Calibration (quantile)")
plt.legend(); plt.tight_layout(); plt.savefig(SAVE_DIR / "calibration.png", dpi=120); plt.show()

# 6) Train/val/test gap (using best CV epoch)
print(f"\\nTrain/Val/Test summary (best epoch {best_ep})")
print(f"  Final train loss : {hist['train_loss'][best_ep-1]:.4f}")
print(f"  Best val F1m     : {hist['val_f1_macro'][best_ep-1]:.4f}")
print(f"  Best val AUPRC   : {hist['val_auprc'][best_ep-1]:.4f}")
print(f"  Test  F1m@0.5    : {f1_score(yt, (yp_score>0.5).astype(int), average='macro', zero_division=0):.4f}")
print(f"  Test  AUPRC      : {auprc:.4f}")
print(f"  Test  AUROC      : {auroc:.4f}")
"""

ANALYSIS_THRESHOLD = """
# ============ ANALYSIS: BCE pos_weight + Post-hoc Threshold Tuning ============
# Hypothesis: with proper pos_weight, the model learns a well-calibrated score
# and the only thing missing is choosing the right operating point.

import pandas as pd
hist = history
ep = range(1, len(hist["train_loss"]) + 1)

# 1) Training curves
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0,0].plot(ep, hist["train_loss"], label="loss"); ax[0,0].plot(ep, hist["train_bin"], label="binary")
ax[0,0].plot(ep, hist["train_sf"], label="sf"); ax[0,0].set_title("Training losses"); ax[0,0].legend()
ax[0,1].plot(ep, hist["val_recall_pos"], label="recall+")
ax[0,1].plot(ep, hist["val_precision_pos"], label="precision+")
ax[0,1].plot(ep, hist["val_f1_macro"], label="F1 macro"); ax[0,1].set_title("Validation @0.5"); ax[0,1].legend()
ax[1,0].plot(ep, hist["val_auroc"], label="AUROC"); ax[1,0].plot(ep, hist["val_auprc"], label="AUPRC")
ax[1,0].set_title("Probabilistic metrics"); ax[1,0].legend()
ax[1,1].plot(ep, hist["gate_cnn"], label="CNN"); ax[1,1].plot(ep, hist["gate_gnn"], label="GNN")
ax[1,1].set_title("Fusion gate weights"); ax[1,1].legend()
plt.tight_layout(); plt.savefig(SAVE_DIR / "training_curves.png", dpi=120); plt.show()

# 2) Tune threshold on held-out 10% slice of TRAIN-VAL (NOT on test!) — fair tuning
# Reuse fold 0's val split for calibration set:
cal_idx = fold_splits[0][1]
ds_cal = HybridDataset([tv_h[i] for i in cal_idx], [tv_s[i] for i in cal_idx],
                       tv_bin[cal_idx], tv_sf_y[cal_idx], [tv_kmer[i] for i in cal_idx])
loader_cal = DataLoader(ds_cal, BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_hybrid)
model.eval()
yt_c, yp_c = [], []
with torch.no_grad():
    for _, X, M, Yb, Ys, xg, ei, bv in loader_cal:
        X=X.to(DEVICE); M=M.to(DEVICE); xg=xg.to(DEVICE); ei=ei.to(DEVICE); bv=bv.to(DEVICE)
        bin_logit, _, _ = model(X, M, xg, ei, bv)
        yp_c.extend(torch.sigmoid(bin_logit).cpu().numpy()); yt_c.extend(Yb.numpy())
yt_c = np.array(yt_c); yp_c = np.array(yp_c)

prec, rec, thr_pr = precision_recall_curve(yt_c, yp_c)
# precision_recall_curve returns thr of length len(prec)-1
f1_curve = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
opt_thr = float(thr_pr[int(np.argmax(f1_curve))])
print(f"\\n>>> Optimal threshold from calibration fold: {opt_thr:.4f}")
print(f"    Calibration F1+ at this threshold: {f1_curve.max():.4f}")

# 3) Test set: report at BOTH t=0.5 (default) AND t=opt_thr
print("\\n--- TEST @ default 0.5 ---")
yp_default = (yp_score > 0.5).astype(int)
print(classification_report(yt, yp_default, target_names=["non-DNA", "DNA"], zero_division=0, digits=4))
print("Confusion:"); print(confusion_matrix(yt, yp_default, labels=[0,1]))

print(f"\\n--- TEST @ tuned {opt_thr:.4f} ---")
yp_tuned = (yp_score > opt_thr).astype(int)
print(classification_report(yt, yp_tuned, target_names=["non-DNA", "DNA"], zero_division=0, digits=4))
print("Confusion:"); print(confusion_matrix(yt, yp_tuned, labels=[0,1]))

# 4) PR / ROC + full threshold sweep on TEST
prec_t, rec_t, thr_t = precision_recall_curve(yt, yp_score)
fpr, tpr, _ = roc_curve(yt, yp_score)
auprc = average_precision_score(yt, yp_score); auroc = roc_auc_score(yt, yp_score)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(rec_t, prec_t, label=f"AUPRC={auprc:.3f}"); ax[0].axhline(yt.mean(), ls="--", c="grey", label="prior")
ax[0].axvline(rec_t[int(np.argmax(2*prec_t[:-1]*rec_t[:-1]/(prec_t[:-1]+rec_t[:-1]+1e-12)))],
              ls=":", c="red", label="opt-F1")
ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision"); ax[0].set_title("PR Curve (TEST)"); ax[0].legend()
ax[1].plot(fpr, tpr, label=f"AUROC={auroc:.3f}"); ax[1].plot([0,1],[0,1], "--", c="grey")
ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].set_title("ROC Curve (TEST)"); ax[1].legend()
plt.tight_layout(); plt.savefig(SAVE_DIR / "pr_roc.png", dpi=120); plt.show()

# 5) Threshold-vs-metric table (TEST)
ts = np.linspace(0.05, 0.95, 19)
rows = []
for t in ts:
    yp_t = (yp_score > t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yt, yp_t, labels=[1], zero_division=0)
    rows.append((t, float(p[0]), float(r[0]), float(f[0])))
df_thr = pd.DataFrame(rows, columns=["threshold", "precision+", "recall+", "F1+"])
print("\\nTEST threshold sweep:")
print(df_thr.to_string(index=False))
df_thr.to_csv(SAVE_DIR / "test_threshold_sweep.csv", index=False)
print(f"\\nFinal answer: tuning gave +{f1_score(yt, yp_tuned, average='macro') - f1_score(yt, yp_default, average='macro'):+.4f} macro-F1 over default 0.5")
"""

ANALYSIS_THREE = """
# ============ ANALYSIS: 3-class DNA/LTR/LINE (no SF cap) ============
# Hypothesis: removing the max_per_sf cap (a) makes evaluation reflect natural prior,
# (b) does not crash DNA recall thanks to inverse-sqrt class weights.

import pandas as pd
hist = history
ep_axis = range(1, len(hist["train_loss"]) + 1)
per_cls = np.array(hist["val_per_class_f1"])

# 1) Training curves
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0,0].plot(ep_axis, hist["train_loss"], label="loss")
ax[0,0].plot(ep_axis, hist["train_cls"], label="class")
ax[0,0].plot(ep_axis, hist["train_sf"], label="sf"); ax[0,0].set_title("Training losses"); ax[0,0].legend()
ax[0,1].plot(ep_axis, hist["val_acc"], label="acc"); ax[0,1].plot(ep_axis, hist["val_f1_macro"], label="F1 macro")
ax[0,1].set_title("Top-level validation"); ax[0,1].legend()
for ci, cn in enumerate(KEEP_CLASSES):
    ax[1,0].plot(ep_axis, per_cls[:, ci], label=cn)
ax[1,0].set_title("Per-class F1 (validation)"); ax[1,0].legend()
ax[1,1].plot(ep_axis, hist["gate_cnn"], label="CNN"); ax[1,1].plot(ep_axis, hist["gate_gnn"], label="GNN")
ax[1,1].set_title("Fusion gate weights"); ax[1,1].legend()
plt.tight_layout(); plt.savefig(SAVE_DIR / "training_curves.png", dpi=120); plt.show()

# 2) Test-set top-level classification report
print("--- TOP-LEVEL (DNA / LTR / LINE) ---")
print(classification_report(yt_c, yp_c, target_names=KEEP_CLASSES, zero_division=0, digits=4))
cm = confusion_matrix(yt_c, yp_c, labels=list(range(n_classes)))
print("Confusion (rows=true, cols=pred):"); print(cm)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
ax.set_xticklabels(KEEP_CLASSES); ax.set_yticklabels(KEEP_CLASSES)
ax.set_xlabel("Pred"); ax.set_ylabel("True"); ax.set_title("Top-level confusion (TEST)")
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black")
plt.colorbar(im); plt.tight_layout(); plt.savefig(SAVE_DIR / "confmat_top.png", dpi=120); plt.show()

# 3) Superfamily classification report (top-N most populous)
print("\\n--- SUPERFAMILY ---")
print(classification_report(yt_s, yp_s, target_names=sf_names, zero_division=0, digits=3))

# Per-SF F1 sorted, with sample counts
sf_f1_arr = f1_score(yt_s, yp_s, average=None, labels=list(range(n_sf)), zero_division=0)
sf_support = np.bincount(yt_s, minlength=n_sf)
df_sf = pd.DataFrame({"superfamily": sf_names, "support": sf_support, "F1": sf_f1_arr})
df_sf = df_sf.sort_values("support", ascending=False)
print("\\nPer-superfamily F1 (sorted by support):")
print(df_sf.to_string(index=False))
df_sf.to_csv(SAVE_DIR / "per_sf_f1.csv", index=False)

# 4) Train/val/test gap
print(f"\\nTrain/Val/Test summary (best epoch {best_ep})")
print(f"  Final train loss : {hist['train_loss'][best_ep-1]:.4f}")
print(f"  Best val F1m     : {hist['val_f1_macro'][best_ep-1]:.4f}")
print(f"  Best val SF F1m  : {hist['val_sf_f1'][best_ep-1]:.4f}")
print(f"  Test  top F1m    : {f1_score(yt_c, yp_c, average='macro', zero_division=0):.4f}")
print(f"  Test  top acc    : {accuracy_score(yt_c, yp_c):.4f}")
print(f"  Test  SF  F1m    : {f1_score(yt_s, yp_s, average='macro', zero_division=0):.4f}")
print(f"  Test  SF  acc    : {accuracy_score(yt_s, yp_s):.4f}")

# 5) Comparison hook: print v4.3-balanced reference (filled in by hand from prior runs)
print("\\nReference (v4.3 with MAX_PER_SF=3000): top F1m ~0.78 / SF F1m ~0.70 (varies by run)")
print("Compare those numbers to the 'Test' lines above to see whether the SF cap was helping or hurting.")
"""

# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------

def build(out: Path, nb_tag: str, title_md: str, intro_md: str, *,
          variant: str, epochs: int = 30, use_sampler: bool = False) -> None:
    """variant in {'focal', 'bce', 'three'}"""
    cells: list[dict] = []
    cells.append(md(title_md))
    cells.append(md(intro_md))
    cells.append(code(CELL_IMPORTS))
    cells.append(code(CELL_CONFIG_BASE.replace('__NB_TAG__', nb_tag).replace('__EPOCHS__', str(epochs))))
    cells.append(code(CELL_DATA_HELPERS))
    cells.append(code(CELL_KMER))
    cells.append(code(CELL_DATASET))
    cells.append(code(CELL_CNN_TOWER))
    cells.append(code(CELL_GNN_FUSION))
    if variant == "three":
        cells.append(code(CELL_MODEL_THREE))
    else:
        cells.append(code(CELL_MODEL_BINARY))

    if variant == "focal":
        cells.append(code(CELL_LOSS_FOCAL))
        cells.append(code(CELL_DATA_PREP_BINARY))
        cells.append(code(CELL_CKPT))
        cells.append(code(CELL_TRAIN_BINARY_FOCAL))
        cells.append(code(CELL_LOOP_BINARY.replace('__USE_SAMPLER__', 'True')))
        cells.append(code(CELL_TEST_BINARY))
        cells.append(md("## Analysis"))
        cells.append(code(ANALYSIS_FOCAL))

    elif variant == "bce":
        cells.append(code(CELL_LOSS_BCE))
        cells.append(code(CELL_DATA_PREP_BINARY))
        cells.append(code(CELL_CKPT))
        cells.append(code(CELL_TRAIN_BINARY_BCE))
        cells.append(code(CELL_LOOP_BINARY.replace('__USE_SAMPLER__', 'False')))
        cells.append(code(CELL_TEST_BINARY))
        cells.append(md("## Analysis"))
        cells.append(code(ANALYSIS_THRESHOLD))

    elif variant == "three":
        cells.append(code(CELL_LOSS_CE))
        cells.append(code(CELL_DATA_PREP_THREE))
        cells.append(code(CELL_CKPT))
        cells.append(code(CELL_TRAIN_THREE))
        cells.append(code(CELL_LOOP_THREE))
        cells.append(code(CELL_TEST_THREE))
        cells.append(md("## Analysis"))
        cells.append(code(ANALYSIS_THREE))
    else:
        raise ValueError(variant)

    out.write_text(json.dumps(notebook(cells), indent=1))


HERE = Path(__file__).parent

build(
    HERE / "binary_dna_natural_focal.ipynb",
    nb_tag="binary_focal",
    title_md="# Binary DNA classification under the natural prior — Focal Loss + Weighted Sampler",
    intro_md="""
        **Question.** When we stop subsampling superfamilies and train on the natural
        ~8/92 (DNA / non-DNA) distribution, can a focal loss combined with a
        `WeightedRandomSampler` recover acceptable DNA recall *without* trashing
        precision?

        **What we change vs the v4.3 baseline.**
        - No `max_per_sf` cap → preserves natural class imbalance.
        - Single binary head (DNA vs non-DNA) instead of 3-class top head.
        - Loss = `BinaryFocalLoss(α_pos=0.75, γ=2)` instead of weighted CE.
        - `WeightedRandomSampler` with `1/sqrt(class_count)` per-sample weights.
        - Auxiliary SF loss kept (weighted CE with label smoothing) to retain
          fine-grained gradients.

        **What we keep.** Architecture, optimizer, scheduler, 5-fold rotating CV,
        held-out 20 % test set, top-5 checkpointing, gate-weight logging.
    """,
    variant="focal",
    epochs=30,
    use_sampler=True,
)

build(
    HERE / "binary_dna_natural_threshold_tuned.ipynb",
    nb_tag="binary_bce_threshold",
    title_md="# Binary DNA classification under the natural prior — BCE + post-hoc Threshold Tuning",
    intro_md="""
        **Question.** Can we get the same DNA-recall lift from *just* a properly weighted
        BCE loss plus post-hoc threshold tuning — without focal loss or weighted sampling?
        If so, the simpler recipe wins.

        **What we change vs the v4.3 baseline.**
        - No `max_per_sf` cap → natural prior.
        - Single binary head (DNA vs non-DNA).
        - Loss = `BCEWithLogitsLoss(pos_weight = neg/pos)`; data shuffled naturally.
        - **No** focal, **no** WeightedRandomSampler.
        - Threshold tuned on a held-out calibration fold (NOT the test set), then
          test set re-evaluated at both 0.5 and the tuned threshold.
        - Auxiliary SF loss kept.
    """,
    variant="bce",
    epochs=30,
)

build(
    HERE / "three_class_natural_weighted.ipynb",
    nb_tag="three_class_natural",
    title_md="# 3-class top-level (DNA/LTR/LINE) under the natural prior — weighted CE",
    intro_md="""
        **Question.** The v4.3 trunk caps every superfamily at `MAX_PER_SF=3000`,
        which throws away most LTR/LINE evidence and (more importantly) makes the
        evaluation distribution unrepresentative of real-world TE annotation. If we
        remove the cap and rely on inverse-sqrt class weights, does the model still
        learn each top-level class — and how does superfamily F1 hold up?

        **What we change vs the v4.3 baseline.**
        - `MAX_PER_SF = None` → use all sequences passing the genome-exclude / SF-min
          filters.
        - Top-level loss = label-smoothed CE with `compute_class_weights(mode="inv_sqrt")`.
        - Auxiliary SF head also trained with weighted label-smoothed CE.
        - Everything else (architecture, optimizer, CV) matches v4.3.
    """,
    variant="three",
    epochs=30,
)

print("Wrote 3 notebooks to", HERE)
