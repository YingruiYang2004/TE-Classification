#!/usr/bin/env python3
"""Hybrid TE Classifier V5 (Unified multi-class) - cluster-ready entrypoint.

Converted from vgp_features_tpase_multiclass_v5.ipynb and refactored for SLURM.
**Architecture matches the notebook exactly.**

Usage:
  python train_hybrid_v5.py -c config_hybrid_v5.yaml

Notes
- Paths in YAML are interpreted relative to the current working directory.
  The recommended SLURM script `cd`s into the script directory (~/TEs/models).
- This script saves top-K checkpoints (default K=5) into `save_dir`.
"""

from __future__ import annotations

import argparse
import gc
import heapq
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ----------------------------
# Device utilities
# ----------------------------

def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Return the best available accelerator."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------
# Global defaults (overridden by YAML at runtime)
# ----------------------------
FIXED_LENGTH = 5000
DNA_TIR_PREFIXES: Set[str] = {
    "DNA/hAT", "DNA/TcMar", "DNA/PIF", "DNA/PiggyBac", "DNA/Academ",
    "DNA/CMC", "DNA/Sola", "DNA/Kolobok", "DNA/P", "DNA/MULE",
    "DNA/Crypton", "DNA/Merlin", "DNA/Ginger", "DNA/Dada", "DNA"
}


# ----------------------------
# FASTA loading and label handling
# ----------------------------

def read_fasta_with_labels(path: str) -> Tuple[List[str], List[str], List[str]]:
    """Read FASTA and parse TE class from headers.

    Header format: >name#classification (e.g., >hAT_1-aAnoBae#DNA/hAT)

    Returns:
      headers: without '>'
      sequences: uppercase strings
      te_labels: strings after '#', or 'Unknown'
    """
    headers, sequences, te_labels = [], [], []
    h: Optional[str] = None
    buf: List[str] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):

                # flush previous record
                if h is not None:
                    seq = "".join(buf).upper()
                    sequences.append(seq)
                    headers.append(h)

                    if "#" in h:
                        te_labels.append(h.split("#")[-1])
                    else:
                        te_labels.append("Unknown")

                h = line[1:]
                buf = []
            else:
                buf.append(line)

    # last record
    if h is not None:
        seq = "".join(buf).upper()
        sequences.append(seq)
        headers.append(h)
        if "#" in h:
            te_labels.append(h.split("#")[-1])
        else:
            te_labels.append("Unknown")

    return headers, sequences, te_labels


def is_dna_tir(label: str, prefixes: Set[str] = DNA_TIR_PREFIXES) -> bool:
    """Heuristic: treat label as DNA/TIR if it matches a known prefix."""
    for prefix in prefixes:
        if label == prefix or label.startswith(prefix + "-") or label.startswith(prefix + "/"):
            return True
    return False


def compute_class_weights(
    y_ids: np.ndarray,
    n_classes: int,
    mode: str = "inv_sqrt",
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute multi-class weights for imbalanced training."""
    counts = np.bincount(np.asarray(y_ids, dtype=np.int64), minlength=n_classes).astype(np.float64)
    if mode == "none":
        w = np.ones(n_classes, dtype=np.float32)
    elif mode == "inv":
        w = (1.0 / (counts + eps)).astype(np.float32)
    elif mode == "inv_sqrt":
        w = (1.0 / np.sqrt(counts + eps)).astype(np.float32)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    w = w / (w.mean() + eps)
    return w


# ----------------------------
# K-mer Feature Extraction (matches notebook exactly)
# ----------------------------

# ASCII -> {0,1,2,3,4} for A,C,G,T,other
_ASCII_MAP = np.full(256, 4, dtype=np.uint8)
for ch, val in [("A", 0), ("C", 1), ("G", 2), ("T", 3), ("a", 0), ("c", 1), ("g", 2), ("t", 3)]:
    _ASCII_MAP[ord(ch)] = val

_COMP = np.array([3, 2, 1, 0], dtype=np.uint8)  # A<->T, C<->G


def kmer_code_forward(arr4: np.ndarray) -> int:
    code = 0
    for v in arr4:
        code = (code << 2) | int(v)
    return code


def kmer_code_rc(arr4: np.ndarray) -> int:
    code = 0
    for v in arr4[::-1]:
        code = (code << 2) | int(_COMP[v])
    return code


def canonical_kmer_code(arr4: np.ndarray) -> int:
    c1 = kmer_code_forward(arr4)
    c2 = kmer_code_rc(arr4)
    return c1 if c1 < c2 else c2


def hash_u32(x: int, dim: int) -> int:
    z = (x * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z ^= (z >> 33)
    z = (z * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF
    z ^= (z >> 29)
    return int(z % dim)


@dataclass
class KmerWindowFeaturizer:
    """Extract k-mer frequency features from sliding windows."""
    k: int = 7
    dim: int = 2048
    window: int = 512
    stride: int = 256
    add_pos: bool = True
    l2_normalize: bool = True

    def featurize_sequence(self, seq: str) -> Tuple[np.ndarray, np.ndarray]:
        arr = _ASCII_MAP[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
        L = int(arr.size)
        
        if L == 0:
            X = np.zeros((1, self.dim + (1 if self.add_pos else 0)), dtype=np.float32)
            return X, np.array([0], dtype=np.int64)

        if L <= self.window:
            starts = np.array([0], dtype=np.int64)
        else:
            starts = np.arange(0, L - self.window + 1, self.stride, dtype=np.int64)
            if starts.size == 0:
                starts = np.array([0], dtype=np.int64)

        out_dim = self.dim + (1 if self.add_pos else 0)
        X = np.zeros((starts.size, out_dim), dtype=np.float32)

        for wi, st in enumerate(starts):
            en = min(st + self.window, L)
            sub = arr[st:en]
            counts = np.zeros(self.dim, dtype=np.float32)
            total = 0

            k = self.k
            if sub.size >= k:
                for i in range(0, sub.size - k + 1):
                    kmer = sub[i:i + k]
                    if np.any(kmer == 4):
                        continue
                    code = canonical_kmer_code(kmer)
                    j = hash_u32(code, self.dim)
                    counts[j] += 1.0
                    total += 1

            if total > 0:
                counts /= float(total)

            if self.l2_normalize:
                nrm = np.linalg.norm(counts)
                if nrm > 0:
                    counts /= nrm

            if self.add_pos:
                center = (st + en) / 2.0
                pos = center / max(1.0, float(L))
                X[wi, :-1] = counts
                X[wi, -1] = pos
            else:
                X[wi, :] = counts

        return X, starts


def build_chain_edge_index(n: int, undirected: bool = True, self_loops: bool = True) -> torch.Tensor:
    """Build edge index for a chain graph (windows connected sequentially)."""
    edges = []
    if n > 1:
        src = np.arange(n - 1, dtype=np.int64)
        dst = np.arange(1, n, dtype=np.int64)
        edges.append((src, dst))
        if undirected:
            edges.append((dst, src))
    if self_loops:
        idx = np.arange(n, dtype=np.int64)
        edges.append((idx, idx))
    if not edges:
        ei = np.zeros((2, 0), dtype=np.int64)
    else:
        s = np.concatenate([e[0] for e in edges])
        d = np.concatenate([e[1] for e in edges])
        ei = np.stack([s, d], axis=0)
    return torch.from_numpy(ei)


# ----------------------------
# CNN encoding + dataset (matches notebook exactly)
# ----------------------------

ENCODE = np.full(256, 4, dtype=np.int64)
for ch, idx in zip(b"ACGTNacgtn", [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]):
    ENCODE[ch] = idx

# Reverse complement: ACGTN -> TGCAN -> indices [3, 2, 1, 0, 4]
REV_COMP = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)


class UnifiedDataset(Dataset):
    """
    Dataset for unified TE classification (V5).
    
    Each sample has:
    - CNN: One-hot encoding placed randomly in fixed-length canvas
    - GNN: Pre-computed k-mer window features
    - class_label: Unified TE type (DNA/hAT, LTR/Gypsy, etc.)
    - is_dna_tir: Binary flag (1 if DNA/TIR, 0 otherwise)
    """
    def __init__(
        self,
        headers: List[str],
        sequences: List[str],
        class_labels: np.ndarray,
        is_dna_tir: np.ndarray,
        kmer_features: List[np.ndarray],
        fixed_length: int = FIXED_LENGTH
    ):
        self.headers = list(headers)
        self.sequences = list(sequences)
        self.class_labels = np.asarray(class_labels, dtype=np.int64)
        self.is_dna_tir = np.asarray(is_dna_tir, dtype=np.int64)
        self.kmer_features = kmer_features
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = len(seq)

        # Encode sequence for CNN
        seq_bytes = seq.encode("ascii", "ignore")
        seq_idx = ENCODE[np.frombuffer(seq_bytes, dtype=np.uint8)]

        # Random placement in canvas
        max_start = max(0, self.fixed_length - seq_len)
        start_pos = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end_pos = start_pos + seq_len

        # K-mer features (pre-computed)
        kmer_feat = self.kmer_features[idx]

        return (
            self.headers[idx],
            seq_idx,
            int(self.class_labels[idx]),
            int(self.is_dna_tir[idx]),
            start_pos,
            end_pos,
            seq_len,
            kmer_feat
        )


def collate_unified(batch, fixed_length=FIXED_LENGTH):
    """
    Collate function for unified model.
    
    Returns:
        headers: list of header strings
        X_cnn: (B, 5, fixed_length) one-hot for CNN
        mask: (B, fixed_length) padding mask
        Y_class: (B,) unified class labels
        Y_binary: (B,) is_dna_tir labels
        x_gnn: (total_nodes, feat_dim) stacked node features
        edge_index: (2, total_edges) graph edges
        batch_vec: (total_nodes,) batch assignment
    """
    (headers, seq_idxs, class_labels, is_dna_tir,
     starts, ends, lengths, kmer_feats) = zip(*batch)

    B = len(batch)
    
    # ---- CNN inputs ----
    X_cnn = torch.zeros((B, 5, fixed_length), dtype=torch.float32)
    mask = torch.zeros((B, fixed_length), dtype=torch.bool)

    for i, (seq_idx, start, end, seq_len) in enumerate(zip(seq_idxs, starts, ends, lengths)):
        actual_len = min(seq_len, fixed_length - start)
        if actual_len > 0:
            idx = torch.from_numpy(seq_idx[:actual_len].astype(np.int64))
            pos = torch.arange(actual_len, dtype=torch.long) + start
            X_cnn[i, idx, pos] = 1.0
            mask[i, start:start + actual_len] = (idx != 4)

    Y_class = torch.tensor(class_labels, dtype=torch.long)
    Y_binary = torch.tensor(is_dna_tir, dtype=torch.long)

    # ---- GNN inputs ----
    xs, eis, batch_vecs = [], [], []
    node_offset = 0

    for gi, kmer_feat in enumerate(kmer_feats):
        x = torch.from_numpy(kmer_feat).to(torch.float32)
        n = x.size(0)
        ei = build_chain_edge_index(n, undirected=True, self_loops=True)
        
        xs.append(x)
        eis.append(ei + node_offset)
        batch_vecs.append(torch.full((n,), gi, dtype=torch.int64))
        node_offset += n

    x_gnn = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.int64)
    batch_vec = torch.cat(batch_vecs, dim=0)

    return (
        list(headers), X_cnn, mask, Y_class, Y_binary,
        x_gnn, edge_index, batch_vec
    )


# ----------------------------
# CNN Building Blocks (matches notebook exactly)
# ----------------------------

class ConvBlock(nn.Module):
    """Residual convolutional block with optional dilation."""
    def __init__(self, c_in, c_out, kernel_size=9, dilation=1, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation, bias=True)
        self.bn = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)

    def forward(self, x):
        y = self.conv(x)
        y = F.gelu(self.bn(y))
        y = self.drop(y)
        return y + self.proj(x)


class MaskedMaxPool1d(nn.Module):
    """Max pooling that respects padding mask."""
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, mask):
        if mask is not None:
            m = mask.unsqueeze(1).float()
            x = x * m + (~mask.unsqueeze(1)) * (-1e9)
        x_p = F.max_pool1d(x, self.kernel_size, self.stride)
        if mask is None:
            return x_p, None
        m_p = F.max_pool1d(mask.float().unsqueeze(1), self.kernel_size, self.stride).squeeze(1) > 0
        return x_p, m_p


def masked_avg_pool(z, mask):
    """Global average pooling respecting mask."""
    if mask is None:
        return z.mean(-1)
    m = mask.unsqueeze(1).float()
    return (z * m).sum(-1) / m.sum(-1).clamp_min(1.0)


class RCFirstConv1d(nn.Module):
    """RC-invariant first convolution (early fusion)."""
    def __init__(self, out_channels, kernel_size=15, dilation=1, bias=True, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(5, out_channels, kernel_size, padding=pad, dilation=dilation, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y1 = self.conv(x)
        x_rc = x.flip(-1).index_select(1, REV_COMP.to(x.device))
        y2 = self.conv(x_rc).flip(-1)
        y = torch.max(y1, y2)
        y = self.batch_norm(y)
        y = F.gelu(y)
        y = self.dropout(y)
        return y


class CNNTower(nn.Module):
    """CNN tower for sequence motif detection."""
    def __init__(
        self,
        width: int = 128,
        motif_kernels: Tuple[int, ...] = (7, 15, 21),
        context_kernel: int = 9,
        context_dilations: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.15,
        rc_mode: str = "late"
    ):
        super().__init__()
        self.rc_mode = rc_mode
        self.out_dim = width
        
        if rc_mode == "early":
            self.motif_convs = nn.ModuleList([
                RCFirstConv1d(width, kernel_size=k, dropout=dropout)
                for k in motif_kernels
            ])
        else:
            self.motif_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(5, width, kernel_size=k, padding=k // 2, bias=True),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
                for k in motif_kernels
            ])
        
        in_ch = width * len(motif_kernels)
        self.mix = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=1, bias=True),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.context_blocks = nn.ModuleList([
            ConvBlock(width, width, kernel_size=context_kernel, dilation=d, dropout=dropout)
            for d in context_dilations
        ])
        self.pool = MaskedMaxPool1d(kernel_size=2, stride=2)

    @staticmethod
    def rc_transform(x, mask):
        x_rc = x.index_select(1, REV_COMP.to(x.device)).flip(-1)
        mask_rc = None if mask is None else mask.flip(-1)
        return x_rc, mask_rc

    def encode(self, x, mask):
        feats = [conv(x) for conv in self.motif_convs]
        z = torch.cat(feats, dim=1)
        z = self.mix(z)
        
        m = mask
        for block in self.context_blocks:
            z = block(z)
            z, m = self.pool(z, m)
        
        return masked_avg_pool(z, m)

    def forward(self, x, mask):
        if self.rc_mode == "late":
            f = self.encode(x, mask)
            x_rc, mask_rc = self.rc_transform(x, mask)
            r = self.encode(x_rc, mask_rc)
            return 0.5 * (f + r)
        else:
            return self.encode(x, mask)


# ----------------------------
# GNN Building Blocks (matches notebook exactly)
# ----------------------------

def scatter_mean(x: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter mean for graph pooling."""
    out = torch.zeros((dim_size, x.size(1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, idx, x)
    cnt = torch.bincount(idx, minlength=dim_size).clamp_min(1).to(x.device).to(x.dtype).unsqueeze(1)
    return out / cnt


class GraphSAGELayer(nn.Module):
    """GraphSAGE-style message passing layer."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.bincount(dst, minlength=x.size(0)).clamp_min(1).to(x.device).to(x.dtype).unsqueeze(1)
        agg = agg / deg
        h = self.lin_self(x) + self.lin_neigh(agg)
        h = F.relu(h)
        return self.dropout(h)


class GNNTower(nn.Module):
    """GNN tower for k-mer composition analysis."""
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.out_dim = hidden
        
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(GraphSAGELayer(d, hidden, dropout=dropout))
            d = hidden
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        
        B = int(batch_vec.max().item()) + 1 if batch_vec.numel() else 0
        return scatter_mean(x, batch_vec, dim_size=B)


# ----------------------------
# Cross-Modal Attention Fusion (matches notebook exactly)
# ----------------------------

class CrossModalAttentionFusion(nn.Module):
    """Fuses CNN and GNN embeddings using cross-modal attention."""
    def __init__(
        self,
        cnn_dim: int = 128,
        gnn_dim: int = 128,
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
        
        self.ln1 = nn.LayerNorm(fusion_dim)
        self.ln2 = nn.LayerNorm(fusion_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, cnn_embed: torch.Tensor, gnn_embed: torch.Tensor):
        c = self.cnn_proj(cnn_embed)
        g = self.gnn_proj(gnn_embed)
        
        c = self.ln1(c)
        g = self.ln2(g)
        
        combined = torch.stack([c, g], dim=1)
        attn_out, _ = self.cross_attn(combined, combined, combined)
        
        c_attn = attn_out[:, 0]
        g_attn = attn_out[:, 1]
        
        gate_input = torch.cat([c_attn, g_attn], dim=-1)
        gate_weights = self.gate(gate_input)
        
        fused = gate_weights[:, 0:1] * c_attn + gate_weights[:, 1:2] * g_attn
        fused = self.out_proj(fused)
        
        return fused, gate_weights


# ----------------------------
# Hybrid TE Classifier V5: Unified Classification (matches notebook exactly)
# ----------------------------

class HybridTEClassifierV5(nn.Module):
    """
    Hybrid TE Classifier V5 with unified multi-class classification.
    
    Key differences from V4:
    1. Single classification head for ALL TE types (DNA/hAT, LTR/Gypsy, LINE/L1, etc.)
    2. Binary classification (DNA/TIR vs others) derived from multi-class predictions
    3. Model learns complete TE taxonomy structure
    
    Architecture:
    - CNN Tower: Sequence motifs with RC-invariance
    - GNN Tower: K-mer compositional patterns
    - Cross-modal Attention Fusion
    - Single unified classification head
    """
    def __init__(
        self,
        num_classes: int,
        class_to_is_dna_tir: torch.Tensor,  # Maps class_id -> 1 if DNA/TIR, 0 otherwise
        # CNN params
        cnn_width: int = 128,
        motif_kernels: Tuple[int, ...] = (7, 15, 21),
        context_dilations: Tuple[int, ...] = (1, 2, 4, 8),
        rc_mode: str = "late",
        # GNN params
        gnn_in_dim: int = 2049,
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        # Fusion params
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.15
    ):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("class_to_is_dna_tir", class_to_is_dna_tir)
        
        # ---- CNN Tower ----
        self.cnn_tower = CNNTower(
            width=cnn_width,
            motif_kernels=motif_kernels,
            context_dilations=context_dilations,
            dropout=dropout,
            rc_mode=rc_mode
        )
        
        # ---- GNN Tower ----
        self.gnn_tower = GNNTower(
            in_dim=gnn_in_dim,
            hidden=gnn_hidden,
            n_layers=gnn_layers,
            dropout=dropout
        )
        
        # ---- Attention Fusion ----
        self.fusion = CrossModalAttentionFusion(
            cnn_dim=cnn_width,
            gnn_dim=gnn_hidden,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ---- Single Unified Classification Head ----
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(
        self,
        x_cnn: torch.Tensor,
        mask: torch.Tensor,
        x_gnn: torch.Tensor,
        edge_index: torch.Tensor,
        batch_vec: torch.Tensor
    ):
        """
        Forward pass.
        
        Returns:
            class_logits: (B, num_classes) unified classification logits
            binary_probs: (B,) probability of being DNA/TIR (derived from class logits)
            gate_weights: (B, 2) fusion weights [CNN, GNN]
        """
        # Get embeddings from both towers
        cnn_embed = self.cnn_tower(x_cnn, mask)
        gnn_embed = self.gnn_tower(x_gnn, edge_index, batch_vec)
        
        # Fuse with attention
        fused, gate_weights = self.fusion(cnn_embed, gnn_embed)
        
        # Unified classification
        class_logits = self.classifier(fused)
        
        # Derive binary probability: sum of probabilities for DNA/TIR classes
        class_probs = F.softmax(class_logits, dim=-1)  # (B, num_classes)
        binary_probs = (class_probs * self.class_to_is_dna_tir.float()).sum(dim=-1)  # (B,)
        
        return class_logits, binary_probs, gate_weights
    
    def predict_with_binary(self, class_logits: torch.Tensor):
        """
        Get both multi-class and binary predictions from logits.
        
        Returns:
            class_pred: (B,) predicted class indices
            binary_pred: (B,) binary predictions (1 if DNA/TIR, 0 otherwise)
            binary_probs: (B,) probability of DNA/TIR
        """
        class_pred = class_logits.argmax(dim=-1)
        class_probs = F.softmax(class_logits, dim=-1)
        binary_probs = (class_probs * self.class_to_is_dna_tir.float()).sum(dim=-1)
        binary_pred = (binary_probs > 0.5).long()
        return class_pred, binary_pred, binary_probs


# ----------------------------
# Loss helpers
# ----------------------------

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = float(smoothing)
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        if self.weight is None:
            loss = (-true_dist * log_probs).sum(dim=1).mean()
        else:
            w = self.weight[target].to(logits.device)
            loss = ((-true_dist * log_probs).sum(dim=1) * w).mean()

        return loss


# ----------------------------
# Training function: Top-K checkpoints
# ----------------------------

class TopKCheckpointManager:
    """Keep the top-K checkpoints by validation score."""

    def __init__(self, save_dir: str, prefix: str, k: int = 5):
        import heapq

        self.heapq = heapq
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.k = int(k)
        self.heap: List[Tuple[float, int]] = []  # min-heap of (score, epoch)
        self.paths: Dict[int, Path] = {}  # epoch -> path

    def _path(self, epoch: int) -> Path:
        return self.save_dir / f"{self.prefix}_epoch{epoch}.pt"

    def maybe_save(
        self,
        score: float,
        epoch: int,
        model: nn.Module,
        arch_config: dict,
        class_names: list,
        class_to_id: dict,
        class_is_dna_tir: dict,
        history: dict,
    ) -> None:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "arch_config": arch_config,
            "class_names": class_names,
            "class_to_id": class_to_id,
            "class_is_dna_tir": class_is_dna_tir,
            "history": dict(history),
            "epoch": epoch,
            "score": float(score),
        }
        path = self._path(epoch)

        # if not yet K, save
        if len(self.heap) < self.k:
            torch.save(ckpt, path)
            self.heapq.heappush(self.heap, (float(score), int(epoch)))
            self.paths[int(epoch)] = path
            return

        # check if better than worst
        worst_score, worst_epoch = self.heap[0]
        if score <= worst_score:
            return

        # replace worst
        self.heapq.heapreplace(self.heap, (float(score), int(epoch)))
        # delete old checkpoint
        old_path = self.paths.get(int(worst_epoch))
        if old_path is not None and old_path.exists():
            try:
                old_path.unlink()
            except OSError:
                pass

        torch.save(ckpt, path)
        self.paths[int(epoch)] = path

    def get_best(self) -> Optional[Tuple[float, int, Path]]:
        if not self.heap:
            return None
        best = max(self.heap, key=lambda t: t[0])
        score, epoch = best
        return score, epoch, self.paths[epoch]

    def get_all_saved_epochs(self) -> List[Tuple[float, int]]:
        return sorted(self.heap, key=lambda t: t[0], reverse=True)


# ----------------------------
# Training pipeline (refactored)
# ----------------------------

def run_train_v5(
    fasta_path: str,
    *,
    # Training params
    batch_size: int = 16,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 10,
    # Dataloader params
    num_workers: int = 0,
    pin_memory: bool = True,
    # CNN params
    fixed_length: int = 5000,
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
    label_smoothing: float = 0.1,
    # Data params
    min_class_count: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    max_samples_per_class: Optional[int] = None,
    # Other
    device: Optional[str] = None,
    save_dir: str = ".",
    ckpt_prefix: str = "hybrid_v5",
    topk: int = 5,
):
    """Train the Hybrid V5 model and return a results dict."""
    import time

    global FIXED_LENGTH
    FIXED_LENGTH = int(fixed_length)

    start_time = time.time()
    dev = resolve_device(device)
    print(f"Using device: {dev}")
    print("\n" + "=" * 60)
    print("HYBRID TE CLASSIFIER V5: UNIFIED MULTI-CLASS")
    print("=" * 60)

    print("\n=== Loading data ===")
    headers, sequences, te_labels = read_fasta_with_labels(fasta_path)
    print(f"Loaded {len(headers)} sequences")

    # filter classes
    te_counts = Counter(te_labels)
    keep_classes = {t for t, c in te_counts.items() if c >= min_class_count}
    print(f"Classes with >= {min_class_count} samples: {len(keep_classes)}")

    filtered_h, filtered_s, filtered_labels = [], [], []
    for h, s, t in zip(headers, sequences, te_labels):
        if t in keep_classes:
            filtered_h.append(h)
            filtered_s.append(s)
            filtered_labels.append(t)

    # optional cap per class
    if max_samples_per_class is not None:
        rng = np.random.default_rng(random_state)
        per_class: Dict[str, List[int]] = {}
        for i, t in enumerate(filtered_labels):
            per_class.setdefault(t, []).append(i)

        idx_keep = []
        for t, idxs in per_class.items():
            if len(idxs) > max_samples_per_class:
                idx_keep.extend(rng.choice(idxs, size=max_samples_per_class, replace=False).tolist())
            else:
                idx_keep.extend(idxs)
        idx_keep = sorted(idx_keep)
        filtered_h = [filtered_h[i] for i in idx_keep]
        filtered_s = [filtered_s[i] for i in idx_keep]
        filtered_labels = [filtered_labels[i] for i in idx_keep]
        print(f"After max_samples_per_class={max_samples_per_class}: {len(filtered_h)} sequences")

    # class mapping
    class_names = sorted(set(filtered_labels))
    class_to_id = {c: i for i, c in enumerate(class_names)}
    n_classes = len(class_names)

    # binary DNA/TIR label derived from class tag
    all_is_dna_tir = np.array([int(is_dna_tir(lbl)) for lbl in filtered_labels], dtype=np.int64)

    all_class_ids = np.array([class_to_id[lbl] for lbl in filtered_labels], dtype=np.int64)

    class_is_dna_tir = {c: bool(is_dna_tir(c)) for c in class_names}

    # k-mer features (using KmerWindowFeaturizer like notebook)
    print("\n=== Building k-mer features ===")
    featurizer = KmerWindowFeaturizer(
        k=kmer_k, dim=kmer_dim, window=kmer_window, stride=kmer_stride,
        add_pos=True, l2_normalize=True
    )
    all_kmer_features = []
    for s in tqdm(filtered_s, desc="k-mer", mininterval=1.0):
        X, _ = featurizer.featurize_sequence(s)
        all_kmer_features.append(X)

    # split
    idx = np.arange(len(filtered_h))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=all_class_ids
    )

    train_h = [filtered_h[i] for i in idx_train]
    train_s = [filtered_s[i] for i in idx_train]
    train_class = all_class_ids[idx_train]
    train_is_dna = all_is_dna_tir[idx_train]
    train_kmer = [all_kmer_features[i] for i in idx_train]

    test_h = [filtered_h[i] for i in idx_test]
    test_s = [filtered_s[i] for i in idx_test]
    test_class = all_class_ids[idx_test]
    test_is_dna = all_is_dna_tir[idx_test]
    test_kmer = [all_kmer_features[i] for i in idx_test]

    print(f"\nTrain: {len(train_h)}, Test: {len(test_h)}")

    # free memory
    del headers, sequences, te_labels
    del filtered_h, filtered_s, filtered_labels, all_class_ids, all_is_dna_tir
    gc.collect()

    return _run_train_v5_trainloop(
        train_h, train_s, train_class, train_is_dna, train_kmer,
        test_h, test_s, test_class, test_is_dna, test_kmer,
        n_classes, class_names, class_to_id, class_is_dna_tir,
        batch_size, epochs, lr, patience,
        num_workers, pin_memory,
        cnn_width, motif_kernels, context_dilations, rc_mode,
        kmer_dim, gnn_hidden, gnn_layers,
        fusion_dim, num_heads, dropout, label_smoothing,
        dev, save_dir, ckpt_prefix, topk, start_time,
    )


def _run_train_v5_trainloop(
    train_h, train_s, train_class, train_is_dna, train_kmer,
    test_h, test_s, test_class, test_is_dna, test_kmer,
    n_classes, class_names, class_to_id, class_is_dna_tir,
    batch_size, epochs, lr, patience,
    num_workers, pin_memory,
    cnn_width, motif_kernels, context_dilations, rc_mode,
    kmer_dim, gnn_hidden, gnn_layers,
    fusion_dim, num_heads, dropout, label_smoothing,
    device: torch.device, save_dir: str, ckpt_prefix: str, topk: int, start_time: float,
):
    # datasets
    ds_train = UnifiedDataset(train_h, train_s, train_class, train_is_dna, train_kmer, fixed_length=FIXED_LENGTH)
    ds_test = UnifiedDataset(test_h, test_s, test_class, test_is_dna, test_kmer, fixed_length=FIXED_LENGTH)

    use_pin = bool(pin_memory) and (device.type == "cuda")
    persistent = (num_workers > 0)

    # Create collate function with fixed_length
    def collate_fn(batch):
        return collate_unified(batch, fixed_length=FIXED_LENGTH)

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_fn,
        pin_memory=use_pin,
        persistent_workers=persistent,
    )
    loader_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_fn,
        pin_memory=use_pin,
        persistent_workers=persistent,
    )

    # Build class_to_is_dna_tir tensor for model
    class_is_dna_tir_tensor = torch.tensor(
        [int(class_is_dna_tir[c]) for c in class_names],
        dtype=torch.float32
    )

    # model (matches notebook interface exactly)
    model = HybridTEClassifierV5(
        num_classes=n_classes,
        class_to_is_dna_tir=class_is_dna_tir_tensor,
        cnn_width=cnn_width,
        motif_kernels=motif_kernels,
        context_dilations=context_dilations,
        rc_mode=rc_mode,
        gnn_in_dim=kmer_dim + 1,  # +1 for position feature
        gnn_hidden=gnn_hidden,
        gnn_layers=gnn_layers,
        fusion_dim=fusion_dim,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    arch_config = {
        "cnn_width": cnn_width,
        "motif_kernels": tuple(motif_kernels),
        "context_dilations": tuple(context_dilations),
        "rc_mode": rc_mode,
        "gnn_in_dim": kmer_dim + 1,
        "gnn_hidden": gnn_hidden,
        "gnn_layers": gnn_layers,
        "fusion_dim": fusion_dim,
        "num_heads": num_heads,
        "num_classes": n_classes,
        "fixed_length": FIXED_LENGTH,
    }

    ckpt_manager = TopKCheckpointManager(save_dir, prefix=ckpt_prefix, k=topk)

    class_weights = compute_class_weights(train_class, n_classes, mode="inv_sqrt")
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = LabelSmoothingCE(smoothing=label_smoothing, weight=class_weights_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=max(1, epochs))

    history = {
        "train_loss": [],
        "val_class_acc": [], "val_class_f1": [],
        "val_binary_acc": [], "val_binary_f1": [],
        "gate_weights_cnn": [], "gate_weights_gnn": []
    }

    best_score = -1e9
    best_epoch = -1
    bad_epochs = 0

    print("\n=== Training ===")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader_train, desc=f"Epoch {epoch}/{epochs}", mininterval=1.0, leave=False)
        for batch in pbar:
            # Unpack tuple (matches notebook collate_unified output)
            headers, X_cnn, mask, Y_class, Y_binary, x_gnn, edge_index, batch_vec = batch
            
            X_cnn = X_cnn.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            Y_class = Y_class.to(device, non_blocking=True)
            x_gnn = x_gnn.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            batch_vec = batch_vec.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            
            class_logits, binary_probs, gate_weights = model(
                X_cnn, mask, x_gnn, edge_index, batch_vec
            )
            
            loss = criterion(class_logits, Y_class)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_loss += loss.item() * X_cnn.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / len(ds_train)

        # ---- Evaluate ----
        model.eval()
        all_class_pred, all_class_true = [], []
        all_binary_pred, all_binary_true = [], []
        all_gate_cnn, all_gate_gnn = [], []

        with torch.no_grad():
            for batch in loader_test:
                headers, X_cnn, mask, Y_class, Y_binary, x_gnn, edge_index, batch_vec = batch
                
                X_cnn = X_cnn.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                x_gnn = x_gnn.to(device, non_blocking=True)
                edge_index = edge_index.to(device, non_blocking=True)
                batch_vec = batch_vec.to(device, non_blocking=True)

                class_logits, binary_probs, gate_weights = model(
                    X_cnn, mask, x_gnn, edge_index, batch_vec
                )
                
                class_pred, binary_pred, _ = model.predict_with_binary(class_logits)
                
                all_class_pred.extend(class_pred.cpu().numpy())
                all_class_true.extend(Y_class.numpy())
                all_binary_pred.extend(binary_pred.cpu().numpy())
                all_binary_true.extend(Y_binary.numpy())
                
                all_gate_cnn.extend(gate_weights[:, 0].cpu().numpy())
                all_gate_gnn.extend(gate_weights[:, 1].cpu().numpy())

        all_class_pred = np.array(all_class_pred)
        all_class_true = np.array(all_class_true)
        all_binary_pred = np.array(all_binary_pred)
        all_binary_true = np.array(all_binary_true)

        class_acc = accuracy_score(all_class_true, all_class_pred)
        class_f1 = f1_score(all_class_true, all_class_pred, average="macro", zero_division=0)
        binary_acc = accuracy_score(all_binary_true, all_binary_pred)
        binary_f1 = f1_score(all_binary_true, all_binary_pred, average="binary", zero_division=0)

        avg_gate_cnn = float(np.mean(all_gate_cnn))
        avg_gate_gnn = float(np.mean(all_gate_gnn))

        history["train_loss"].append(train_loss)
        history["val_class_acc"].append(class_acc)
        history["val_class_f1"].append(class_f1)
        history["val_binary_acc"].append(binary_acc)
        history["val_binary_f1"].append(binary_f1)
        history["gate_weights_cnn"].append(avg_gate_cnn)
        history["gate_weights_gnn"].append(avg_gate_gnn)

        # Combined score: emphasize multi-class (matches notebook)
        combined_score = 0.7 * class_f1 + 0.3 * binary_f1

        print(f"Ep {epoch:2d}: loss {train_loss:.4f} | class acc {class_acc:.4f} F1 {class_f1:.4f} | "
              f"binary acc {binary_acc:.4f} F1 {binary_f1:.4f} | gate CNN:{avg_gate_cnn:.2f} GNN:{avg_gate_gnn:.2f}")

        # Save checkpoint if in top-k
        ckpt_manager.maybe_save(
            score=combined_score,
            epoch=epoch,
            model=model,
            arch_config=arch_config,
            class_names=class_names,
            class_to_id=class_to_id,
            class_is_dna_tir=class_is_dna_tir,
            history=history,
        )

        # Early stopping
        if combined_score > best_score + 1e-4:
            best_score = combined_score
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    return _run_train_v5_finaleval(
        model=model,
        ckpt_manager=ckpt_manager,
        loader_test=loader_test,
        ds_test=ds_test,
        n_classes=n_classes,
        class_names=class_names,
        class_to_id=class_to_id,
        class_is_dna_tir=class_is_dna_tir,
        history=history,
        device=device,
        save_dir=save_dir,
        start_time=start_time,
    )


def _run_train_v5_finaleval(
    *,
    model: nn.Module,
    ckpt_manager: TopKCheckpointManager,
    loader_test: DataLoader,
    ds_test: UnifiedDataset,
    n_classes: int,
    class_names: list,
    class_to_id: dict,
    class_is_dna_tir: dict,
    history: dict,
    device: torch.device,
    save_dir: str,
    start_time: float,
) -> dict:
    import time

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    best = ckpt_manager.get_best()
    if best is not None:
        best_score, best_epoch, best_path = best
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        print(f"Loaded best checkpoint epoch {best_epoch} score {best_score:.4f} from {best_path}")

    # evaluation
    model.eval()
    all_class_pred, all_class_true = [], []
    all_binary_pred, all_binary_true = [], []
    
    with torch.no_grad():
        for batch in loader_test:
            headers, X_cnn, mask, Y_class, Y_binary, x_gnn, edge_index, batch_vec = batch
            
            X_cnn = X_cnn.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_gnn = x_gnn.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            batch_vec = batch_vec.to(device, non_blocking=True)

            class_logits, binary_probs, gate_weights = model(
                X_cnn, mask, x_gnn, edge_index, batch_vec
            )
            
            class_pred, binary_pred, _ = model.predict_with_binary(class_logits)
            
            all_class_pred.extend(class_pred.cpu().numpy())
            all_class_true.extend(Y_class.numpy())
            all_binary_pred.extend(binary_pred.cpu().numpy())
            all_binary_true.extend(Y_binary.numpy())

    all_class_pred = np.array(all_class_pred)
    all_class_true = np.array(all_class_true)
    all_binary_pred = np.array(all_binary_pred)
    all_binary_true = np.array(all_binary_true)

    # Multi-class metrics
    class_acc = accuracy_score(all_class_true, all_class_pred)
    class_bal = balanced_accuracy_score(all_class_true, all_class_pred)
    class_f1m = f1_score(all_class_true, all_class_pred, average="macro", zero_division=0)
    class_f1w = f1_score(all_class_true, all_class_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(all_class_true, all_class_pred)
    
    # Binary metrics
    binary_acc = accuracy_score(all_binary_true, all_binary_pred)
    binary_f1 = f1_score(all_binary_true, all_binary_pred, average="binary", zero_division=0)

    report = classification_report(all_class_true, all_class_pred, target_names=class_names, digits=4, zero_division=0)

    elapsed = time.time() - start_time

    # Save a small text report
    save_dir_p = Path(save_dir)
    save_dir_p.mkdir(parents=True, exist_ok=True)
    (save_dir_p / "final_report.txt").write_text(
        f"=== Multi-class Classification ===\n"
        f"Accuracy: {class_acc:.4f}\nBalanced Accuracy: {class_bal:.4f}\n"
        f"F1 macro: {class_f1m:.4f}\nF1 weighted: {class_f1w:.4f}\n\n"
        f"=== Binary (DNA-TIR) Classification ===\n"
        f"Accuracy: {binary_acc:.4f}\nF1: {binary_f1:.4f}\n\n"
        f"Saved checkpoints: {ckpt_manager.get_all_saved_epochs()}\n\n"
        f"Classification report:\n{report}\n"
    )

    np.save(save_dir_p / "confusion_matrix.npy", cm)

    print(f"\n=== Multi-class ===")
    print(f"Accuracy: {class_acc:.4f} | Balanced: {class_bal:.4f} | F1(macro): {class_f1m:.4f} | F1(weighted): {class_f1w:.4f}")
    print(f"\n=== Binary (DNA-TIR) ===")
    print(f"Accuracy: {binary_acc:.4f} | F1: {binary_f1:.4f}")
    print(f"\nSaved: {(save_dir_p / 'final_report.txt')}")
    print(f"Elapsed: {elapsed/60:.2f} min")

    return {
        "class_acc": float(class_acc),
        "class_bal_acc": float(class_bal),
        "class_f1_macro": float(class_f1m),
        "class_f1_weighted": float(class_f1w),
        "binary_acc": float(binary_acc),
        "binary_f1": float(binary_f1),
        "history": history,
        "best": ckpt_manager.get_best(),
        "saved_epochs": ckpt_manager.get_all_saved_epochs(),
        "save_dir": str(save_dir_p),
        "elapsed_sec": float(elapsed),
    }


# ----------------------------
# CLI / YAML
# ----------------------------

def _as_tuple(x) -> Tuple[int, ...]:
    if x is None:
        return tuple()
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    raise TypeError(f"Expected list/tuple for tuple field, got {type(x)}")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    paths = cfg.get("paths", {})
    model = cfg.get("model", {})
    train = cfg.get("train", {})
    runtime = cfg.get("runtime", {})
    output = cfg.get("output", {})

    fasta_path = paths["fasta_path"]

    fixed_length = int(model.get("fixed_length", FIXED_LENGTH))
    cnn_width = int(model.get("cnn_width", 128))
    motif_kernels = _as_tuple(model.get("motif_kernels", [7, 15, 21]))
    context_dilations = _as_tuple(model.get("context_dilations", [1, 2, 4, 8]))
    rc_mode = str(model.get("rc_fusion_mode", "late"))

    kmer_cfg = model.get("kmer", {})
    kmer_k = int(kmer_cfg.get("k", 7))
    kmer_dim = int(kmer_cfg.get("dim", 2048))
    kmer_window = int(kmer_cfg.get("window", 512))
    kmer_stride = int(kmer_cfg.get("stride", 256))
    gnn_hidden = int(kmer_cfg.get("gnn_hidden", 128))
    gnn_layers = int(kmer_cfg.get("gnn_layers", 3))

    fusion_cfg = model.get("fusion", {})
    fusion_dim = int(fusion_cfg.get("fusion_dim", 256))
    num_heads = int(fusion_cfg.get("num_heads", 4))

    dropout = float(model.get("dropout", 0.15))

    batch_size = int(train.get("batch_size", 16))
    epochs = int(train.get("epochs", 30))
    lr = float(train.get("lr", 1e-3))
    patience = int(train.get("patience", 10))
    label_smoothing = float(train.get("label_smoothing", 0.1))
    min_class_count = int(train.get("min_class_count", 50))
    test_size = float(train.get("test_size", 0.2))
    random_state = int(train.get("random_state", 42))
    max_samples_per_class = train.get("max_samples_per_class", None)
    if max_samples_per_class is not None:
        max_samples_per_class = int(max_samples_per_class)

    num_workers = int(runtime.get("num_workers", 0))
    pin_memory = bool(runtime.get("pin_memory", True))
    device = runtime.get("device", None)

    save_dir = str(output.get("save_dir", "./outputs_v5"))
    ckpt_prefix = str(output.get("ckpt_prefix", "hybrid_v5"))
    topk = int(output.get("topk", 5))

    # Minor H100-friendly setting (safe no-op on older torch)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    run_train_v5(
        fasta_path=fasta_path,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        num_workers=num_workers,
        pin_memory=pin_memory,
        fixed_length=fixed_length,
        cnn_width=cnn_width,
        motif_kernels=motif_kernels,
        context_dilations=context_dilations,
        rc_mode=rc_mode,
        kmer_k=kmer_k,
        kmer_dim=kmer_dim,
        kmer_window=kmer_window,
        kmer_stride=kmer_stride,
        gnn_hidden=gnn_hidden,
        gnn_layers=gnn_layers,
        fusion_dim=fusion_dim,
        num_heads=num_heads,
        dropout=dropout,
        label_smoothing=label_smoothing,
        min_class_count=min_class_count,
        test_size=test_size,
        random_state=random_state,
        max_samples_per_class=max_samples_per_class,
        device=device,
        save_dir=save_dir,
        ckpt_prefix=ckpt_prefix,
        topk=topk,
    )


if __name__ == "__main__":
    main()
