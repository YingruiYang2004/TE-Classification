#!/usr/bin/env python3
"""
Hierarchical TE Classification Model Training Script (V3)
===========================================================

This script trains a hierarchical CNN model for transposable element classification:
1. Binary classification: transposase+ vs None
2. Multi-class classification: superfamily (only for transposase+ sequences)

Usage:
    python train_hierarchical_v3.py -c config_hierarchical_v3.yaml
    python train_hierarchical_v3.py -f data/vgp/all_vgp_tes.fa -l data/vgp/features-tpase -o checkpoints

Author: Alex Yang
Date: 2026-01-09
"""

import os
import gc
import sys
import math
import argparse
import time
from pathlib import Path
from collections import Counter

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

# Try to import tqdm, fall back to simple range if not available
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ============================================================
# Configuration
# ============================================================

# Default values (can be overridden by config file or CLI args)
DEFAULTS = {
    "fixed_length": 40000,
    "min_class_count": 100,
    "width": 128,
    "motif_kernels": [7, 15, 21],
    "context_kernel": 9,
    "context_dilations": [1, 2, 4, 8],
    "dropout": 0.15,
    "rc_mode": "late",
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "patience": 10,
    "binary_weight": 1.0,
    "superfamily_weight": 1.0,
    "aux_weight": 0.1,
    "label_smoothing": 0.1,
    "test_size": 0.2,
    "random_state": 42,
    "subsample_none": 20000,
    "num_workers": 0,
}

# ============================================================
# Encoding
# ============================================================

# Mapping ACGT to 0-3, N to 4
ENCODE = np.full(256, 4, dtype=np.int64)
for ch, idx in zip(b"ACGTNacgtn", [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]):
    ENCODE[ch] = idx

# Reverse complement: ACGTN -> TGCAN -> indices [3, 2, 1, 0, 4]
REV_COMP = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)


# ============================================================
# Data Loading Functions
# ============================================================

def read_fasta(path):
    """Read FASTA file and return headers and sequences."""
    headers, sequences = [], []
    h, buf = None, []
    
    with open(path, 'r') as f:
        for line in f:
            if not line:
                continue
            if line[0] == '>':
                if h is not None:
                    sequences.append(''.join(buf).upper())
                    buf = []
                h = line[1:].strip()
                headers.append(h)
            else:
                buf.append(line.strip())
        if h is not None:
            sequences.append(''.join(buf).upper())
            
    return headers, sequences


def load_hierarchical_labels(label_path):
    """
    Load labels for hierarchical classification.
    
    Returns:
        label_dict: header -> tag (original tag string)
        binary_dict: header -> 0 (None) or 1 (has transposase)
    """
    label_path = Path(label_path)
    label_dict = {}
    binary_dict = {}
    
    superfamilies = Counter()
    
    with label_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            header = parts[0].lstrip('>')
            tag = parts[1]
            
            label_dict[header] = tag
            binary_dict[header] = 0 if tag == "None" else 1
            superfamilies[tag] += 1
    
    print(f"Loaded {len(label_dict)} sequences for hierarchical classification")
    print(f"\nSuperfamily distribution:")
    for tag, count in sorted(superfamilies.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(label_dict)
        has_tpase = "✗" if tag == "None" else "✓"
        print(f"  {has_tpase} {tag}: {count} ({pct:.1f}%)")
    
    n_pos = sum(1 for v in binary_dict.values() if v == 1)
    n_neg = len(binary_dict) - n_pos
    print(f"\nBinary split: {n_pos} transposase+ ({100*n_pos/len(binary_dict):.1f}%) | "
          f"{n_neg} None ({100*n_neg/len(binary_dict):.1f}%)")
    
    return label_dict, binary_dict


def compute_class_weights(y_ids, n_classes, mode="inv_sqrt", eps=1e-6):
    """Compute class weights for imbalanced multi-class."""
    counts = np.bincount(np.asarray(y_ids, dtype=np.int64), minlength=n_classes).astype(np.float64)
    if mode == "none":
        w = np.ones(n_classes, dtype=np.float32)
    elif mode == "inv":
        w = 1.0 / (counts + eps)
    elif mode == "inv_sqrt":
        w = 1.0 / np.sqrt(counts + eps)
    else:
        raise ValueError(f"Unknown mode={mode}")
    w = w / (w.mean() + 1e-12)
    return w.astype(np.float32)


# ============================================================
# Dataset and DataLoader
# ============================================================

class SeqDatasetHierarchical(Dataset):
    """Dataset for hierarchical TE classification."""
    
    def __init__(self, headers, sequences, binary_labels, class_labels, fixed_length):
        self.headers = list(headers)
        self.sequences = list(sequences)
        self.binary_labels = np.asarray(binary_labels, dtype=np.int64)
        self.class_labels = np.asarray(class_labels, dtype=np.int64)
        self.fixed_length = fixed_length
        self.seq_lengths = np.array([len(s) for s in sequences], dtype=np.int64)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = len(seq)
        
        # Encode sequence
        seq_bytes = seq.encode("ascii", "ignore")
        seq_idx = ENCODE[np.frombuffer(seq_bytes, dtype=np.uint8)]
        
        # Random placement
        max_start = max(0, self.fixed_length - seq_len)
        if max_start > 0:
            start_pos = np.random.randint(0, max_start + 1)
        else:
            start_pos = 0
        
        end_pos = start_pos + seq_len
        
        return (
            self.headers[idx],
            seq_idx,
            int(self.binary_labels[idx]),
            int(self.class_labels[idx]),
            start_pos,
            end_pos,
            seq_len
        )


def collate_hierarchical(batch, fixed_length):
    """Collate function for hierarchical classification."""
    headers, seq_idxs, binary_labels, class_labels, starts, ends, lengths = zip(*batch)
    
    B = len(batch)
    X = torch.zeros((B, 5, fixed_length), dtype=torch.float32)
    mask = torch.zeros((B, fixed_length), dtype=torch.bool)
    
    for i, (seq_idx, start, end, seq_len) in enumerate(zip(seq_idxs, starts, ends, lengths)):
        actual_len = min(seq_len, fixed_length - start)
        if actual_len > 0:
            idx = torch.from_numpy(seq_idx[:actual_len].astype(np.int64))
            pos = torch.arange(actual_len, dtype=torch.long) + start
            X[i, idx, pos] = 1.0
            mask[i, start:start + actual_len] = (idx != 4)
    
    Y_binary = torch.tensor(binary_labels, dtype=torch.long)
    Y_class = torch.tensor(class_labels, dtype=torch.long)
    
    starts_norm = torch.tensor(starts, dtype=torch.float32) / fixed_length
    ends_norm = torch.tensor(ends, dtype=torch.float32) / fixed_length
    lengths_norm = torch.tensor(lengths, dtype=torch.float32) / fixed_length
    
    return list(headers), X, mask, Y_binary, Y_class, starts_norm, ends_norm, lengths_norm


# ============================================================
# Network Building Blocks
# ============================================================

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
    """RC-invariant first convolution layer."""
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


# ============================================================
# Hierarchical Model
# ============================================================

class HierarchicalRCCNN(nn.Module):
    """
    Hierarchical RC-invariant CNN for TE classification.
    
    Two-task model:
    1. Binary head: transposase+ vs None
    2. Multi-class head: superfamily classification
    """
    def __init__(
        self,
        num_superfamilies,
        width=128,
        motif_kernels=(7, 15, 21),
        context_kernel=9,
        context_dilations=(1, 2, 4, 8),
        dropout=0.15,
        rc_mode="late"
    ):
        super().__init__()
        self.num_superfamilies = int(num_superfamilies)
        self.rc_mode = rc_mode
        
        # Motif detection layers
        if rc_mode == "early":
            self.motif_convs = nn.ModuleList([
                RCFirstConv1d(width, kernel_size=k, dropout=dropout)
                for k in motif_kernels
            ])
        else:
            self.motif_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(5, width, kernel_size=k, padding=k//2, bias=True),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
                for k in motif_kernels
            ])
        
        # Mix layer
        in_ch = width * len(motif_kernels)
        self.mix = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=1, bias=True),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Context blocks
        self.context_blocks = nn.ModuleList([
            ConvBlock(width, width, kernel_size=context_kernel, dilation=d, dropout=dropout)
            for d in context_dilations
        ])
        self.pool = MaskedMaxPool1d(kernel_size=2, stride=2)
        
        # Binary head
        self.binary_head = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        
        # Multi-class head
        self.superfamily_head = nn.Sequential(
            nn.Linear(width, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_superfamilies),
        )
        
        # Auxiliary boundary prediction
        self.boundary_head = nn.Sequential(
            nn.Linear(width, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
        
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
            pooled = 0.5 * (f + r)
        else:
            pooled = self.encode(x, mask)
        
        binary_logits = self.binary_head(pooled)
        superfamily_logits = self.superfamily_head(pooled)
        boundary_pred = self.boundary_head(pooled)
        
        return binary_logits, superfamily_logits, boundary_pred


# ============================================================
# Loss Functions
# ============================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=-1)
        if self.weight is not None:
            log_prob = log_prob * self.weight.unsqueeze(0)
        loss = -(smooth_one_hot * log_prob).sum(dim=-1)
        return loss.mean()


# ============================================================
# Utility Functions
# ============================================================

def resolve_device(requested=None):
    """Return the best available accelerator."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(state, path, filename):
    """Save model checkpoint with versioning."""
    os.makedirs(path, exist_ok=True)
    base_path = os.path.join(path, f"{filename}.pt")
    
    if os.path.exists(base_path):
        n = 1
        while n <= 100:
            versioned_path = os.path.join(path, f"{filename}_v{n}.pt")
            if not os.path.exists(versioned_path):
                torch.save(state, versioned_path)
                print(f"Saved checkpoint to {versioned_path}")
                return versioned_path
            n += 1
    else:
        torch.save(state, base_path)
        print(f"Saved checkpoint to {base_path}")
        return base_path


# ============================================================
# Training Function
# ============================================================

def run_train_v3(config):
    """
    Train hierarchical TE classifier (v3).
    
    Args:
        config: dict with all configuration options
    
    Returns:
        dict with model, history, and evaluation results
    """
    # Extract config
    fasta_path = config["fasta_path"]
    label_path = config["label_path"]
    save_dir = config.get("save_dir", "checkpoints")
    
    fixed_length = config.get("fixed_length", DEFAULTS["fixed_length"])
    min_class_count = config.get("min_class_count", DEFAULTS["min_class_count"])
    
    batch_size = config.get("batch_size", DEFAULTS["batch_size"])
    epochs = config.get("epochs", DEFAULTS["epochs"])
    lr = config.get("learning_rate", DEFAULTS["learning_rate"])
    patience = config.get("patience", DEFAULTS["patience"])
    
    width = config.get("width", DEFAULTS["width"])
    motif_kernels = tuple(config.get("motif_kernels", DEFAULTS["motif_kernels"]))
    context_kernel = config.get("context_kernel", DEFAULTS["context_kernel"])
    context_dilations = tuple(config.get("context_dilations", DEFAULTS["context_dilations"]))
    dropout = config.get("dropout", DEFAULTS["dropout"])
    rc_mode = config.get("rc_mode", DEFAULTS["rc_mode"])
    
    binary_weight = config.get("binary_weight", DEFAULTS["binary_weight"])
    superfamily_weight = config.get("superfamily_weight", DEFAULTS["superfamily_weight"])
    aux_weight = config.get("aux_weight", DEFAULTS["aux_weight"])
    label_smoothing = config.get("label_smoothing", DEFAULTS["label_smoothing"])
    
    test_size = config.get("test_size", DEFAULTS["test_size"])
    random_state = config.get("random_state", DEFAULTS["random_state"])
    subsample_none = config.get("subsample_none", DEFAULTS["subsample_none"])
    num_workers = config.get("num_workers", DEFAULTS["num_workers"])
    
    device_str = config.get("device", None)
    device = resolve_device(device_str)
    
    print("=" * 60)
    print("Hierarchical TE Classification Training (V3)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"RC fusion mode: {rc_mode}")
    print(f"Canvas size: {fixed_length} bp")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    
    # ---- Load data ----
    print("\n=== Loading data ===")
    headers, sequences = read_fasta(fasta_path)
    label_dict, binary_dict = load_hierarchical_labels(label_path)
    
    # Match headers to labels
    all_h, all_s, all_tags, all_binary = [], [], [], []
    for h, s in zip(headers, sequences):
        if h not in label_dict:
            continue
        all_h.append(h)
        all_s.append(s)
        all_tags.append(label_dict[h])
        all_binary.append(binary_dict[h])
    
    del headers, sequences
    gc.collect()
    
    print(f"\nMatched {len(all_h)} sequences")
    
    # Subsample None sequences
    if subsample_none is not None:
        none_indices = [i for i, b in enumerate(all_binary) if b == 0]
        tpase_indices = [i for i, b in enumerate(all_binary) if b == 1]
        
        if len(none_indices) > subsample_none:
            np.random.seed(random_state)
            sampled_none = np.random.choice(none_indices, subsample_none, replace=False)
            keep_indices = list(tpase_indices) + list(sampled_none)
            keep_indices.sort()
            
            all_h = [all_h[i] for i in keep_indices]
            all_s = [all_s[i] for i in keep_indices]
            all_tags = [all_tags[i] for i in keep_indices]
            all_binary = [all_binary[i] for i in keep_indices]
            
            print(f"Subsampled None sequences: {len(none_indices)} -> {subsample_none}")
            print(f"Total after subsampling: {len(all_h)}")
    
    # Build superfamily mapping
    tpase_tags = [t for t, b in zip(all_tags, all_binary) if b == 1]
    tag_counts = Counter(tpase_tags)
    
    keep_superfamilies = {t for t, c in tag_counts.items() if c >= min_class_count}
    dropped = {t: c for t, c in tag_counts.items() if c < min_class_count}
    if dropped:
        print(f"\nFiltering rare superfamilies (<{min_class_count}): {dropped}")
    
    superfamily_names = sorted(keep_superfamilies)
    superfamily_to_id = {t: i for i, t in enumerate(superfamily_names)}
    n_superfamilies = len(superfamily_names)
    
    print(f"\nSuperfamilies ({n_superfamilies}): {superfamily_names}")
    
    # Filter and assign labels
    filtered_h, filtered_s, filtered_tags, filtered_binary, filtered_class = [], [], [], [], []
    
    for h, s, tag, binary in zip(all_h, all_s, all_tags, all_binary):
        if binary == 0:
            filtered_h.append(h)
            filtered_s.append(s)
            filtered_tags.append(tag)
            filtered_binary.append(0)
            filtered_class.append(0)
        elif tag in superfamily_to_id:
            filtered_h.append(h)
            filtered_s.append(s)
            filtered_tags.append(tag)
            filtered_binary.append(1)
            filtered_class.append(superfamily_to_id[tag])
    
    all_h = filtered_h
    all_s = filtered_s
    all_tags = filtered_tags
    all_binary = np.array(filtered_binary, dtype=np.int64)
    all_class_ids = np.array(filtered_class, dtype=np.int64)
    
    del filtered_h, filtered_s, filtered_tags, filtered_binary, filtered_class
    gc.collect()
    
    print(f"\nFinal dataset: {len(all_h)} sequences")
    print(f"  Binary: {(all_binary == 1).sum()} transposase+ | {(all_binary == 0).sum()} None")
    
    # ---- Train/test split ----
    idx_train, idx_test = train_test_split(
        np.arange(len(all_h)), test_size=test_size, stratify=all_binary, random_state=random_state
    )
    
    train_h = [all_h[i] for i in idx_train]
    train_s = [all_s[i] for i in idx_train]
    train_binary = all_binary[idx_train]
    train_class = all_class_ids[idx_train]
    
    test_h = [all_h[i] for i in idx_test]
    test_s = [all_s[i] for i in idx_test]
    test_binary = all_binary[idx_test]
    test_class = all_class_ids[idx_test]
    
    print(f"\nTrain: {len(train_h)}, Test: {len(test_h)}")
    
    # ---- Model ----
    model = HierarchicalRCCNN(
        num_superfamilies=n_superfamilies,
        width=width,
        motif_kernels=motif_kernels,
        context_kernel=context_kernel,
        context_dilations=context_dilations,
        dropout=dropout,
        rc_mode=rc_mode
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ---- Loss functions ----
    n_pos = float((train_binary == 1).sum())
    n_neg = float((train_binary == 0).sum())
    binary_weight_tensor = torch.tensor(
        [n_pos / (n_pos + n_neg), n_neg / (n_pos + n_neg)], 
        dtype=torch.float32, device=device
    )
    binary_loss_fn = nn.CrossEntropyLoss(weight=binary_weight_tensor)
    
    tpase_mask = train_binary == 1
    tpase_classes = train_class[tpase_mask]
    sf_weights = compute_class_weights(tpase_classes, n_superfamilies, mode="inv_sqrt")
    sf_weights_t = torch.tensor(sf_weights, dtype=torch.float32, device=device)
    
    if label_smoothing > 0:
        superfamily_loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=sf_weights_t)
    else:
        superfamily_loss_fn = nn.CrossEntropyLoss(weight=sf_weights_t)
    
    boundary_loss_fn = nn.MSELoss()
    
    # ---- Optimizer ----
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.get("weight_decay", DEFAULTS["weight_decay"]))
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    
    # ---- Datasets ----
    print("\n=== Creating datasets ===")
    ds_train = SeqDatasetHierarchical(train_h, train_s, train_binary, train_class, fixed_length)
    ds_test = SeqDatasetHierarchical(test_h, test_s, test_binary, test_class, fixed_length)
    
    n_train_tpase = int((train_binary == 1).sum())
    
    del train_h, train_s, train_binary, train_class
    del test_h, test_s, test_binary, test_class
    del all_h, all_s, all_tags, all_binary, all_class_ids
    gc.collect()
    
    # Create collate function with fixed_length bound
    def collate_fn(batch):
        return collate_hierarchical(batch, fixed_length)
    
    loader_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn
    )
    loader_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    # ---- Training loop ----
    print("\n=== Training hierarchical model ===")
    history = {
        "train_loss": [], "train_binary_loss": [], "train_sf_loss": [],
        "val_binary_acc": [], "val_binary_f1": [], "val_sf_acc": [], "val_sf_f1": []
    }
    best_state, best_epoch = None, None
    best_score = -math.inf
    bad = 0
    
    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        running_loss, running_bin, running_sf = 0.0, 0.0, 0.0
        
        pbar = tqdm(loader_train, desc=f"Epoch {ep}/{epochs}", leave=False)
        for _, X, mask, Y_bin, Y_sf, starts, ends, lengths in pbar:
            X = X.to(device)
            mask = mask.to(device)
            Y_bin = Y_bin.to(device)
            Y_sf = Y_sf.to(device)
            boundary_target = torch.stack([starts, ends, lengths], dim=1).to(device)
            
            binary_logits, sf_logits, boundary_pred = model(X, mask)
            
            bin_loss = binary_loss_fn(binary_logits, Y_bin)
            
            tpase_mask_batch = Y_bin == 1
            if tpase_mask_batch.sum() > 0:
                sf_loss = superfamily_loss_fn(sf_logits[tpase_mask_batch], Y_sf[tpase_mask_batch])
            else:
                sf_loss = torch.tensor(0.0, device=device)
            
            aux_loss = boundary_loss_fn(boundary_pred, boundary_target)
            
            loss = binary_weight * bin_loss + superfamily_weight * sf_loss + aux_weight * aux_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running_loss += loss.item() * X.size(0)
            running_bin += bin_loss.item() * X.size(0)
            running_sf += sf_loss.item() * tpase_mask_batch.sum().item() if tpase_mask_batch.sum() > 0 else 0
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        train_loss = running_loss / len(ds_train)
        train_bin = running_bin / len(ds_train)
        train_sf = running_sf / max(1, n_train_tpase)
        
        # ---- Evaluate ----
        model.eval()
        all_bin_pred, all_bin_true = [], []
        all_sf_pred, all_sf_true = [], []
        
        with torch.no_grad():
            for _, X, mask, Y_bin, Y_sf, starts, ends, lengths in loader_test:
                X = X.to(device)
                mask = mask.to(device)
                
                binary_logits, sf_logits, _ = model(X, mask)
                
                bin_pred = binary_logits.argmax(dim=1).cpu().numpy()
                sf_pred = sf_logits.argmax(dim=1).cpu().numpy()
                
                all_bin_pred.extend(bin_pred)
                all_bin_true.extend(Y_bin.numpy())
                
                tpase_mask_batch = Y_bin == 1
                all_sf_pred.extend(sf_pred[tpase_mask_batch.numpy()])
                all_sf_true.extend(Y_sf[tpase_mask_batch].numpy())
        
        all_bin_pred = np.array(all_bin_pred)
        all_bin_true = np.array(all_bin_true)
        all_sf_pred = np.array(all_sf_pred)
        all_sf_true = np.array(all_sf_true)
        
        bin_acc = accuracy_score(all_bin_true, all_bin_pred)
        bin_f1 = f1_score(all_bin_true, all_bin_pred, average="binary")
        sf_acc = accuracy_score(all_sf_true, all_sf_pred) if len(all_sf_true) > 0 else 0
        sf_f1 = f1_score(all_sf_true, all_sf_pred, average="macro", zero_division=0) if len(all_sf_true) > 0 else 0
        
        history["train_loss"].append(train_loss)
        history["train_binary_loss"].append(train_bin)
        history["train_sf_loss"].append(train_sf)
        history["val_binary_acc"].append(bin_acc)
        history["val_binary_f1"].append(bin_f1)
        history["val_sf_acc"].append(sf_acc)
        history["val_sf_f1"].append(sf_f1)
        
        combined_score = 0.5 * bin_f1 + 0.5 * sf_f1
        
        print(f"Epoch {ep}: loss {train_loss:.4f} | binary acc {bin_acc:.4f} F1 {bin_f1:.4f} | "
              f"superfamily acc {sf_acc:.4f} F1 {sf_f1:.4f}")
        
        if combined_score > best_score + 1e-4:
            best_score = combined_score
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break
    
    # ---- Final evaluation ----
    print("\n" + "=" * 60)
    print("Final TEST SET Evaluation")
    print("=" * 60)
    
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    
    model.eval()
    all_bin_pred, all_bin_true = [], []
    all_sf_pred, all_sf_true = [], []
    
    with torch.no_grad():
        for _, X, mask, Y_bin, Y_sf, starts, ends, lengths in loader_test:
            X = X.to(device)
            mask = mask.to(device)
            
            binary_logits, sf_logits, _ = model(X, mask)
            
            bin_pred = binary_logits.argmax(dim=1).cpu().numpy()
            sf_pred = sf_logits.argmax(dim=1).cpu().numpy()
            
            all_bin_pred.extend(bin_pred)
            all_bin_true.extend(Y_bin.numpy())
            
            tpase_mask_batch = Y_bin == 1
            all_sf_pred.extend(sf_pred[tpase_mask_batch.numpy()])
            all_sf_true.extend(Y_sf[tpase_mask_batch].numpy())
    
    all_bin_pred = np.array(all_bin_pred)
    all_bin_true = np.array(all_bin_true)
    all_sf_pred = np.array(all_sf_pred)
    all_sf_true = np.array(all_sf_true)
    
    print("\n--- Binary Classification (Transposase+ vs None) ---")
    print(classification_report(all_bin_true, all_bin_pred, target_names=["None", "Transposase+"], zero_division=0))
    
    print("\n--- Superfamily Classification (Transposase+ only) ---")
    print(classification_report(all_sf_true, all_sf_pred, target_names=superfamily_names, zero_division=0))
    
    # Save checkpoint
    if best_state is not None:
        ckpt = {
            "model_state_dict": best_state,
            "superfamily_names": superfamily_names,
            "superfamily_to_id": superfamily_to_id,
            "arch": {
                "width": width,
                "motif_kernels": motif_kernels,
                "context_kernel": context_kernel,
                "context_dilations": context_dilations,
                "rc_mode": rc_mode,
                "num_superfamilies": n_superfamilies,
                "fixed_length": fixed_length,
            },
            "best_epoch": best_epoch,
            "history": history,
        }
        save_checkpoint(ckpt, save_dir, config.get("checkpoint_prefix", "rc_cnn_hierarchical_v3"))
    
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "superfamily_names": superfamily_names,
        "superfamily_to_id": superfamily_to_id,
        "test_binary_pred": all_bin_pred,
        "test_binary_true": all_bin_true,
        "test_sf_pred": all_sf_pred,
        "test_sf_true": all_sf_true,
        "device": str(device),
    }


# ============================================================
# CLI Interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Hierarchical TE Classification Model (V3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to YAML config file")
    
    # Data paths (override config)
    parser.add_argument("-f", "--fasta", type=str, dest="fasta_path",
                        help="Path to FASTA file")
    parser.add_argument("-l", "--labels", type=str, dest="label_path",
                        help="Path to label file")
    parser.add_argument("-o", "--output", type=str, dest="save_dir",
                        help="Directory to save checkpoints")
    
    # Training params (override config)
    parser.add_argument("-b", "--batch-size", type=int, dest="batch_size",
                        help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, dest="epochs",
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, dest="learning_rate",
                        help="Learning rate")
    parser.add_argument("-p", "--patience", type=int, dest="patience",
                        help="Early stopping patience")
    parser.add_argument("-d", "--device", type=str, dest="device",
                        help="Device (cuda, mps, cpu)")
    
    # Model params
    parser.add_argument("--width", type=int, help="Base channel width")
    parser.add_argument("--fixed-length", type=int, dest="fixed_length",
                        help="Fixed canvas length (bp)")
    parser.add_argument("--subsample-none", type=int, dest="subsample_none",
                        help="Subsample None sequences (0 = no subsampling)")
    
    return parser.parse_args()


def load_config(config_path):
    """Load config from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested config
    flat = {}
    if "data" in config:
        flat.update(config["data"])
    if "sequence" in config:
        flat.update(config["sequence"])
    if "model" in config:
        flat.update(config["model"])
    if "training" in config:
        flat.update(config["training"])
    if "split" in config:
        flat.update(config["split"])
    if "output" in config:
        flat.update(config["output"])
    if "hardware" in config:
        flat.update(config["hardware"])
    
    return flat


def main():
    args = parse_args()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Auto-detect config file if not provided
    if args.config:
        config = load_config(args.config)
    else:
        # Look for default config file in script directory
        default_config = script_dir / "config_hierarchical_v3.yaml"
        if default_config.exists():
            print(f"Auto-loading config from: {default_config}")
            config = load_config(default_config)
        else:
            config = {}
    
    # Override with CLI args
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    config.update(cli_overrides)
    
    # Validate required params
    if "fasta_path" not in config or "label_path" not in config:
        print("Error: fasta_path and label_path are required")
        print("Provide via config file (-c) or CLI args (-f, -l)")
        sys.exit(1)
    
    # Set default save_dir
    if "save_dir" not in config:
        config["save_dir"] = "checkpoints"
    
    # Handle subsample_none = 0 as None (no subsampling)
    if config.get("subsample_none") == 0:
        config["subsample_none"] = None
    
    # Run training
    start_time = time.time()
    results = run_train_v3(config)
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best epoch: {results['best_epoch']}")
    
    return results


if __name__ == "__main__":
    main()
