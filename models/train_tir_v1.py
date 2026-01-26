#!/usr/bin/env python3
"""
TIR Binary Classification Model Training Script (V1)
=====================================================

This script trains a CNN model for TIR (Terminal Inverted Repeat) presence/absence
binary classification:
- TRUE: sequence has TIR
- FALSE: sequence does not have TIR

Usage:
    python train_tir_v1.py -c config_tir_v1.yaml
    python train_tir_v1.py -f data/vgp/all_vgp_tes.fa -l data/vgp/features -o checkpoints

Author: Alex Yang
Date: 2026-01-16
"""

import os
import gc
import sys
import math
import argparse
import time
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, classification_report,
    roc_auc_score, average_precision_score
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
    "width": 128,
    "motif_kernels": [7, 15, 21],
    "context_kernel": 9,
    "context_dilations": [1, 2, 4, 8],
    "dropout": 0.15,
    "rc_mode": "early",
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.0003,     # Lower LR to prevent collapse
    "weight_decay": 0.0001,
    "patience": 15,
    "label_smoothing": 0.0,      # Disabled by default
    "class_weight_mode": "inv_sqrt",  # Stronger class weighting
    "test_size": 0.2,
    "random_state": 42,
    "subset_size": None,  # Limit total samples (e.g., 30000)
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


def load_tir_labels(label_path):
    """
    Load TIR presence/absence labels.
    
    Expected format: >header\tTRUE or >header\tFALSE
    
    Returns:
        label_dict: header -> 1 (TRUE/has TIR) or 0 (FALSE/no TIR)
    """
    label_path = Path(label_path)
    label_dict = {}
    n_true = 0
    n_false = 0
    
    with label_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            header = parts[0].lstrip('>')
            tag = parts[1].upper()
            
            if tag == "TRUE":
                label_dict[header] = 1
                n_true += 1
            elif tag == "FALSE":
                label_dict[header] = 0
                n_false += 1
            else:
                print(f"Warning: Unknown label '{tag}' for {header}")
                continue
    
    total = n_true + n_false
    print(f"Loaded {total} TIR labels:")
    print(f"  ✓ TRUE (has TIR):  {n_true} ({100*n_true/total:.1f}%)")
    print(f"  ✗ FALSE (no TIR):  {n_false} ({100*n_false/total:.1f}%)")
    
    return label_dict


def compute_binary_class_weights(labels, mode="balanced", eps=1e-6):
    """Compute class weights for imbalanced binary classification."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    
    if mode == "none":
        w = np.ones(2, dtype=np.float32)
    elif mode == "inv":
        w = 1.0 / (counts + eps)
    elif mode == "inv_sqrt":
        w = 1.0 / np.sqrt(counts + eps)
    elif mode == "balanced":
        # sklearn-style balanced weights
        n_samples = len(labels)
        w = n_samples / (2 * counts + eps)
    else:
        raise ValueError(f"Unknown mode={mode}")
    
    w = w / (w.mean() + 1e-12)  # normalize to mean=1
    return w.astype(np.float32)


# ============================================================
# Dataset and DataLoader
# ============================================================

class TIRDataset(Dataset):
    """
    Dataset for TIR presence/absence binary classification.
    
    Each sequence is placed at a random position within a fixed-length canvas.
    This provides data augmentation and forces model to learn position-invariant features.
    """
    def __init__(self, headers, sequences, labels, fixed_length):
        self.headers = list(headers)
        self.sequences = list(sequences)
        self.labels = np.asarray(labels, dtype=np.int64)
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
        
        # Random placement within canvas
        max_start = max(0, self.fixed_length - seq_len)
        if max_start > 0:
            start_pos = np.random.randint(0, max_start + 1)
        else:
            start_pos = 0
        
        end_pos = start_pos + seq_len
        
        return (
            self.headers[idx],
            seq_idx,
            int(self.labels[idx]),
            start_pos,
            end_pos,
            seq_len
        )


def collate_tir(batch, fixed_length):
    """
    Collate function for TIR binary classification.
    
    Returns:
        headers: list of header strings
        X: (B, 5, fixed_length) one-hot encoded sequences
        mask: (B, fixed_length) True where real bases exist
        Y: (B,) binary labels (0=FALSE, 1=TRUE)
        lengths: (B,) sequence lengths (normalized to [0,1])
    """
    headers, seq_idxs, labels, starts, ends, lengths = zip(*batch)
    
    B = len(batch)
    X = torch.zeros((B, 5, fixed_length), dtype=torch.float32)
    mask = torch.zeros((B, fixed_length), dtype=torch.bool)
    
    for i, (seq_idx, start, end, seq_len) in enumerate(zip(seq_idxs, starts, ends, lengths)):
        actual_len = min(seq_len, fixed_length - start)
        if actual_len > 0:
            idx = torch.from_numpy(seq_idx[:actual_len].astype(np.int64))
            pos = torch.arange(actual_len, dtype=torch.long) + start
            X[i, idx, pos] = 1.0
            mask[i, start:start + actual_len] = (idx != 4)  # Mask out N's
    
    Y = torch.tensor(labels, dtype=torch.long)
    lengths_norm = torch.tensor(lengths, dtype=torch.float32) / fixed_length
    
    return list(headers), X, mask, Y, lengths_norm


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
# TIR Binary Classification Model
# ============================================================

class TIRCNN(nn.Module):
    """
    RC-invariant CNN for TIR presence/absence binary classification.
    
    Architecture:
    - Multi-scale motif detection (different kernel sizes)
    - Dilated convolutions for context
    - RC-invariance via early or late fusion
    - Binary output: TRUE (has TIR) vs FALSE (no TIR)
    """
    def __init__(
        self,
        width=128,
        motif_kernels=(7, 15, 21),
        context_kernel=9,
        context_dilations=(1, 2, 4, 8),
        dropout=0.15,
        rc_mode="early"
    ):
        super().__init__()
        self.rc_mode = rc_mode
        
        # ---- Motif detection layers ----
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
        
        # ---- Mix layer ----
        in_ch = width * len(motif_kernels)
        self.mix = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=1, bias=True),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # ---- Context blocks ----
        self.context_blocks = nn.ModuleList([
            ConvBlock(width, width, kernel_size=context_kernel, dilation=d, dropout=dropout)
            for d in context_dilations
        ])
        self.pool = MaskedMaxPool1d(kernel_size=2, stride=2)
        
        # ---- Binary classification head ----
        self.classifier = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),  # Binary: 0=FALSE (no TIR), 1=TRUE (has TIR)
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
        """
        Returns:
            logits: (B, 2) - binary classification logits
        """
        if self.rc_mode == "late":
            f = self.encode(x, mask)
            x_rc, mask_rc = self.rc_transform(x, mask)
            r = self.encode(x_rc, mask_rc)
            pooled = 0.5 * (f + r)
        else:
            pooled = self.encode(x, mask)
        
        logits = self.classifier(pooled)
        return logits


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

def run_train_tir(config):
    """
    Train TIR presence/absence binary classifier.
    
    Task: Predict TRUE (has TIR) vs FALSE (no TIR)
    
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
    
    label_smoothing = config.get("label_smoothing", DEFAULTS["label_smoothing"])
    class_weight_mode = config.get("class_weight_mode", DEFAULTS["class_weight_mode"])
    
    test_size = config.get("test_size", DEFAULTS["test_size"])
    random_state = config.get("random_state", DEFAULTS["random_state"])
    subset_size = config.get("subset_size", DEFAULTS["subset_size"])
    num_workers = config.get("num_workers", DEFAULTS["num_workers"])
    
    device_str = config.get("device", None)
    device = resolve_device(device_str)
    
    print("=" * 60)
    print("TIR Binary Classification Training (V1)")
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
    label_dict = load_tir_labels(label_path)
    
    # Match headers to labels
    all_h, all_s, all_labels = [], [], []
    skipped = 0
    for h, s in zip(headers, sequences):
        if h not in label_dict:
            skipped += 1
            continue
        all_h.append(h)
        all_s.append(s)
        all_labels.append(label_dict[h])
    
    del headers, sequences
    gc.collect()
    
    print(f"\nMatched {len(all_h)} sequences (skipped {skipped} without labels)")
    
    all_labels = np.array(all_labels, dtype=np.int64)
    n_true = (all_labels == 1).sum()
    n_false = (all_labels == 0).sum()
    print(f"Full distribution: {n_true} TRUE ({100*n_true/len(all_labels):.1f}%) | "
          f"{n_false} FALSE ({100*n_false/len(all_labels):.1f}%)")
    
    # ---- Subset sampling (stratified) ----
    if subset_size is not None and subset_size < len(all_h):
        print(f"\n=== Subsampling to {subset_size} samples (stratified) ===")
        
        idx_keep, _ = train_test_split(
            np.arange(len(all_h)), 
            train_size=subset_size, 
            stratify=all_labels, 
            random_state=random_state
        )
        
        all_h = [all_h[i] for i in idx_keep]
        all_s = [all_s[i] for i in idx_keep]
        all_labels = all_labels[idx_keep]
        
        n_true = (all_labels == 1).sum()
        n_false = (all_labels == 0).sum()
        print(f"Subset distribution: {n_true} TRUE ({100*n_true/len(all_labels):.1f}%) | "
              f"{n_false} FALSE ({100*n_false/len(all_labels):.1f}%)")
        
        gc.collect()
    
    # ---- Train/test split ----
    idx_train, idx_test = train_test_split(
        np.arange(len(all_h)), test_size=test_size, stratify=all_labels, random_state=random_state
    )
    
    train_h = [all_h[i] for i in idx_train]
    train_s = [all_s[i] for i in idx_train]
    train_labels = all_labels[idx_train]
    
    test_h = [all_h[i] for i in idx_test]
    test_s = [all_s[i] for i in idx_test]
    test_labels = all_labels[idx_test]
    
    print(f"\nTrain: {len(train_h)}, Test: {len(test_h)}")
    
    # ---- Model ----
    model = TIRCNN(
        width=width,
        motif_kernels=motif_kernels,
        context_kernel=context_kernel,
        context_dilations=context_dilations,
        dropout=dropout,
        rc_mode=rc_mode
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ---- Loss function with class weights ----
    class_weights = compute_binary_class_weights(train_labels, mode=class_weight_mode)
    print(f"Class weights ({class_weight_mode}): FALSE={class_weights[0]:.3f}, TRUE={class_weights[1]:.3f}")
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    if label_smoothing > 0:
        print(f"Label smoothing: {label_smoothing}")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
    
    # ---- Optimizer ----
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.get("weight_decay", DEFAULTS["weight_decay"]))
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    
    # ---- Datasets ----
    print("\n=== Creating datasets ===")
    ds_train = TIRDataset(train_h, train_s, train_labels, fixed_length)
    ds_test = TIRDataset(test_h, test_s, test_labels, fixed_length)
    
    del train_h, train_s, test_h, test_s
    del all_h, all_s, all_labels
    gc.collect()
    
    # Create collate function with fixed_length bound
    def collate_fn(batch):
        return collate_tir(batch, fixed_length)
    
    loader_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn
    )
    loader_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    print(f"DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    print(f"Batches per epoch: {len(loader_train)}")
    
    # ---- Training loop ----
    print("\n=== Training TIR classifier ===")
    history = {
        "train_loss": [],
        "val_acc": [], "val_bal_acc": [], "val_f1": [],
        "val_auroc": [], "val_auprc": []
    }
    best_state, best_epoch = None, None
    best_score = -math.inf
    bad = 0
    
    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        n_samples = 0
        
        pbar = tqdm(loader_train, desc=f"Epoch {ep}/{epochs}", leave=False)
        for _, X, mask, Y, lengths in pbar:
            X = X.to(device)
            mask = mask.to(device)
            Y = Y.to(device)
            
            logits = model(X, mask)
            loss = loss_fn(logits, Y)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            running_loss += loss.item() * X.size(0)
            n_samples += X.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        train_loss = running_loss / n_samples
        
        # ---- Evaluate ----
        model.eval()
        all_pred, all_true, all_probs = [], [], []
        
        with torch.no_grad():
            for _, X, mask, Y, lengths in loader_test:
                X = X.to(device)
                mask = mask.to(device)
                
                logits = model(X, mask)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(TRUE)
                pred = logits.argmax(dim=1).cpu().numpy()
                
                all_pred.extend(pred)
                all_true.extend(Y.numpy())
                all_probs.extend(probs)
        
        all_pred = np.array(all_pred)
        all_true = np.array(all_true)
        all_probs = np.array(all_probs)
        
        acc = accuracy_score(all_true, all_pred)
        bal_acc = balanced_accuracy_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred, average="binary")
        auroc = roc_auc_score(all_true, all_probs)
        auprc = average_precision_score(all_true, all_probs)
        
        history["train_loss"].append(train_loss)
        history["val_acc"].append(acc)
        history["val_bal_acc"].append(bal_acc)
        history["val_f1"].append(f1)
        history["val_auroc"].append(auroc)
        history["val_auprc"].append(auprc)
        
        print(f"Epoch {ep}: loss {train_loss:.4f} | acc {acc:.4f} bal_acc {bal_acc:.4f} F1 {f1:.4f} | "
              f"AUROC {auroc:.4f} AUPRC {auprc:.4f}")
        
        # Use F1 as primary metric
        if f1 > best_score + 1e-4:
            best_score = f1
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
    print(f"Best epoch: {best_epoch}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    
    model.eval()
    all_pred, all_true, all_probs = [], [], []
    
    with torch.no_grad():
        for _, X, mask, Y, lengths in loader_test:
            X = X.to(device)
            mask = mask.to(device)
            
            logits = model(X, mask)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            
            all_pred.extend(pred)
            all_true.extend(Y.numpy())
            all_probs.extend(probs)
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    all_probs = np.array(all_probs)
    
    test_acc = accuracy_score(all_true, all_pred)
    test_bal_acc = balanced_accuracy_score(all_true, all_pred)
    test_f1 = f1_score(all_true, all_pred, average="binary")
    test_auroc = roc_auc_score(all_true, all_probs)
    test_auprc = average_precision_score(all_true, all_probs)
    
    print("\n--- Classification Report ---")
    print(classification_report(all_true, all_pred, target_names=["FALSE (no TIR)", "TRUE (has TIR)"], zero_division=0))
    
    print(f"\nAUROC: {test_auroc:.4f}")
    print(f"AUPRC: {test_auprc:.4f}")
    
    # Save checkpoint
    if best_state is not None:
        ckpt = {
            "model_state_dict": best_state,
            "arch": {
                "width": width,
                "motif_kernels": motif_kernels,
                "context_kernel": context_kernel,
                "context_dilations": context_dilations,
                "rc_mode": rc_mode,
                "fixed_length": fixed_length,
            },
            "best_epoch": best_epoch,
            "history": history,
            "test_acc": test_acc,
            "test_bal_acc": test_bal_acc,
            "test_f1": test_f1,
            "test_auroc": test_auroc,
            "test_auprc": test_auprc,
        }
        save_checkpoint(ckpt, save_dir, config.get("checkpoint_prefix", "tir_cnn_v1"))
    
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "test_bal_acc": test_bal_acc,
        "test_f1": test_f1,
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
        "test_pred": all_pred,
        "test_true": all_true,
        "test_probs": all_probs,
        "device": str(device),
    }


# ============================================================
# CLI Interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TIR Binary Classification Model (V1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to YAML config file")
    
    # Data paths (override config)
    parser.add_argument("-f", "--fasta", type=str, dest="fasta_path",
                        help="Path to FASTA file")
    parser.add_argument("-l", "--labels", type=str, dest="label_path",
                        help="Path to TIR label file (TRUE/FALSE)")
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
    parser.add_argument("--subset-size", type=int, dest="subset_size",
                        help="Limit total samples (stratified sampling)")
    parser.add_argument("--rc-mode", type=str, dest="rc_mode", choices=["early", "late"],
                        help="RC fusion mode (early or late)")
    
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
        default_config = script_dir / "config_tir_v1.yaml"
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
    
    # Run training
    start_time = time.time()
    results = run_train_tir(config)
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"\nFinal Test Metrics:")
    print(f"  Accuracy:     {results['test_acc']:.4f}")
    print(f"  Balanced Acc: {results['test_bal_acc']:.4f}")
    print(f"  F1 Score:     {results['test_f1']:.4f}")
    print(f"  AUROC:        {results['test_auroc']:.4f}")
    print(f"  AUPRC:        {results['test_auprc']:.4f}")
    
    return results


if __name__ == "__main__":
    main()
