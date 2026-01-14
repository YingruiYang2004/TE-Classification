#!/usr/bin/env python3
"""
GNN TE Classification - GPU Cluster Training Script
Generated from notebook. Run with: python train_gnn_cluster.py
"""
import os
import sys
import argparse
from pathlib import Path
from collections import Counter, namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix)
from tqdm import tqdm
import time

# ============================================================================
# DATA LOADING
# ============================================================================
def read_fasta(path: Path) -> Tuple[List[str], List[str]]:
    headers, seqs = [], []
    with open(path) as f:
        seq_parts = []
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    seqs.append("".join(seq_parts))
                header = line[1:]
                headers.append(header)
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            seqs.append("".join(seq_parts))
    return headers, seqs

def header_to_tag(header: str) -> str:
    parts = header.split("#")
    return parts[1].strip() if len(parts) > 1 else "Unknown"

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
class KmerWindowFeaturizer:
    def __init__(self, k: int = 7, dim: int = 2048, window: int = 512, stride: int = 256,
                 add_pos: bool = True, l2_normalize: bool = True):
        self.k = k
        self.dim = dim
        self.window = window
        self.stride = stride
        self.add_pos = add_pos
        self.l2_normalize = l2_normalize
        self.base_map = {"A": 0, "C": 1, "G": 2, "T": 3}

    def _kmer_to_hash(self, kmer: str) -> int:
        h = 0
        for c in kmer.upper():
            h = h * 4 + self.base_map.get(c, 0)
        return h % self.dim

    def featurize_sequence(self, seq: str) -> Tuple[np.ndarray, int]:
        seq = seq.upper()
        L = len(seq)
        if L < self.k:
            starts = [0]
        else:
            starts = list(range(0, max(1, L - self.window + 1), self.stride))
            if starts and starts[-1] + self.window < L:
                starts.append(L - self.window)

        n_windows = len(starts)
        out_dim = self.dim + 1 if self.add_pos else self.dim
        X = np.zeros((n_windows, out_dim), dtype=np.float32)

        for i, s in enumerate(starts):
            e = min(s + self.window, L)
            substr = seq[s:e]
            if len(substr) >= self.k:
                for j in range(len(substr) - self.k + 1):
                    kmer = substr[j:j + self.k]
                    if all(c in "ACGT" for c in kmer):
                        X[i, self._kmer_to_hash(kmer)] += 1
            norm = np.linalg.norm(X[i, :self.dim])
            if norm > 0 and self.l2_normalize:
                X[i, :self.dim] /= norm
            if self.add_pos:
                X[i, self.dim] = (s + e) / (2.0 * max(1, L))
        return X, n_windows

# ============================================================================
# GRAPH UTILITIES
# ============================================================================
GraphSample = namedtuple("GraphSample", ["x", "edge_index", "y", "header"])

def build_chain_edge_index(n: int, undirected: bool = True, self_loops: bool = True) -> torch.LongTensor:
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long)
    src, dst = [], []
    for i in range(n - 1):
        src.append(i)
        dst.append(i + 1)
        if undirected:
            src.append(i + 1)
            dst.append(i)
    if self_loops:
        for i in range(n):
            src.append(i)
            dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)

class PrecomputedGraphDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, headers):
        self.features = features
        self.labels = labels.astype(np.int64)
        self.headers = headers

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        X = self.features[idx]
        x = torch.from_numpy(X).to(torch.float32)
        ei = build_chain_edge_index(x.size(0), undirected=True, self_loops=True)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.int64)
        return GraphSample(x=x, edge_index=ei, y=y, header=self.headers[idx])

def collate_graphs(batch):
    xs, eis, ys, hs = [], [], [], []
    offset = 0
    for g in batch:
        xs.append(g.x)
        eis.append(g.edge_index + offset)
        ys.append(g.y)
        hs.append(g.header)
        offset += g.x.size(0)
    x = torch.cat(xs, dim=0)
    ei = torch.cat(eis, dim=1)
    y = torch.stack(ys, dim=0)
    batch_vec = torch.cat([torch.full((g.x.size(0),), i, dtype=torch.long) for i, g in enumerate(batch)])
    return x, ei, batch_vec, y, hs

# ============================================================================
# GNN MODEL
# ============================================================================
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, edge_index):
        src, dst = edge_index
        neigh = torch.zeros_like(x)
        counts = torch.zeros(x.size(0), 1, device=x.device)
        neigh.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        counts.scatter_add_(0, dst.unsqueeze(1), torch.ones_like(src, dtype=torch.float).unsqueeze(1))
        counts = counts.clamp(min=1)
        neigh = neigh / counts
        return self.lin_self(x) + self.lin_neigh(neigh)

class GNNClassifier(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, n_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden] * n_layers
        for i in range(n_layers):
            self.layers.append(GraphSAGELayer(dims[i], dims[i + 1]))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = self.drop(x)
        pooled = torch.zeros(batch.max() + 1, x.size(1), device=x.device)
        pooled.scatter_add_(0, batch.unsqueeze(1).expand(-1, x.size(1)), x)
        counts = torch.bincount(batch, minlength=batch.max() + 1).float().unsqueeze(1)
        pooled = pooled / counts.clamp(min=1)
        return self.head(pooled)

# ============================================================================
# TRAINING UTILITIES
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, 
                             reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    logits_list, y_list = [], []
    for x, ei, bv, y, _ in loader:
        logits = model(x.to(device), ei.to(device), bv.to(device))
        logits_list.append(logits.cpu())
        y_list.append(y.cpu())
    return torch.cat(logits_list), torch.cat(y_list)

def evaluate_multiclass(logits, y):
    y_np = y.numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    pred = probs.argmax(axis=1)
    out = {'acc': accuracy_score(y_np, pred), 'balanced_acc': balanced_accuracy_score(y_np, pred)}
    try:
        out['auroc'] = roc_auc_score(y_np, probs, multi_class='ovr', average='macro')
    except:
        out['auroc'] = float('nan')
    return out

# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', type=str, required=True)
    parser.add_argument('--out', type=str, default='./output')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--min-class', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    headers, sequences = read_fasta(Path(args.fasta))
    tags = [header_to_tag(h) for h in headers]

    # Filter
    exclude = {'Unknown'}
    keep = [(h, s, t) for h, s, t in zip(headers, sequences, tags) if t not in exclude]
    headers, sequences, tags = zip(*keep)
    headers, sequences, tags = list(headers), list(sequences), list(tags)

    # Label encoding
    from collections import Counter
    tag_counts = Counter(tags)
    tag_to_id = {t: i for i, t in enumerate(sorted(set(tags)))}
    id_to_tag = list(tag_to_id.keys())
    y = np.array([tag_to_id[t] for t in tags])

    # Drop rare
    counts = np.bincount(y)
    keep_cls = np.where(counts >= args.min_class)[0]
    keep_mask = np.isin(y, keep_cls)
    headers = [h for h, m in zip(headers, keep_mask) if m]
    sequences = [s for s, m in zip(sequences, keep_mask) if m]
    y = y[keep_mask]
    old_to_new = {old: new for new, old in enumerate(keep_cls)}
    y = np.array([old_to_new[i] for i in y])
    id_to_tag = [id_to_tag[old] for old in keep_cls]

    print(f"Final: {len(sequences)} sequences, {len(id_to_tag)} classes")

    # Featurize
    feat = KmerWindowFeaturizer(k=7, dim=2048, window=512, stride=256)
    print("Featurizing...")
    features = [feat.featurize_sequence(s)[0] for s in tqdm(sequences)]

    # Train
    idx = np.arange(len(sequences))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    in_dim = 2048 + 1

    for fold_i, (tr, va) in enumerate(skf.split(idx_train, y[idx_train])):
        print(f"\nFold {fold_i + 1}/{args.folds}")

        ds_tr = PrecomputedGraphDataset([features[idx_train[i]] for i in tr], y[idx_train[tr]], 
                                        [headers[idx_train[i]] for i in tr])
        ds_va = PrecomputedGraphDataset([features[idx_train[i]] for i in va], y[idx_train[va]],
                                        [headers[idx_train[i]] for i in va])

        loader_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                                                  num_workers=args.workers, collate_fn=collate_graphs)
        loader_va = torch.utils.data.DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                                                  num_workers=args.workers, collate_fn=collate_graphs)

        model = GNNClassifier(in_dim, args.hidden, len(id_to_tag), args.layers, 0.2).to(device)
        class_w = compute_class_weights(y[idx_train[tr]], len(id_to_tag)).to(device)
        loss_fn = FocalLoss(alpha=class_w, gamma=2.0)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

        best_score, best_state = -1, None
        for ep in range(1, args.epochs + 1):
            model.train()
            for x, ei, bv, yb, _ in loader_tr:
                x, ei, bv, yb = x.to(device), ei.to(device), bv.to(device), yb.to(device)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        loss = loss_fn(model(x, ei, bv), yb)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss = loss_fn(model(x, ei, bv), yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            scheduler.step()

            val_logits, val_y = predict_loader(model, loader_va, device)
            m = evaluate_multiclass(val_logits, val_y)
            print(f"  Ep {ep:02d} | bacc={m['balanced_acc']:.4f} | auroc={m['auroc']:.4f}")

            if m['balanced_acc'] > best_score:
                best_score = m['balanced_acc']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Save fold
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'state': best_state, 'id_to_tag': id_to_tag, 'best_bacc': best_score}, 
                   out_dir / f'fold_{fold_i}.pt')

    print("\nDone!")

if __name__ == '__main__':
    main()
