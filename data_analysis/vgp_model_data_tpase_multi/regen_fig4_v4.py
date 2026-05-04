"""
Standalone script to regenerate Figs 4-5 from the v4 hybrid model.
Saves to thesis/figures/new_figures/ with corrected labels.
  Fig 4 → v4_confusion.png (binary labels: non-DNA vs DNA transposon)
  Fig 5 → v4_training.png  (training history curves)
"""

import gc
import json
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # TE Classification/
FASTA_PATH = ROOT / "data/vgp/all_vgp_tes.fa"
LABEL_PATH  = ROOT / "data/vgp/features-tpase"
CHECKPOINT  = Path(__file__).parent / "hybrid_v4_epoch39.pt"
OUT_DIR     = ROOT / "thesis/figures/new_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
FIXED_LENGTH    = 5000
MIN_CLASS_COUNT = 20
KMER_K          = 7
KMER_DIM        = 2048
KMER_WINDOW     = 512
KMER_STRIDE     = 256
GNN_HIDDEN      = 128
GNN_LAYERS      = 3
CNN_WIDTH       = 128
MOTIF_KERNELS   = (7, 15, 21)
CONTEXT_DILATIONS = (1, 2, 4, 8)
RC_FUSION_MODE  = "late"
FUSION_DIM      = 256
NUM_HEADS       = 4
DROPOUT         = 0.15
BATCH_SIZE      = 16
SUBSAMPLE_NONE  = 5000
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# ─── Encoding ─────────────────────────────────────────────────────────────────
ENCODE = np.full(256, 4, dtype=np.int64)
for _ch, _idx in zip(b"ACGTNacgtn", [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]):
    ENCODE[_ch] = _idx
REV_COMP = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)

# ─── K-mer Featuriser ─────────────────────────────────────────────────────────
_ASCII_MAP = np.full(256, 4, dtype=np.uint8)
for _ch, _val in [("A", 0), ("C", 1), ("G", 2), ("T", 3),
                  ("a", 0), ("c", 1), ("g", 2), ("t", 3)]:
    _ASCII_MAP[ord(_ch)] = _val
_COMP = np.array([3, 2, 1, 0], dtype=np.uint8)


def _canonical(arr4):
    c1 = c2 = 0
    for v in arr4:
        c1 = (c1 << 2) | int(v)
    for v in arr4[::-1]:
        c2 = (c2 << 2) | int(_COMP[v])
    return c1 if c1 < c2 else c2


def _hash_u32(x, dim):
    z = (x * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z ^= (z >> 33)
    z = (z * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF
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
                    code = _canonical(kmer)
                    j = _hash_u32(code, self.dim)
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


def build_chain_edge_index(n: int, undirected: bool = True, self_loops: bool = True):
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

# ─── Data loading ─────────────────────────────────────────────────────────────

def read_fasta(path):
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
    label_path = Path(label_path)
    label_dict, binary_dict = {}, {}
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
    return label_dict, binary_dict

# ─── Dataset ──────────────────────────────────────────────────────────────────

class HybridDataset(Dataset):
    def __init__(self, headers, sequences, binary_labels, class_labels,
                 kmer_features, fixed_length=FIXED_LENGTH):
        self.headers = list(headers)
        self.sequences = list(sequences)
        self.binary_labels = np.asarray(binary_labels, dtype=np.int64)
        self.class_labels  = np.asarray(class_labels,  dtype=np.int64)
        self.kmer_features = kmer_features
        self.fixed_length  = fixed_length
        self.seq_lengths   = np.array([len(s) for s in sequences], dtype=np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = len(seq)
        seq_bytes = seq.encode("ascii", "ignore")
        seq_idx = ENCODE[np.frombuffer(seq_bytes, dtype=np.uint8)]
        max_start = max(0, self.fixed_length - seq_len)
        start_pos = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end_pos = start_pos + seq_len
        kmer_feat = self.kmer_features[idx]
        return (
            self.headers[idx], seq_idx,
            int(self.binary_labels[idx]), int(self.class_labels[idx]),
            start_pos, end_pos, seq_len, kmer_feat
        )


def collate_hybrid(batch, fixed_length=FIXED_LENGTH):
    (headers, seq_idxs, binary_labels, class_labels,
     starts, ends, lengths, kmer_feats) = zip(*batch)
    B = len(batch)
    X_cnn = torch.zeros((B, 5, fixed_length), dtype=torch.float32)
    mask  = torch.zeros((B, fixed_length), dtype=torch.bool)
    for i, (seq_idx, start, end, seq_len) in enumerate(zip(seq_idxs, starts, ends, lengths)):
        actual_len = min(seq_len, fixed_length - start)
        if actual_len > 0:
            idx = torch.from_numpy(seq_idx[:actual_len].astype(np.int64))
            pos = torch.arange(actual_len, dtype=torch.long) + start
            X_cnn[i, idx, pos] = 1.0
            mask[i, start:start + actual_len] = (idx != 4)
    Y_binary = torch.tensor(binary_labels, dtype=torch.long)
    Y_class  = torch.tensor(class_labels,  dtype=torch.long)
    xs, eis, batch_vecs = [], [], []
    node_offset = 0
    for gi, kmer_feat in enumerate(kmer_feats):
        x  = torch.from_numpy(kmer_feat).to(torch.float32)
        n  = x.size(0)
        ei = build_chain_edge_index(n, undirected=True, self_loops=True)
        xs.append(x)
        eis.append(ei + node_offset)
        batch_vecs.append(torch.full((n,), gi, dtype=torch.int64))
        node_offset += n
    x_gnn      = torch.cat(xs, dim=0)
    edge_index  = torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.int64)
    batch_vec   = torch.cat(batch_vecs, dim=0)
    return (list(headers), X_cnn, mask, Y_binary, Y_class, x_gnn, edge_index, batch_vec)

# ─── Model architecture ───────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=9, dilation=1, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation, bias=True)
        self.bn   = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)

    def forward(self, x):
        y = self.conv(x)
        y = F.gelu(self.bn(y))
        y = self.drop(y)
        return y + self.proj(x)


class MaskedMaxPool1d(nn.Module):
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
    if mask is None:
        return z.mean(-1)
    m = mask.unsqueeze(1).float()
    return (z * m).sum(-1) / m.sum(-1).clamp_min(1.0)


class RCFirstConv1d(nn.Module):
    def __init__(self, out_channels, kernel_size=15, dilation=1, bias=True, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = (kernel_size // 2) * dilation
        self.conv       = nn.Conv1d(5, out_channels, kernel_size, padding=pad, dilation=dilation, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        y1   = self.conv(x)
        x_rc = x.flip(-1).index_select(1, REV_COMP.to(x.device))
        y2   = self.conv(x_rc).flip(-1)
        y    = torch.max(y1, y2)
        y    = self.batch_norm(y)
        y    = F.gelu(y)
        y    = self.dropout(y)
        return y


class CNNTower(nn.Module):
    def __init__(self, width=128, motif_kernels=(7, 15, 21),
                 context_kernel=9, context_dilations=(1, 2, 4, 8),
                 dropout=0.15, rc_mode="late"):
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
                    nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout)
                )
                for k in motif_kernels
            ])
        in_ch = width * len(motif_kernels)
        self.mix = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=1, bias=True),
            nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout),
        )
        self.context_blocks = nn.ModuleList([
            ConvBlock(width, width, kernel_size=context_kernel, dilation=d, dropout=dropout)
            for d in context_dilations
        ])
        self.pool = MaskedMaxPool1d(kernel_size=2, stride=2)

    @staticmethod
    def rc_transform(x, mask):
        x_rc   = x.index_select(1, REV_COMP.to(x.device)).flip(-1)
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
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.bincount(dst, minlength=x.size(0)).clamp_min(1).to(x.device).to(x.dtype).unsqueeze(1)
        agg = agg / deg
        h = self.lin_self(x) + self.lin_neigh(agg)
        h = F.relu(h)
        return self.dropout(h)


class GNNTower(nn.Module):
    def __init__(self, in_dim, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.out_dim = hidden
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(GraphSAGELayer(d, hidden, dropout=dropout))
            d = hidden
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, batch_vec):
        for layer in self.layers:
            x = layer(x, edge_index)
        B = int(batch_vec.max().item()) + 1 if batch_vec.numel() else 0
        return scatter_mean(x, batch_vec, dim_size=B)


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, cnn_dim=128, gnn_dim=128, fusion_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.cnn_proj   = nn.Linear(cnn_dim, fusion_dim)
        self.gnn_proj   = nn.Linear(gnn_dim, fusion_dim)
        self.ln1        = nn.LayerNorm(fusion_dim)
        self.ln2        = nn.LayerNorm(fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim), nn.GELU(),
            nn.Linear(fusion_dim, 2), nn.Softmax(dim=-1)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, cnn_embed, gnn_embed):
        c = self.ln1(self.cnn_proj(cnn_embed))
        g = self.ln2(self.gnn_proj(gnn_embed))
        combined = torch.stack([c, g], dim=1)
        attn_out, _ = self.cross_attn(combined, combined, combined)
        c_attn, g_attn = attn_out[:, 0], attn_out[:, 1]
        gate_weights = self.gate(torch.cat([c_attn, g_attn], dim=-1))
        fused = gate_weights[:, 0:1] * c_attn + gate_weights[:, 1:2] * g_attn
        return self.out_proj(fused), gate_weights


class HybridTEClassifierV4(nn.Module):
    def __init__(self, num_superfamilies, cnn_width=128, motif_kernels=(7, 15, 21),
                 context_dilations=(1, 2, 4, 8), rc_mode="late",
                 gnn_in_dim=2049, gnn_hidden=128, gnn_layers=3,
                 fusion_dim=256, num_heads=4, dropout=0.15):
        super().__init__()
        self.num_superfamilies = num_superfamilies
        self.cnn_tower = CNNTower(
            width=cnn_width, motif_kernels=motif_kernels,
            context_dilations=context_dilations, dropout=dropout, rc_mode=rc_mode
        )
        self.gnn_tower = GNNTower(in_dim=gnn_in_dim, hidden=gnn_hidden,
                                  n_layers=gnn_layers, dropout=dropout)
        self.fusion = CrossModalAttentionFusion(
            cnn_dim=cnn_width, gnn_dim=gnn_hidden,
            fusion_dim=fusion_dim, num_heads=num_heads, dropout=dropout
        )
        self.binary_head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 2)
        )
        self.superfamily_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_superfamilies)
        )

    def forward(self, x_cnn, mask, x_gnn, edge_index, batch_vec):
        cnn_embed = self.cnn_tower(x_cnn, mask)
        gnn_embed = self.gnn_tower(x_gnn, edge_index, batch_vec)
        fused, gate_weights = self.fusion(cnn_embed, gnn_embed)
        return self.binary_head(fused), self.superfamily_head(fused), gate_weights

# ─── Main ─────────────────────────────────────────────────────────────────────

def build_test_set(ckpt_sf_names=None):
    """Reproduce the exact test split used during training.
    
    If ckpt_sf_names is provided, use exactly those superfamilies (matching
    the training run's SF whitelist) rather than re-deriving from data.
    """
    print("Loading FASTA …")
    headers, sequences = read_fasta(str(FASTA_PATH))
    print("Loading labels …")
    label_dict, binary_dict = load_hierarchical_labels(str(LABEL_PATH))

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
    print(f"Matched {len(all_h)} sequences")

    # Subsample None
    none_idx  = [i for i, b in enumerate(all_binary) if b == 0]
    tpase_idx = [i for i, b in enumerate(all_binary) if b == 1]
    if len(none_idx) > SUBSAMPLE_NONE:
        np.random.seed(RANDOM_STATE)
        sampled_none = np.random.choice(none_idx, SUBSAMPLE_NONE, replace=False)
        keep_idx = sorted(list(tpase_idx) + list(sampled_none))
        all_h     = [all_h[i]    for i in keep_idx]
        all_s     = [all_s[i]    for i in keep_idx]
        all_tags  = [all_tags[i] for i in keep_idx]
        all_binary= [all_binary[i] for i in keep_idx]
        print(f"Subsampled None: {len(none_idx)} → {SUBSAMPLE_NONE}")

    # Build superfamily mapping
    # If checkpoint SF names are provided, use exactly those (reproduces training filter)
    if ckpt_sf_names is not None:
        keep_sfs = set(ckpt_sf_names)
        superfamily_names = sorted(keep_sfs)
    else:
        tpase_tags = [t for t, b in zip(all_tags, all_binary) if b == 1]
        tag_counts = Counter(tpase_tags)
        keep_sfs   = {t for t, c in tag_counts.items() if c >= MIN_CLASS_COUNT}
        superfamily_names = sorted(keep_sfs)
    superfamily_to_id = {t: i for i, t in enumerate(superfamily_names)}

    filtered_h, filtered_s, filtered_bin, filtered_cls = [], [], [], []
    for h, s, tag, binary in zip(all_h, all_s, all_tags, all_binary):
        if binary == 0:
            filtered_h.append(h); filtered_s.append(s)
            filtered_bin.append(0); filtered_cls.append(0)
        elif tag in superfamily_to_id:
            filtered_h.append(h); filtered_s.append(s)
            filtered_bin.append(1); filtered_cls.append(superfamily_to_id[tag])

    all_h  = filtered_h;  all_s    = filtered_s
    all_bin = np.array(filtered_bin, dtype=np.int64)
    all_cls = np.array(filtered_cls,  dtype=np.int64)
    del filtered_h, filtered_s, filtered_bin, filtered_cls
    gc.collect()

    print(f"Final: {len(all_h)} sequences | {(all_bin==1).sum()} tpase+ | {(all_bin==0).sum()} None")
    print(f"Superfamilies: {superfamily_names}")

    # Pre-compute k-mer features
    print("Pre-computing k-mer features …")
    featurizer = KmerWindowFeaturizer(
        k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW, stride=KMER_STRIDE,
        add_pos=True, l2_normalize=True
    )
    all_kmer = []
    for seq in tqdm(all_s, desc="Featurizing"):
        X, _ = featurizer.featurize_sequence(seq)
        all_kmer.append(X)

    # Train/test split
    idx_train, idx_test = train_test_split(
        np.arange(len(all_h)), test_size=TEST_SIZE,
        stratify=all_bin, random_state=RANDOM_STATE
    )
    test_h    = [all_h[i]  for i in idx_test]
    test_s    = [all_s[i]  for i in idx_test]
    test_bin  = all_bin[idx_test]
    test_cls  = all_cls[idx_test]
    test_kmer = [all_kmer[i] for i in idx_test]
    return test_h, test_s, test_bin, test_cls, test_kmer, superfamily_names, superfamily_to_id


def run():
    warnings.filterwarnings("ignore")

    if not CHECKPOINT.exists():
        sys.exit(f"Checkpoint not found: {CHECKPOINT}")

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    superfamily_names = ckpt.get("superfamily_names")
    arch = ckpt.get("arch", {})
    epoch = ckpt.get("epoch", "?")
    score = ckpt.get("score", "?")
    print(f"Checkpoint: epoch={epoch}, score={score}")
    print(f"Superfamilies from ckpt: {superfamily_names}")

    # Use checkpoint's SF names to reproduce the exact training filter/split
    ckpt_sf_to_id = ckpt.get("superfamily_to_id")  # may be None

    # Build test set using checkpoint SF names so the filtered pool matches training
    test_h, test_s, test_bin, test_cls, test_kmer, sf_names_data, sf_to_id = build_test_set(
        ckpt_sf_names=superfamily_names if superfamily_names is not None else None
    )

    # If checkpoint has sf names use those, else data-derived
    if superfamily_names is None:
        superfamily_names = sf_names_data
    n_sf = len(superfamily_names)

    # Build model
    model = HybridTEClassifierV4(
        num_superfamilies=n_sf,
        cnn_width=arch.get("cnn_width", CNN_WIDTH),
        motif_kernels=tuple(arch.get("motif_kernels", MOTIF_KERNELS)),
        context_dilations=tuple(arch.get("context_dilations", CONTEXT_DILATIONS)),
        rc_mode=arch.get("rc_mode", RC_FUSION_MODE),
        gnn_in_dim=KMER_DIM + 1,
        gnn_hidden=arch.get("gnn_hidden", GNN_HIDDEN),
        gnn_layers=arch.get("gnn_layers", GNN_LAYERS),
        fusion_dim=arch.get("fusion_dim", FUSION_DIM),
        num_heads=arch.get("num_heads", NUM_HEADS),
        dropout=arch.get("dropout", DROPOUT),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print("Model loaded.")

    # DataLoader
    ds = HybridDataset(test_h, test_s, test_bin, test_cls, test_kmer, FIXED_LENGTH)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_hybrid(b, fixed_length=FIXED_LENGTH)
    )

    all_bin_pred, all_bin_true = [], []
    all_sf_pred,  all_sf_true  = [], []

    with torch.no_grad():
        for _, X_cnn, mask, Y_bin, Y_sf, x_gnn, edge_index, batch_vec in tqdm(loader, desc="Eval"):
            X_cnn      = X_cnn.to(device)
            mask       = mask.to(device)
            x_gnn      = x_gnn.to(device)
            edge_index = edge_index.to(device)
            batch_vec  = batch_vec.to(device)

            bin_logits, sf_logits, _ = model(X_cnn, mask, x_gnn, edge_index, batch_vec)
            bin_pred = bin_logits.argmax(dim=1).cpu().numpy()
            sf_pred  = sf_logits.argmax(dim=1).cpu().numpy()

            all_bin_pred.extend(bin_pred)
            all_bin_true.extend(Y_bin.numpy())

            tpase_mask = Y_bin.numpy() == 1
            all_sf_pred.extend(sf_pred[tpase_mask])
            all_sf_true.extend(Y_sf.numpy()[tpase_mask])

    all_bin_pred = np.array(all_bin_pred)
    all_bin_true = np.array(all_bin_true)

    # ─── Fig 4: Confusion matrices ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Binary confusion matrix — CORRECTED LABELS
    ax1 = axes[0]
    cm_bin = confusion_matrix(all_bin_true, all_bin_pred)
    sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["non-DNA", "DNA transposon"],
                yticklabels=["non-DNA", "DNA transposon"])
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Binary Classification Confusion Matrix")
    ax1.set_box_aspect(1)

    # Superfamily confusion matrix (top 10 classes)
    ax2 = axes[1]
    all_sf_pred_a = np.array(all_sf_pred)
    all_sf_true_a = np.array(all_sf_true)
    cm_sf = confusion_matrix(all_sf_true_a, all_sf_pred_a,
                             labels=list(range(n_sf)))
    class_support = cm_sf.sum(axis=1)
    top_classes  = np.argsort(class_support)[::-1][:10]
    cm_sf_top    = cm_sf[np.ix_(top_classes, top_classes)]
    top_names    = [superfamily_names[i] for i in top_classes]
    sns.heatmap(cm_sf_top, annot=True, fmt="d", cmap="Blues", ax=ax2,
                xticklabels=top_names, yticklabels=top_names)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Superfamily Confusion Matrix (Top 10 Classes)")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_yticklabels(), rotation=0)
    ax2.set_box_aspect(1)

    plt.tight_layout()
    out4 = OUT_DIR / "v4_confusion.png"
    plt.savefig(str(out4), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out4}")

    # ─── Fig 5: Training curves (from checkpoint history) ──────────────────
    history = ckpt.get("history")
    if history is None:
        print("No training history in checkpoint — skipping Fig 5")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Hybrid V4 Training History (CNN + K-mer GNN)", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"],        "b-",  label="Total Loss",      linewidth=2)
    ax.plot(epochs, history["train_binary_loss"], "g--", label="Binary Loss",     linewidth=1.5)
    ax.plot(epochs, history["train_sf_loss"],     "r--", label="Superfamily Loss",linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Losses")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["val_binary_acc"], "b-", label="Accuracy", linewidth=2)
    ax.plot(epochs, history["val_binary_f1"],  "g-", label="F1 Score", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Binary Classification (DNA transposon vs non-DNA)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1])

    ax = axes[0, 2]
    ax.plot(epochs, history["val_sf_acc"], "b-", label="Accuracy",        linewidth=2)
    ax.plot(epochs, history["val_sf_f1"],  "g-", label="F1 Score (Macro)",linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Superfamily Classification")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1])

    ax = axes[1, 0]
    if "gate_weights_cnn" in history:
        ax.plot(epochs, history["gate_weights_cnn"], "b-", label="CNN Weight", linewidth=2)
        ax.plot(epochs, history["gate_weights_gnn"], "r-", label="GNN Weight", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gate Weight"); ax.set_title("Attention Fusion Weights")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1])

    ax = axes[1, 1]
    combined = [0.5 * b + 0.5 * s for b, s in zip(history["val_binary_f1"], history["val_sf_f1"])]
    ax.plot(epochs, combined, "purple", linewidth=2, label="Combined Score")
    best_ep = int(np.argmax(combined)) + 1
    ax.axvline(x=best_ep, color="red", linestyle="--", alpha=0.7, label=f"Best epoch {best_ep}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Combined Score (0.5·Binary + 0.5·SF)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1])

    axes[1, 2].axis("off")

    plt.tight_layout()
    out5 = OUT_DIR / "v4_training.png"
    plt.savefig(str(out5), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out5}")


if __name__ == "__main__":
    run()
