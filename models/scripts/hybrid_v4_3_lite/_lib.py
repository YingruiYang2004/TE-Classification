"""
hybrid_v4_3_lite shared library.

Contains all non-interactive code so the driver notebooks (01-03) stay short and
focus on visible, runnable progress. Mirrors notebook 01 for the architecture
and lifts the dataset / featurizer / loss helpers from the v4.3 trunk with the
slim-down + augmentation hooks added.

Lite vs v4.3:
  * CNN width 128 -> 64; kernels (7,15,21) -> (7,15); 4 dilated blocks -> 3.
  * 4 sinusoidal positional-encoding channels appended to one-hot input.
  * GNN hidden 128 -> 64; layers 3 -> 2; k-mer 7-mer/2048-d -> 6-mer/1024-d.
  * Fusion 256 -> 128; heads 4 -> 2.
  * Dropouts raised (CNN 0.30, GNN 0.25, fusion 0.30, head 0.30).
  * No boundary / segmentation heads.
  * Train-time augmentations: RC flip, canvas re-anchor, N-noise.
  * Embedding-level MixUp on the top-level head only.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# =============================================================================
# Defaults (mirror notebook 01 section 1)
# =============================================================================

FIXED_LENGTH = 20_000  # max sequence in dataset is 19,907 bp; 20k is lossless

CNN_WIDTH = 64
MOTIF_KERNELS: Tuple[int, ...] = (7, 15)
CONTEXT_KERNEL = 9
CONTEXT_DILATIONS: Tuple[int, ...] = (1, 2, 4)
POS_ENC_CHANNELS = 4
CNN_DROPOUT = 0.30

KMER_K = 6
KMER_DIM = 1024
KMER_WINDOW = 512
KMER_STRIDE = 256
GNN_HIDDEN = 64
GNN_LAYERS = 2
GNN_DROPOUT = 0.25
GNN_IN_DIM = KMER_DIM + 1

FUSION_DIM = 128
NUM_HEADS = 2
FUSION_DROPOUT = 0.30
HEAD_DROPOUT = 0.30


# =============================================================================
# Device helper
# =============================================================================

def resolve_device(requested: Optional[str] = None) -> torch.device:
    import os
    if requested is None:
        requested = os.environ.get('FORCE_DEVICE')  # e.g. 'cpu' for overnight CPU runs
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# =============================================================================
# FASTA + label loading
# =============================================================================

def read_fasta(path: str | Path) -> Tuple[List[str], List[str]]:
    headers: List[str] = []
    sequences: List[str] = []
    h: Optional[str] = None
    buf: List[str] = []
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


def load_labels(
    label_path: str | Path,
    keep_classes: Optional[Sequence[str]] = ('DNA', 'LTR', 'LINE'),
) -> Tuple[dict, dict, dict]:
    """Returns (header -> superfamily-tag, header -> top-level class id, class -> id).

    If `keep_classes` is None, every top-level class encountered is kept and
    assigned an id in first-seen order.
    """
    keep_all = keep_classes is None
    class_to_id: dict = {} if keep_all else {c: i for i, c in enumerate(keep_classes)}
    label_dict: dict = {}
    class_dict: dict = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            header = parts[0].lstrip('>')
            tag = parts[1]
            top_class = tag.split('/')[0]
            if keep_all:
                if top_class not in class_to_id:
                    class_to_id[top_class] = len(class_to_id)
                label_dict[header] = tag
                class_dict[header] = class_to_id[top_class]
            elif top_class in class_to_id:
                label_dict[header] = tag
                class_dict[header] = class_to_id[top_class]
    return label_dict, class_dict, class_to_id


def extract_genome_id(header: str) -> str:
    """`hAT_1-aAnoBae#DNA/hAT` -> `aAnoBae`."""
    return header.split('#')[0].rsplit('-', 1)[-1]


def compute_class_weights(y_ids: np.ndarray, n_classes: int, mode: str = 'inv_sqrt') -> np.ndarray:
    counts = np.bincount(np.asarray(y_ids, dtype=np.int64), minlength=n_classes).astype(np.float64)
    eps = 1e-6
    if mode == 'none':
        w = np.ones(n_classes, dtype=np.float32)
    elif mode == 'inv':
        w = 1.0 / (counts + eps)
    elif mode == 'inv_sqrt':
        w = 1.0 / np.sqrt(counts + eps)
    else:
        raise ValueError(f'unknown mode={mode}')
    w = w / (w.mean() + 1e-12)
    return w.astype(np.float32)


# =============================================================================
# K-mer featurizer (k=6, dim=1024)
# =============================================================================

_ASCII_MAP = np.full(256, 4, dtype=np.uint8)
for _ch, _val in [('A', 0), ('C', 1), ('G', 2), ('T', 3),
                  ('a', 0), ('c', 1), ('g', 2), ('t', 3)]:
    _ASCII_MAP[ord(_ch)] = _val
_COMP = np.array([3, 2, 1, 0], dtype=np.uint8)


def _kmer_code_forward(arr4: np.ndarray) -> int:
    code = 0
    for v in arr4:
        code = (code << 2) | int(v)
    return code


def _kmer_code_rc(arr4: np.ndarray) -> int:
    code = 0
    for v in arr4[::-1]:
        code = (code << 2) | int(_COMP[v])
    return code


def _canonical_kmer_code(arr4: np.ndarray) -> int:
    c1 = _kmer_code_forward(arr4)
    c2 = _kmer_code_rc(arr4)
    return c1 if c1 < c2 else c2


def _hash_u32(x: int, dim: int) -> int:
    z = (x * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z ^= (z >> 33)
    z = (z * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF
    z ^= (z >> 29)
    return int(z % dim)


@dataclass
class KmerWindowFeaturizer:
    k: int = KMER_K
    dim: int = KMER_DIM
    window: int = KMER_WINDOW
    stride: int = KMER_STRIDE
    add_pos: bool = True
    l2_normalize: bool = True

    def featurize_sequence(self, seq: str) -> np.ndarray:
        arr = _ASCII_MAP[np.frombuffer(seq.encode('ascii', 'ignore'), dtype=np.uint8)]
        L = int(arr.size)
        out_dim = self.dim + (1 if self.add_pos else 0)
        if L == 0:
            return np.zeros((1, out_dim), dtype=np.float32)
        if L <= self.window:
            starts = np.array([0], dtype=np.int64)
        else:
            starts = np.arange(0, L - self.window + 1, self.stride, dtype=np.int64)
            if starts.size == 0:
                starts = np.array([0], dtype=np.int64)
        X = np.zeros((starts.size, out_dim), dtype=np.float32)
        k = self.k
        for wi, st in enumerate(starts):
            en = min(st + self.window, L)
            sub = arr[st:en]
            counts = np.zeros(self.dim, dtype=np.float32)
            total = 0
            if sub.size >= k:
                for i in range(0, sub.size - k + 1):
                    kmer = sub[i:i + k]
                    if np.any(kmer == 4):
                        continue
                    code = _canonical_kmer_code(kmer)
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
        return X


# =============================================================================
# Dataset / collate
# =============================================================================

ENCODE = np.full(256, 4, dtype=np.int64)
for _ch, _idx in zip(b'ACGTNacgtn', [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]):
    ENCODE[_ch] = _idx

REV_COMP = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)
_COMP_T = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)  # ACGT->TGCA, N->N


def _build_chain_edge_index(n: int) -> torch.Tensor:
    if n <= 0:
        return torch.zeros((2, 0), dtype=torch.long)
    if n == 1:
        idx = np.array([[0], [0]], dtype=np.int64)
        return torch.from_numpy(idx)
    src = np.arange(n - 1, dtype=np.int64)
    dst = np.arange(1, n, dtype=np.int64)
    self_loops = np.arange(n, dtype=np.int64)
    s = np.concatenate([src, dst, self_loops])
    d = np.concatenate([dst, src, self_loops])
    return torch.from_numpy(np.stack([s, d], axis=0))


@dataclass
class AugmentConfig:
    """Training-time augmentation toggle. Set `enabled=False` for val/test."""
    enabled: bool = True
    p_rc_flip: float = 0.5
    p_n_noise: float = 0.5
    n_noise_frac: float = 0.01     # fraction of bases per sequence to randomly mark as N


class HybridDataset(Dataset):
    """CNN one-hot + pre-computed GNN k-mer features.

    Random canvas placement is *always* applied at train time (matches v4.3).
    With `augment.enabled=True`, additionally:
      * with prob p_rc_flip, flip both the encoded sequence and the kmer feats;
      * with prob p_n_noise, randomly tag a small fraction of bases as N.
    """
    def __init__(
        self,
        headers: List[str],
        sequences: List[str],
        toplevel_labels: np.ndarray,
        sf_labels: np.ndarray,
        kmer_features: List[np.ndarray],
        fixed_length: int = FIXED_LENGTH,
        augment: Optional[AugmentConfig] = None,
        rng_seed: Optional[int] = None,
    ):
        self.headers = list(headers)
        self.sequences = list(sequences)
        self.toplevel_labels = np.asarray(toplevel_labels, dtype=np.int64)
        self.sf_labels = np.asarray(sf_labels, dtype=np.int64)
        self.kmer_features = kmer_features
        self.fixed_length = fixed_length
        self.augment = augment if augment is not None else AugmentConfig(enabled=False)
        self._rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        seq_len = len(seq)
        seq_idx = ENCODE[np.frombuffer(seq.encode('ascii', 'ignore'), dtype=np.uint8)].copy()
        kmer_feat = self.kmer_features[idx]

        rc_flipped = False
        if self.augment.enabled and seq_len > 0:
            # 1) RC flip (sequence + kmer windows mirrored)
            if self._rng.random() < self.augment.p_rc_flip:
                # complement: A<->T, C<->G; N stays N
                comp_lut = np.array([3, 2, 1, 0, 4], dtype=np.int64)
                seq_idx = comp_lut[seq_idx][::-1].copy()
                # mirror window order; per-window L2-normalised kmer counts are
                # canonical (palindrome-safe) so the column block is unchanged,
                # but the position channel must flip 1-x.
                if kmer_feat.size > 0:
                    kmer_feat = kmer_feat[::-1].copy()
                    if kmer_feat.shape[-1] >= 1:  # last col is normalised position
                        kmer_feat[:, -1] = 1.0 - kmer_feat[:, -1]
                rc_flipped = True
            # 2) N-noise: tag a small fraction of valid bases as N (idx=4)
            if self._rng.random() < self.augment.p_n_noise:
                n_to_corrupt = max(1, int(self.augment.n_noise_frac * seq_len))
                pos = self._rng.integers(0, seq_len, size=n_to_corrupt)
                seq_idx[pos] = 4

        # Canvas placement (random anchor; matches v4.3)
        max_start = max(0, self.fixed_length - seq_len)
        start_pos = int(self._rng.integers(0, max_start + 1)) if max_start > 0 else 0

        return (
            self.headers[idx],
            seq_idx,
            int(self.toplevel_labels[idx]),
            int(self.sf_labels[idx]),
            start_pos,
            seq_len,
            kmer_feat,
            rc_flipped,
        )


def collate_hybrid(batch, fixed_length: int = FIXED_LENGTH):
    headers, seq_idxs, top_labels, sf_labels, starts, lengths, kmer_feats, _rc = zip(*batch)
    B = len(batch)

    X_cnn = torch.zeros((B, 5, fixed_length), dtype=torch.float32)
    mask = torch.zeros((B, fixed_length), dtype=torch.bool)
    for i, (seq_idx, start, seq_len) in enumerate(zip(seq_idxs, starts, lengths)):
        actual_len = min(seq_len, fixed_length - start)
        if actual_len > 0:
            idx = torch.from_numpy(seq_idx[:actual_len].astype(np.int64))
            pos = torch.arange(actual_len, dtype=torch.long) + start
            X_cnn[i, idx, pos] = 1.0
            mask[i, start:start + actual_len] = (idx != 4)

    Y_top = torch.tensor(top_labels, dtype=torch.long)
    Y_sf = torch.tensor(sf_labels, dtype=torch.long)

    xs, eis, batch_vecs = [], [], []
    node_offset = 0
    for gi, kmer_feat in enumerate(kmer_feats):
        x = torch.from_numpy(np.ascontiguousarray(kmer_feat)).to(torch.float32)
        n = x.size(0)
        ei = _build_chain_edge_index(n)
        xs.append(x)
        eis.append(ei + node_offset)
        batch_vecs.append(torch.full((n,), gi, dtype=torch.int64))
        node_offset += n
    x_gnn = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.int64)
    batch_vec = torch.cat(batch_vecs, dim=0)

    return list(headers), X_cnn, mask, Y_top, Y_sf, x_gnn, edge_index, batch_vec


# =============================================================================
# Model (mirrors notebook 01 sections 3-7; kept here so notebooks 02-03 can import)
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 9, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation, bias=True)
        self.bn = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout1d(dropout)
        self.proj = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.bn(self.conv(x)))
        y = self.drop(y)
        return y + self.proj(x)


class MaskedMaxPool1d(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Match v4.3's MaskedMaxPool1d exactly (which trained 40 epochs without
        # NaN on this same machine). Earlier we tried (a) a smaller sentinel
        # and (b) zero-masking the pool output; (b) poisons BatchNorm running
        # stats because ~75% of positions become exact zeros, so eval() mode
        # normalises real activations by a near-zero variance and produces NaN.
        if mask is not None:
            m = mask.unsqueeze(1).float()
            x = x * m + (~mask.unsqueeze(1)) * (-1e9)
        x_p = F.max_pool1d(x, self.kernel_size, self.stride)
        if mask is None:
            return x_p, None
        m_p = F.max_pool1d(mask.float().unsqueeze(1), self.kernel_size, self.stride).squeeze(1) > 0
        return x_p, m_p


def _masked_avg_pool(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return z.mean(-1)
    m = mask.unsqueeze(1).float()
    return (z * m).sum(-1) / m.sum(-1).clamp_min(1.0)


class CNNTowerLite(nn.Module):
    def __init__(
        self,
        width: int = CNN_WIDTH,
        motif_kernels: Tuple[int, ...] = MOTIF_KERNELS,
        context_kernel: int = CONTEXT_KERNEL,
        context_dilations: Tuple[int, ...] = CONTEXT_DILATIONS,
        pos_enc_channels: int = POS_ENC_CHANNELS,
        dropout: float = CNN_DROPOUT,
    ):
        super().__init__()
        self.out_dim = width
        self.pos_enc_channels = pos_enc_channels
        in_ch = 5 + pos_enc_channels
        self.motif_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, width, kernel_size=k, padding=k // 2, bias=True),
                nn.BatchNorm1d(width),
                nn.GELU(),
                nn.Dropout1d(dropout),
            )
            for k in motif_kernels
        ])
        self.mix = nn.Sequential(
            nn.Conv1d(width * len(motif_kernels), width, kernel_size=1, bias=True),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout1d(dropout),
        )
        self.context_blocks = nn.ModuleList([
            ConvBlock(width, width, kernel_size=context_kernel, dilation=d, dropout=dropout)
            for d in context_dilations
        ])
        self.pool = MaskedMaxPool1d(kernel_size=2, stride=2)
        self.register_buffer('_pe_cache', torch.zeros(0), persistent=False)

    def _build_pos_enc(self, length: int, device: torch.device) -> torch.Tensor:
        pe_dim = self.pos_enc_channels
        pos = torch.linspace(0.0, 1.0, length, device=device)
        freqs = (2.0 ** torch.arange(pe_dim // 2, device=device)) * math.pi
        ang = pos.unsqueeze(0) * freqs.unsqueeze(1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=0)

    def _add_pos_enc(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_enc_channels == 0:
            return x
        B, _, L = x.shape
        if (
            self._pe_cache.numel() == 0
            or self._pe_cache.shape[-1] != L
            or self._pe_cache.device != x.device
        ):
            self._pe_cache = self._build_pos_enc(L, x.device)
        return torch.cat([x, self._pe_cache.unsqueeze(0).expand(B, -1, -1)], dim=1)

    @staticmethod
    def rc_transform(x: torch.Tensor, mask: torch.Tensor):
        x_rc = x.index_select(1, REV_COMP.to(x.device)).flip(-1)
        mask_rc = None if mask is None else mask.flip(-1)
        return x_rc, mask_rc

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self._add_pos_enc(x)
        feats = [conv(x) for conv in self.motif_convs]
        z = self.mix(torch.cat(feats, dim=1))
        m = mask
        for block in self.context_blocks:
            z = block(z)
            z, m = self.pool(z, m)
        return _masked_avg_pool(z, m)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        f = self.encode(x, mask)
        x_rc, mask_rc = self.rc_transform(x, mask)
        r = self.encode(x_rc, mask_rc)
        return 0.5 * (f + r)


def _scatter_mean(x: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, x.size(1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, idx, x)
    # MPS-safe count (avoid bincount).
    ones = torch.ones(idx.size(0), device=x.device, dtype=x.dtype)
    cnt = torch.zeros(dim_size, device=x.device, dtype=x.dtype)
    cnt.index_add_(0, idx, ones)
    cnt = cnt.clamp_min(1).unsqueeze(1)
    return out / cnt


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        # MPS-safe degree count (bincount is not supported on MPS).
        ones = torch.ones(dst.size(0), device=x.device, dtype=x.dtype)
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, ones)
        deg = deg.clamp_min(1).unsqueeze(1)
        agg = agg / deg
        h = self.lin_self(x) + self.lin_neigh(agg)
        return self.dropout(F.relu(h))


class GNNTowerLite(nn.Module):
    def __init__(
        self,
        in_dim: int = GNN_IN_DIM,
        hidden: int = GNN_HIDDEN,
        n_layers: int = GNN_LAYERS,
        dropout: float = GNN_DROPOUT,
    ):
        super().__init__()
        self.out_dim = hidden
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(GraphSAGELayer(d, hidden, dropout=dropout))
            d = hidden
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_vec: torch.Tensor,
                batch_size: Optional[int] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        if batch_size is None:
            # Fall back to inferring from batch_vec; do it on CPU to avoid MPS int64 quirks.
            B = int(batch_vec.detach().to('cpu').max().item()) + 1 if batch_vec.numel() else 0
        else:
            B = int(batch_size)
        return _scatter_mean(x, batch_vec, dim_size=B)


class CrossModalAttentionFusionLite(nn.Module):
    def __init__(
        self,
        cnn_dim: int = CNN_WIDTH,
        gnn_dim: int = GNN_HIDDEN,
        fusion_dim: int = FUSION_DIM,
        num_heads: int = NUM_HEADS,
        dropout: float = FUSION_DROPOUT,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
        self.ln1 = nn.LayerNorm(fusion_dim)
        self.ln2 = nn.LayerNorm(fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=-1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, cnn_embed: torch.Tensor, gnn_embed: torch.Tensor):
        c = self.ln1(self.cnn_proj(cnn_embed))
        g = self.ln2(self.gnn_proj(gnn_embed))
        combined = torch.stack([c, g], dim=1)
        attn_out, _ = self.cross_attn(combined, combined, combined)
        c_attn, g_attn = attn_out[:, 0], attn_out[:, 1]
        gate_weights = self.gate(torch.cat([c_attn, g_attn], dim=-1))
        fused = gate_weights[:, 0:1] * c_attn + gate_weights[:, 1:2] * g_attn
        return self.out_proj(fused), gate_weights


class HybridTEClassifierV43Lite(nn.Module):
    def __init__(
        self,
        num_toplevel: int,
        num_superfamilies: int,
        cnn_width: int = CNN_WIDTH,
        motif_kernels: Tuple[int, ...] = MOTIF_KERNELS,
        context_dilations: Tuple[int, ...] = CONTEXT_DILATIONS,
        pos_enc_channels: int = POS_ENC_CHANNELS,
        cnn_dropout: float = CNN_DROPOUT,
        gnn_in_dim: int = GNN_IN_DIM,
        gnn_hidden: int = GNN_HIDDEN,
        gnn_layers: int = GNN_LAYERS,
        gnn_dropout: float = GNN_DROPOUT,
        fusion_dim: int = FUSION_DIM,
        num_heads: int = NUM_HEADS,
        fusion_dropout: float = FUSION_DROPOUT,
        head_dropout: float = HEAD_DROPOUT,
    ):
        super().__init__()
        self.num_toplevel = num_toplevel
        self.num_superfamilies = num_superfamilies
        self.cnn_tower = CNNTowerLite(
            width=cnn_width, motif_kernels=motif_kernels,
            context_dilations=context_dilations, pos_enc_channels=pos_enc_channels,
            dropout=cnn_dropout,
        )
        self.gnn_tower = GNNTowerLite(
            in_dim=gnn_in_dim, hidden=gnn_hidden,
            n_layers=gnn_layers, dropout=gnn_dropout,
        )
        self.fusion = CrossModalAttentionFusionLite(
            cnn_dim=cnn_width, gnn_dim=gnn_hidden,
            fusion_dim=fusion_dim, num_heads=num_heads, dropout=fusion_dropout,
        )
        self.toplevel_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim), nn.GELU(),
            nn.Dropout(head_dropout), nn.Linear(fusion_dim, num_toplevel),
        )
        self.superfamily_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim), nn.GELU(),
            nn.Dropout(head_dropout), nn.Linear(fusion_dim, num_superfamilies),
        )

    def encode(self, x_cnn, mask, x_gnn, edge_index, batch_vec):
        c = self.cnn_tower(x_cnn, mask)
        # Pass explicit batch size = B from CNN input to dodge MPS int64 .max() bugs.
        g = self.gnn_tower(x_gnn, edge_index, batch_vec, batch_size=x_cnn.size(0))
        fused, gate_weights = self.fusion(c, g)
        return fused, gate_weights

    def forward(self, x_cnn, mask, x_gnn, edge_index, batch_vec,
                mixup_lam: Optional[float] = None,
                mixup_perm: Optional[torch.Tensor] = None):
        fused, gate_weights = self.encode(x_cnn, mask, x_gnn, edge_index, batch_vec)
        if mixup_lam is not None and mixup_perm is not None:
            fused_mix = mixup_lam * fused + (1.0 - mixup_lam) * fused[mixup_perm]
            top_logits = self.toplevel_head(fused_mix)
        else:
            top_logits = self.toplevel_head(fused)
        sf_logits = self.superfamily_head(fused)
        return top_logits, sf_logits, gate_weights


# =============================================================================
# Loss
# =============================================================================

class LabelSmoothingCE(nn.Module):
    """Weighted cross entropy with label smoothing."""
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer('weight', weight if weight is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(-1)
        log_p = F.log_softmax(logits, dim=-1)
        oh = torch.zeros_like(log_p).scatter_(1, targets.unsqueeze(1), 1)
        smooth = (1 - self.smoothing) * oh + self.smoothing / n
        if self.weight.numel() > 0:
            w = self.weight[targets].unsqueeze(1)
            return -(smooth * log_p * w).sum(dim=-1).mean()
        return -(smooth * log_p).sum(dim=-1).mean()


# =============================================================================
# Sample-prep helpers used by the training notebook
# =============================================================================

def filter_and_subsample(
    headers: List[str],
    sequences: List[str],
    label_dict: dict,
    class_dict: dict,
    *,
    exclude_genomes: Optional[Iterable[str]] = None,
    min_class_count: int = 100,
    max_per_sf: Optional[int] = 3000,
    random_state: int = 42,
):
    """Apply v4.3-style filtering: genome-exclusion, min-count SF filter, per-SF cap."""
    excl = set(exclude_genomes) if exclude_genomes else set()
    rng = np.random.default_rng(random_state)

    h_keep, s_keep, tags, top = [], [], [], []
    n_exc = 0
    for h, s in zip(headers, sequences):
        if h not in label_dict:
            continue
        if excl and extract_genome_id(h) in excl:
            n_exc += 1
            continue
        h_keep.append(h)
        s_keep.append(s)
        tags.append(label_dict[h])
        top.append(class_dict[h])

    counts = Counter(tags)
    sf_names = sorted([t for t, c in counts.items() if c >= min_class_count])
    sf_to_id = {t: i for i, t in enumerate(sf_names)}

    h2, s2, tag2, top2, sf2 = [], [], [], [], []
    for h, s, t, tp in zip(h_keep, s_keep, tags, top):
        if t in sf_to_id:
            h2.append(h); s2.append(s); tag2.append(t); top2.append(tp); sf2.append(sf_to_id[t])
    top2 = np.array(top2, dtype=np.int64)
    sf2 = np.array(sf2, dtype=np.int64)

    if max_per_sf is not None:
        keep_idx = []
        for sf_name in sf_names:
            sf_id = sf_to_id[sf_name]
            idxs = np.where(sf2 == sf_id)[0]
            if len(idxs) > max_per_sf:
                idxs = rng.choice(idxs, max_per_sf, replace=False)
            keep_idx.extend(idxs.tolist())
        keep_idx = sorted(keep_idx)
        h2 = [h2[i] for i in keep_idx]
        s2 = [s2[i] for i in keep_idx]
        tag2 = [tag2[i] for i in keep_idx]
        top2 = top2[keep_idx]
        sf2 = sf2[keep_idx]

    return {
        'headers': h2,
        'sequences': s2,
        'tags': tag2,
        'toplevel': top2,
        'sf': sf2,
        'sf_names': sf_names,
        'sf_to_id': sf_to_id,
        'n_excluded_genomes': n_exc,
    }
