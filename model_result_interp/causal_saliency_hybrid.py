"""Hybrid V4.3 (CNN+GNN+attention) variant of causal_saliency.

Mirrors `causal_saliency.py` API but for the hybrid checkpoint
`data_analysis/vgp_model_data_tpase_multi/v4.3/hybrid_v4.3_epoch40.pt`
(superfamily F1 ~ 0.99 on validation).

Architecture (from `models/train_hybrid_v5.py` + checkpoint state_dict shapes):
  CNN tower: motif_convs (k=7,15,21) -> mix -> 4x dilated ConvBlocks -> masked avg pool.
  GNN tower: 3-layer GraphSAGE on chain-graph of k-mer windows.
  Fusion: cross-modal attention + softmax gate over CNN/GNN -> 256-D embedding.
  Two heads (this v4.3 variant has both, NOT the unified V5 head):
    class_head      : 256 -> 128 -> 3   (DNA/LTR/LINE)
    superfamily_head: 256 -> 256 -> 23  (DNA/hAT etc.)
The thesis Section 3.6 saliency story is at the superfamily level
(LTR/Gypsy <-> LTR/Pao etc.), so attribution defaults to the superfamily head.

Critical correctness note: occluding the input changes the k-mer window
features. The GNN branch must be re-featurised on every perturbed sequence,
otherwise occlusion measures only the CNN branch (which would invalidate the
"intervention" interpretation). All occlusion functions in this module
re-featurise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FIXED_LENGTH = 20000  # v4.3 canvas

# Sequence base index lookup (separate from the v3 module so we can keep
# this module standalone). 0..3 = ACGT, 4 = N/anything else.
_ASCII_MAP_5 = np.full(256, 4, dtype=np.int64)
for ch, idx in zip(b"ACGTNacgtn", [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]):
    _ASCII_MAP_5[ch] = idx
ENCODE = _ASCII_MAP_5  # alias for readability

# k-mer featurizer uses 4-symbol alphabet (no N). Treat N as 4 for masking.
_ASCII_MAP_4 = np.full(256, 4, dtype=np.uint8)
for ch, val in [("A", 0), ("C", 1), ("G", 2), ("T", 3),
                ("a", 0), ("c", 1), ("g", 2), ("t", 3)]:
    _ASCII_MAP_4[ord(ch)] = val

REV_COMP = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# CNN tower (must match training)
# ============================================================================


class _ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=9, dilation=1, dropout=0.1):
        super().__init__()
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


class _MaskedMaxPool1d(nn.Module):
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


def _masked_avg_pool(z, mask):
    if mask is None:
        return z.mean(-1)
    m = mask.unsqueeze(1).float()
    return (z * m).sum(-1) / m.sum(-1).clamp_min(1.0)


class CNNTower(nn.Module):
    def __init__(self, width=128, motif_kernels=(7, 15, 21),
                 context_dilations=(1, 2, 4, 8), dropout=0.15, rc_mode="late",
                 context_kernel=9):
        super().__init__()
        self.rc_mode = rc_mode
        self.motif_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(5, width, kernel_size=k, padding=k // 2, bias=True),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for k in motif_kernels
            ]
        )
        in_ch = width * len(motif_kernels)
        self.mix = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=1, bias=True),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_blocks = nn.ModuleList(
            [_ConvBlock(width, width, kernel_size=context_kernel, dilation=d, dropout=dropout)
             for d in context_dilations]
        )
        self.pool = _MaskedMaxPool1d(2, 2)
        self.out_dim = width

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
        return _masked_avg_pool(z, m)

    def forward(self, x, mask):
        if self.rc_mode == "late":
            f = self.encode(x, mask)
            x_rc, mask_rc = self.rc_transform(x, mask)
            r = self.encode(x_rc, mask_rc)
            return 0.5 * (f + r)
        return self.encode(x, mask)


# ============================================================================
# GNN tower
# ============================================================================


def _scatter_mean(x, idx, dim_size):
    out = torch.zeros((dim_size, x.size(1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, idx, x)
    cnt = torch.bincount(idx, minlength=dim_size).clamp_min(1).to(x.device).to(x.dtype).unsqueeze(1)
    return out / cnt


class _GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

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
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.layers.append(_GraphSAGELayer(d, hidden, dropout=dropout))
            d = hidden

    def forward(self, x, edge_index, batch_vec):
        for lyr in self.layers:
            x = lyr(x, edge_index)
        B = int(batch_vec.max().item()) + 1 if batch_vec.numel() else 0
        return _scatter_mean(x, batch_vec, dim_size=B)


# ============================================================================
# Cross-modal attention fusion
# ============================================================================


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, cnn_dim=128, gnn_dim=128, fusion_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
        self.ln1 = nn.LayerNorm(fusion_dim)
        self.ln2 = nn.LayerNorm(fusion_dim)
        self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads=num_heads,
                                                dropout=dropout, batch_first=True)
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

    def forward(self, cnn_embed, gnn_embed):
        c = self.ln1(self.cnn_proj(cnn_embed))
        g = self.ln2(self.gnn_proj(gnn_embed))
        combined = torch.stack([c, g], dim=1)
        attn_out, _ = self.cross_attn(combined, combined, combined)
        c_attn = attn_out[:, 0]
        g_attn = attn_out[:, 1]
        gate_in = torch.cat([c_attn, g_attn], dim=-1)
        gate_w = self.gate(gate_in)
        fused = gate_w[:, 0:1] * c_attn + gate_w[:, 1:2] * g_attn
        return self.out_proj(fused), gate_w


# ============================================================================
# Hybrid V4.3 model (two heads)
# ============================================================================


class HybridV43(nn.Module):
    """V4.3 hybrid with separate class (3-way) and superfamily (23-way) heads.

    Forward returns (class_logits, sf_logits).  No binary head, no aux ops --
    we only need the prediction path for attribution.
    """
    def __init__(self, num_classes=3, num_superfamilies=23,
                 cnn_width=128, motif_kernels=(7, 15, 21),
                 context_dilations=(1, 2, 4, 8),
                 gnn_in_dim=2049, gnn_hidden=128, gnn_layers=3,
                 fusion_dim=256, num_heads=4, rc_mode="late", dropout=0.15):
        super().__init__()
        self.fixed_length = FIXED_LENGTH
        self.cnn_tower = CNNTower(width=cnn_width, motif_kernels=motif_kernels,
                                  context_dilations=context_dilations,
                                  dropout=dropout, rc_mode=rc_mode)
        self.gnn_tower = GNNTower(in_dim=gnn_in_dim, hidden=gnn_hidden,
                                  n_layers=gnn_layers, dropout=dropout)
        self.fusion = CrossModalAttentionFusion(
            cnn_dim=cnn_width, gnn_dim=gnn_hidden,
            fusion_dim=fusion_dim, num_heads=num_heads, dropout=dropout,
        )
        # class head: 256 -> 128 -> 3 (matches v4.3 checkpoint shapes)
        self.class_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        # superfamily head: 256 -> 256 -> 23
        self.superfamily_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_superfamilies),
        )

    def forward(self, x_cnn, mask, x_gnn, edge_index, batch_vec):
        cnn_e = self.cnn_tower(x_cnn, mask)
        gnn_e = self.gnn_tower(x_gnn, edge_index, batch_vec)
        fused, _ = self.fusion(cnn_e, gnn_e)
        return self.class_head(fused), self.superfamily_head(fused)


def load_hybrid_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    sf_names = list(ckpt["superfamily_names"])
    class_names = list(arch.get("class_names", ["DNA", "LTR", "LINE"]))
    model = HybridV43(
        num_classes=int(arch.get("num_classes", 3)),
        num_superfamilies=int(arch.get("num_superfamilies", len(sf_names))),
        cnn_width=int(arch.get("cnn_width", 128)),
        motif_kernels=tuple(arch.get("motif_kernels", (7, 15, 21))),
        context_dilations=tuple(arch.get("context_dilations", (1, 2, 4, 8))),
        gnn_in_dim=int(arch.get("gnn_in_dim", 2049)),
        gnn_hidden=int(arch.get("gnn_hidden", 128)),
        gnn_layers=int(arch.get("gnn_layers", 3)),
        fusion_dim=int(arch.get("fusion_dim", 256)),
        num_heads=int(arch.get("num_heads", 4)),
        rc_mode=str(arch.get("rc_mode", "late")),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        # Only complain about non-trivial mismatches (boundary head etc. in old ckpts).
        bad_missing = [k for k in missing if "boundary" not in k]
        bad_unexpected = [k for k in unexpected if "boundary" not in k]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"state_dict mismatch: missing={bad_missing}, unexpected={bad_unexpected}"
            )
    model.eval()
    return model, class_names, sf_names


# ============================================================================
# Vectorised k-mer featurizer (matches train_hybrid_v5.KmerWindowFeaturizer)
# ============================================================================


@dataclass
class KmerFeaturizer:
    k: int = 7
    dim: int = 2048
    window: int = 512
    stride: int = 256
    add_pos: bool = True
    l2_normalize: bool = True

    def featurise_4(self, arr4: np.ndarray) -> np.ndarray:
        """Featurise a uint8 array (values 0..3 = ACGT, 4 = N) into a
        (n_windows, dim+1) float32 matrix. Vectorised: O(L) per window in numpy.
        """
        L = int(arr4.size)
        if L == 0:
            return np.zeros((1, self.dim + (1 if self.add_pos else 0)), dtype=np.float32)
        if L <= self.window:
            starts = np.array([0], dtype=np.int64)
        else:
            starts = np.arange(0, L - self.window + 1, self.stride, dtype=np.int64)
            if starts.size == 0:
                starts = np.array([0], dtype=np.int64)
        out_dim = self.dim + (1 if self.add_pos else 0)
        X = np.zeros((starts.size, out_dim), dtype=np.float32)

        # Precompute per-position k-mer codes for the WHOLE sequence.
        codes_fwd, codes_rc, valid = _kmer_codes_vectorised(arr4, self.k)
        canon = np.minimum(codes_fwd, codes_rc)
        # SplitMix64-style hash modulo dim, vectorised.
        hashed = _hash_codes(canon, self.dim)

        for wi, st in enumerate(starts):
            en = min(st + self.window, L)
            # k-mer positions inside [st, en - k + 1)
            kmer_lo = st
            kmer_hi = en - self.k + 1
            if kmer_hi <= kmer_lo:
                continue
            mask_w = valid[kmer_lo:kmer_hi]
            h_w = hashed[kmer_lo:kmer_hi]
            if not mask_w.any():
                continue
            h_v = h_w[mask_w]
            counts = np.bincount(h_v, minlength=self.dim).astype(np.float32)
            total = counts.sum()
            if total > 0:
                counts /= total
            if self.l2_normalize:
                nrm = np.linalg.norm(counts)
                if nrm > 0:
                    counts /= nrm
            if self.add_pos:
                center = (st + en) / 2.0
                X[wi, :-1] = counts
                X[wi, -1] = center / max(1.0, float(L))
            else:
                X[wi, :] = counts
        return X


def _kmer_codes_vectorised(arr4: np.ndarray, k: int):
    """Return (codes_fwd, codes_rc, valid) over each k-mer starting at i in
    range(0, L-k+1).
    A k-mer is invalid if it contains an N (value 4).
    """
    L = int(arr4.size)
    n = max(0, L - k + 1)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, bool)
    a = arr4.astype(np.int64)
    a_clip = np.minimum(a, 3)  # don't let N propagate bit-level garbage
    # Forward k-mer code: sum_{j} base[i+j] * 4**(k-1-j)
    powers = (4 ** np.arange(k - 1, -1, -1, dtype=np.int64))
    codes_fwd = np.zeros(n, dtype=np.int64)
    for j in range(k):
        codes_fwd += a_clip[j : j + n] * powers[j]
    # RC code: complement (3 - base) and reverse the order in the k-mer.
    a_comp = 3 - a_clip
    # rev order: bases at positions (i+k-1, i+k-2, ..., i) get powers[0..k-1].
    codes_rc = np.zeros(n, dtype=np.int64)
    for j in range(k):
        # power for the bit position in the rc: when reading rc left-to-right,
        # the original index k-1-j sits in the high bit.
        codes_rc += a_comp[(k - 1 - j) : (k - 1 - j) + n] * powers[j]
    # Validity: no N in the window.
    is_n = (arr4 == 4)
    # cumulative count of N up to index i (exclusive)
    cum_n = np.concatenate(([0], np.cumsum(is_n.astype(np.int64))))
    valid = (cum_n[k:] - cum_n[:-k]) == 0
    valid = valid[: n]
    return codes_fwd, codes_rc, valid


def _hash_codes(codes: np.ndarray, dim: int) -> np.ndarray:
    """Match train_hybrid_v5.hash_u32 (SplitMix64 + mod), vectorised."""
    M1 = np.uint64(0x9E3779B97F4A7C15)
    M2 = np.uint64(0xC2B2AE3D27D4EB4F)
    z = codes.astype(np.uint64) * M1
    z ^= z >> np.uint64(33)
    z = z * M2
    z ^= z >> np.uint64(29)
    return (z % np.uint64(dim)).astype(np.int64)


# ============================================================================
# Encoded sequence + canvas placement
# ============================================================================


@dataclass
class EncodedHybrid:
    base_idx: np.ndarray  # (FIXED_LENGTH,) int64 in 0..4
    seq_arr4: np.ndarray  # (seq_len,) uint8 in 0..4 (the *real* sequence portion)
    start: int
    end: int
    header: str
    label_seq_len: int  # used for k-mer position normalisation; matches L during training

    @property
    def length(self) -> int:
        return self.end - self.start


def encode_sequence(seq: str, header: str = "", fixed_length: int = FIXED_LENGTH) -> EncodedHybrid:
    seq_bytes = seq.encode("ascii", "ignore")
    arr4 = _ASCII_MAP_4[np.frombuffer(seq_bytes, dtype=np.uint8)]
    base = np.where(arr4 == 4, 4, arr4.astype(np.int64))  # 0..4 already
    seq_len = min(int(base.size), fixed_length)
    base = base[:seq_len].astype(np.int64)
    seq_arr4 = arr4[:seq_len].astype(np.uint8)
    start = max(0, (fixed_length - seq_len) // 2)
    end = start + seq_len
    canvas = np.full(fixed_length, 4, dtype=np.int64)
    canvas[start:end] = base
    return EncodedHybrid(
        base_idx=canvas, seq_arr4=seq_arr4,
        start=start, end=end, header=header, label_seq_len=seq_len,
    )


# ============================================================================
# Tensor builders
# ============================================================================


def to_onehot_mask(base_indices: np.ndarray, starts: Sequence[int], ends: Sequence[int],
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.from_numpy(base_indices).to(device)
    B, L = base.shape
    X = torch.zeros((B, 5, L), dtype=torch.float32, device=device)
    X.scatter_(1, base.unsqueeze(1), 1.0)
    in_window = torch.zeros((B, L), dtype=torch.bool, device=device)
    for i, (s, e) in enumerate(zip(starts, ends)):
        in_window[i, s:e] = True
    mask = in_window & (base != 4)
    return X, mask


def build_chain_edge_index(node_offsets: list[int], node_counts: list[int]) -> torch.Tensor:
    """Build undirected chain + self-loops over a batched stack of per-graph nodes.

    node_offsets[i]: start index of graph i's nodes in the stacked node tensor.
    node_counts[i]: number of nodes for graph i.
    """
    src_list, dst_list = [], []
    for off, n in zip(node_offsets, node_counts):
        if n > 1:
            s = np.arange(off, off + n - 1, dtype=np.int64)
            d = np.arange(off + 1, off + n, dtype=np.int64)
            src_list.append(s); dst_list.append(d)
            src_list.append(d); dst_list.append(s)
        idx = np.arange(off, off + n, dtype=np.int64)
        src_list.append(idx); dst_list.append(idx)
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long)
    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    return torch.from_numpy(np.stack([src, dst], axis=0)).long()


def featurise_batch(seq_arr4_list: list[np.ndarray], featurizer: KmerFeaturizer,
                    device: torch.device):
    """Return (x_gnn, edge_index, batch_vec) for a batch of sequences."""
    feats = [featurizer.featurise_4(a) for a in seq_arr4_list]
    counts = [f.shape[0] for f in feats]
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1])).tolist()
    x_gnn = np.concatenate(feats, axis=0).astype(np.float32)
    x_gnn_t = torch.from_numpy(x_gnn).to(device)
    edge_index = build_chain_edge_index(offsets, counts).to(device)
    batch_vec = np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(counts)])
    batch_vec_t = torch.from_numpy(batch_vec).to(device)
    return x_gnn_t, edge_index, batch_vec_t


# ============================================================================
# Forward helpers (head selection)
# ============================================================================


HEAD_SUPERFAMILY = "sf"
HEAD_CLASS = "class"


def _head_logits(model, fused, head: str):
    if head == HEAD_SUPERFAMILY:
        return model.superfamily_head(fused)
    if head == HEAD_CLASS:
        return model.class_head(fused)
    raise ValueError(head)


def _forward_full(model: HybridV43, x_cnn, mask, x_gnn, edge_index, batch_vec):
    """Forward through the model, returning (class_logits, sf_logits)."""
    cnn_e = model.cnn_tower(x_cnn, mask)
    gnn_e = model.gnn_tower(x_gnn, edge_index, batch_vec)
    fused, _ = model.fusion(cnn_e, gnn_e)
    return model.class_head(fused), model.superfamily_head(fused)


@torch.no_grad()
def predict_logits(model, encs: Sequence[EncodedHybrid], featurizer: KmerFeaturizer,
                   device, batch_size: int = 16, head: str = HEAD_SUPERFAMILY):
    out = []
    for i in range(0, len(encs), batch_size):
        chunk = encs[i : i + batch_size]
        base = np.stack([e.base_idx for e in chunk], axis=0)
        starts = [e.start for e in chunk]
        ends = [e.end for e in chunk]
        X, mask = to_onehot_mask(base, starts, ends, device)
        x_gnn, edge_index, batch_vec = featurise_batch(
            [e.seq_arr4 for e in chunk], featurizer, device,
        )
        cl, sf = _forward_full(model, X, mask, x_gnn, edge_index, batch_vec)
        out.append((sf if head == HEAD_SUPERFAMILY else cl).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


# ============================================================================
# Saliency / IG (CNN-input gradients only)
# ============================================================================


def compute_saliency(model, enc: EncodedHybrid, target_class: int, featurizer: KmerFeaturizer,
                     device, head: str = HEAD_SUPERFAMILY) -> np.ndarray:
    """Vanilla input-gradient saliency on the CNN one-hot input.

    Note: gradient is taken w.r.t. the CNN one-hot, NOT the GNN k-mer features.
    This is the standard convention for sequence saliency (we attribute to
    nucleotide positions, not to k-mer-window meta-features) and matches the
    thesis figure.
    """
    base = enc.base_idx[None, :]
    X, mask = to_onehot_mask(base, [enc.start], [enc.end], device)
    X = X.detach().clone().requires_grad_(True)
    x_gnn, edge_index, batch_vec = featurise_batch([enc.seq_arr4], featurizer, device)
    cl, sf = _forward_full(model, X, mask, x_gnn, edge_index, batch_vec)
    logits = sf if head == HEAD_SUPERFAMILY else cl
    score = logits[0, target_class]
    grad = torch.autograd.grad(score, X)[0]
    return (grad * X).sum(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)


def compute_integrated_gradients(model, enc: EncodedHybrid, target_class: int,
                                 featurizer: KmerFeaturizer, device, steps: int = 16,
                                 head: str = HEAD_SUPERFAMILY) -> np.ndarray:
    """Integrated gradients on the CNN input with an N-baseline.

    The GNN branch is held fixed at the *original* k-mer features (we are
    attributing to per-nucleotide positions, not to graph features); this
    matches the standard sequence-attribution convention.
    """
    base = enc.base_idx[None, :]
    X1, mask = to_onehot_mask(base, [enc.start], [enc.end], device)
    base0 = np.full_like(enc.base_idx, 4)[None, :]
    X0, _ = to_onehot_mask(base0, [enc.start], [enc.end], device)
    delta = X1 - X0
    x_gnn, edge_index, batch_vec = featurise_batch([enc.seq_arr4], featurizer, device)
    accum = torch.zeros_like(X1)
    for k in range(1, steps + 1):
        alpha = k / steps
        X_k = (X0 + alpha * delta).detach().requires_grad_(True)
        cl, sf = _forward_full(model, X_k, mask, x_gnn, edge_index, batch_vec)
        logits = sf if head == HEAD_SUPERFAMILY else cl
        score = logits[0, target_class]
        grad = torch.autograd.grad(score, X_k)[0]
        accum = accum + grad
    avg_grad = accum / steps
    return (avg_grad * delta).sum(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)


# ============================================================================
# Perturbations (also update seq_arr4 -> re-featurise GNN)
# ============================================================================


def _apply_perturbation(enc: EncodedHybrid, start: int, end: int, mode: str,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (new_canvas_base_idx, new_seq_arr4)."""
    new_base = enc.base_idx.copy()
    new_seq4 = enc.seq_arr4.copy()
    # Convert canvas-coord (start, end) -> seq-coord (rs, re)
    rs = max(0, start - enc.start)
    re = min(enc.length, end - enc.start)

    if mode == "N":
        new_base[start:end] = 4
        new_seq4[rs:re] = 4
        return new_base, new_seq4
    if mode == "shuffle":
        # Permute ACGT positions; leave N positions alone.
        win = new_seq4[rs:re].copy()
        acgt = np.where(win != 4)[0]
        if acgt.size > 1:
            perm = rng.permutation(acgt.size)
            win[acgt] = win[acgt[perm]]
        new_seq4[rs:re] = win
        # mirror into canvas (canvas index 0..3 = base, 4 = N)
        new_base[start:end] = win.astype(np.int64)
        return new_base, new_seq4
    if mode == "reverse":
        rc_lookup_4 = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
        rc = rc_lookup_4[new_seq4[rs:re][::-1]]
        new_seq4[rs:re] = rc
        new_base[start:end] = rc.astype(np.int64)
        return new_base, new_seq4
    raise ValueError(mode)


def occlusion_profile(model, enc: EncodedHybrid, target_classes: Sequence[int],
                      featurizer: KmerFeaturizer, device,
                      window: int = 100, stride: int = 50, mode: str = "N",
                      batch_size: int = 16, rng_seed: int = 0,
                      head: str = HEAD_SUPERFAMILY) -> dict:
    rng = np.random.default_rng(rng_seed)
    s_lo, s_hi = enc.start, enc.end
    starts = np.arange(s_lo, max(s_lo + 1, s_hi - window + 1), stride, dtype=np.int64)
    ends = np.minimum(starts + window, s_hi)
    centers = (starts + ends) // 2
    P = len(starts)
    T = len(target_classes)

    target_t = torch.as_tensor(list(target_classes), dtype=torch.long, device=device)

    # Baseline
    base_X, base_mask = to_onehot_mask(enc.base_idx[None, :], [enc.start], [enc.end], device)
    x_gnn, edge_index, batch_vec = featurise_batch([enc.seq_arr4], featurizer, device)
    with torch.no_grad():
        cl, sf = _forward_full(model, base_X, base_mask, x_gnn, edge_index, batch_vec)
    base_logits = (sf if head == HEAD_SUPERFAMILY else cl)[0, target_t].detach().cpu().numpy()

    drops = np.zeros((T, P), dtype=np.float32)
    for i in range(0, P, batch_size):
        bs = starts[i : i + batch_size]
        be = ends[i : i + batch_size]
        bsize = len(bs)
        bases = np.empty((bsize, FIXED_LENGTH), dtype=np.int64)
        seq4s: list[np.ndarray] = []
        for j, (a, b) in enumerate(zip(bs, be)):
            nb, ns = _apply_perturbation(enc, int(a), int(b), mode, rng)
            bases[j] = nb
            seq4s.append(ns)
        Xb, maskb = to_onehot_mask(bases, [enc.start] * bsize, [enc.end] * bsize, device)
        x_gnn_b, ei_b, bv_b = featurise_batch(seq4s, featurizer, device)
        with torch.no_grad():
            cl_b, sf_b = _forward_full(model, Xb, maskb, x_gnn_b, ei_b, bv_b)
        sel = (sf_b if head == HEAD_SUPERFAMILY else cl_b)[:, target_t].detach().cpu().numpy()
        drops[:, i : i + bsize] = (base_logits[:, None] - sel.T)

    return {"centers": centers, "drops": drops, "baseline_logits": base_logits,
            "starts": starts, "ends": ends, "window": window, "stride": stride, "mode": mode}


def keep_only_window_profile(model, enc: EncodedHybrid, target_classes: Sequence[int],
                             featurizer: KmerFeaturizer, device,
                             window: int = 300, stride: int = 150,
                             batch_size: int = 16,
                             head: str = HEAD_SUPERFAMILY) -> dict:
    s_lo, s_hi = enc.start, enc.end
    starts = np.arange(s_lo, max(s_lo + 1, s_hi - window + 1), stride, dtype=np.int64)
    ends = np.minimum(starts + window, s_hi)
    centers = (starts + ends) // 2
    P = len(starts)
    T = len(target_classes)
    target_t = torch.as_tensor(list(target_classes), dtype=torch.long, device=device)
    survived = np.zeros((T, P), dtype=np.float32)

    for i in range(0, P, batch_size):
        bs = starts[i : i + batch_size]
        be = ends[i : i + batch_size]
        bsize = len(bs)
        bases = np.full((bsize, FIXED_LENGTH), 4, dtype=np.int64)
        seq4s: list[np.ndarray] = []
        for j, (a, b) in enumerate(zip(bs, be)):
            bases[j, a:b] = enc.base_idx[a:b]
            new_seq4 = np.full_like(enc.seq_arr4, 4)
            rs = a - enc.start
            re = b - enc.start
            new_seq4[rs:re] = enc.seq_arr4[rs:re]
            seq4s.append(new_seq4)
        Xb, maskb = to_onehot_mask(bases, [s_lo] * bsize, [s_hi] * bsize, device)
        x_gnn_b, ei_b, bv_b = featurise_batch(seq4s, featurizer, device)
        with torch.no_grad():
            cl_b, sf_b = _forward_full(model, Xb, maskb, x_gnn_b, ei_b, bv_b)
        survived[:, i : i + bsize] = (sf_b if head == HEAD_SUPERFAMILY else cl_b)[:, target_t].detach().cpu().numpy().T

    return {"centers": centers, "survived": survived, "starts": starts, "ends": ends}


def deletion_curve(model, enc: EncodedHybrid, saliency: np.ndarray, target_class: int,
                   featurizer: KmerFeaturizer, device, n_steps: int = 20,
                   rng_seed: int = 0, head: str = HEAD_SUPERFAMILY) -> dict:
    """Petsiuk-style deletion: progressively N-mask the top-k%-saliency positions
    and the k%-random positions; report true-class logit at each k."""
    rng = np.random.default_rng(rng_seed)
    s, e = enc.start, enc.end
    L = e - s
    region_sal = saliency[s:e]
    order_sal = np.argsort(-region_sal)
    perm_rand = rng.permutation(L)

    fractions = np.linspace(0.0, 1.0, n_steps + 1)

    base_X, base_mask = to_onehot_mask(enc.base_idx[None, :], [s], [e], device)
    x_gnn, edge_index, batch_vec = featurise_batch([enc.seq_arr4], featurizer, device)
    with torch.no_grad():
        cl, sf = _forward_full(model, base_X, base_mask, x_gnn, edge_index, batch_vec)
    base_logit = float((sf if head == HEAD_SUPERFAMILY else cl)[0, target_class].item())

    sal_logits = np.zeros(n_steps + 1, dtype=np.float32)
    rnd_logits = np.zeros(n_steps + 1, dtype=np.float32)
    sal_logits[0] = base_logit
    rnd_logits[0] = base_logit

    n_to_remove = (fractions[1:] * L).astype(np.int64)
    bases_sal: list[np.ndarray] = []
    seq4_sal: list[np.ndarray] = []
    bases_rnd: list[np.ndarray] = []
    seq4_rnd: list[np.ndarray] = []
    for k in n_to_remove:
        b_sal = enc.base_idx.copy(); b_sal[s + order_sal[:k]] = 4
        s_sal = enc.seq_arr4.copy(); s_sal[order_sal[:k]] = 4
        bases_sal.append(b_sal); seq4_sal.append(s_sal)
        b_rnd = enc.base_idx.copy(); b_rnd[s + perm_rand[:k]] = 4
        s_rnd = enc.seq_arr4.copy(); s_rnd[perm_rand[:k]] = 4
        bases_rnd.append(b_rnd); seq4_rnd.append(s_rnd)

    all_bases = np.stack(bases_sal + bases_rnd, axis=0)
    all_seq4 = seq4_sal + seq4_rnd
    n_each = n_steps
    Xb, maskb = to_onehot_mask(all_bases, [s] * (2 * n_each), [e] * (2 * n_each), device)
    x_gnn_b, ei_b, bv_b = featurise_batch(all_seq4, featurizer, device)
    with torch.no_grad():
        cl_b, sf_b = _forward_full(model, Xb, maskb, x_gnn_b, ei_b, bv_b)
    vals = (sf_b if head == HEAD_SUPERFAMILY else cl_b)[:, target_class].detach().cpu().numpy()
    sal_logits[1:] = vals[:n_each]
    rnd_logits[1:] = vals[n_each:]

    auc_sal = float(np.trapz(sal_logits, fractions))
    auc_rnd = float(np.trapz(rnd_logits, fractions))
    return {"fractions": fractions, "saliency_curve": sal_logits, "random_curve": rnd_logits,
            "auc_saliency": auc_sal, "auc_random": auc_rnd, "auc_gap": auc_rnd - auc_sal}


def saliency_occlusion_correlation(saliency: np.ndarray, occlusion: dict,
                                   target_class_index: int = 0) -> float:
    starts = occlusion["starts"]; ends = occlusion["ends"]
    drops = occlusion["drops"][target_class_index]
    win_sal = np.array([saliency[s:e].mean() for s, e in zip(starts, ends)], dtype=np.float64)
    if np.std(win_sal) == 0 or np.std(drops) == 0:
        return float("nan")
    r1 = _rankdata(win_sal); r2 = _rankdata(drops)
    return float(np.corrcoef(r1, r2)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


# ============================================================================
# I/O (delegates to the shared simple parsers)
# ============================================================================


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


def load_multiclass_labels(label_path):
    d: dict[str, str] = {}
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            d[parts[0].lstrip(">")] = parts[1]
    return d


def load_tir_labels(label_path):
    d: dict[str, int] = {}
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            v = parts[1].strip().upper()
            if v in ("TRUE", "T", "1", "YES"):
                d[parts[0].lstrip(">")] = 1
            elif v in ("FALSE", "F", "0", "NO"):
                d[parts[0].lstrip(">")] = 0
            else:
                try:
                    d[parts[0].lstrip(">")] = int(float(v))
                except ValueError:
                    continue
    return d
