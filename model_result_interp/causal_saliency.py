"""
Causal saliency / occlusion library for the v3-style ImprovedRCCNN model.

Purpose
-------
Address the supervisor critique that gradient-based saliency is correlational
by providing intervention-based attribution: occlusion in three perturbation
modes (N-mask, shuffle, reverse-complement) and metrics that quantify how
well a gradient saliency profile predicts the actual logit drop produced by
those interventions.

The module is self-contained: it redefines the ImprovedRCCNN architecture
(matching `model_result_interp.ipynb`) so it can load the existing checkpoint
without notebook coupling.

Conventions
-----------
- Sequence encoding: 5-channel one-hot, channels = (A, C, G, T, N).
- Mask: shape (B, L) bool, True at positions that hold real ACGT bases.
- Canvas: FIXED_LENGTH = 25565, sequence center-placed, padded with N (idx 4)
  outside the [start, end) window. Positions outside the window have mask=False.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FIXED_LENGTH = 25565

# Byte -> base index lookup. N and any non-ACGT byte map to 4.
ENCODE = np.full(256, 4, dtype=np.int64)
for ch, idx in zip(b"ACGTNacgtn", [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]):
    ENCODE[ch] = idx

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
# Model architecture (mirrors model_result_interp.ipynb)
# ============================================================================


class ConvBlock(nn.Module):
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


class ImprovedRCCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        width=128,
        motif_kernels=(7, 15, 21),
        context_kernel=9,
        context_dilations=(1, 2, 4, 8),
        dropout=0.15,
        rc_mode="late",
        aux_weight=0.1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.rc_mode = rc_mode
        self.aux_weight = aux_weight

        if rc_mode == "early":
            self.motif_convs = nn.ModuleList(
                [RCFirstConv1d(width, kernel_size=k, dropout=dropout) for k in motif_kernels]
            )
        else:
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
            [
                ConvBlock(width, width, kernel_size=context_kernel, dilation=d, dropout=dropout)
                for d in context_dilations
            ]
        )
        self.pool = MaskedMaxPool1d(kernel_size=2, stride=2)

        self.class_head = nn.Sequential(
            nn.Linear(width, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes),
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(width, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
            nn.Sigmoid(),
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
        class_logits = self.class_head(pooled)
        boundary_pred = self.boundary_head(pooled)
        return class_logits, boundary_pred


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    tag_to_id = ckpt["tag_to_id"]
    arch = ckpt["arch"]
    model = ImprovedRCCNN(
        num_classes=len(class_names),
        width=arch.get("width", 128),
        motif_kernels=arch.get("motif_kernels", (7, 15, 21)),
        context_kernel=arch.get("context_kernel", 9),
        context_dilations=arch.get("context_dilations", (1, 2, 4, 8)),
        rc_mode=arch.get("rc_mode", "late"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, tag_to_id


# ============================================================================
# Encoding
# ============================================================================


@dataclass
class EncodedSeq:
    """A single sequence laid out on the FIXED_LENGTH canvas.

    base_idx: int64[L] indices in 0..4 (4 = N or padding).
    start, end: bounds of the real (non-padded) portion in canvas coords.
    """

    base_idx: np.ndarray  # shape (FIXED_LENGTH,)
    start: int
    end: int
    header: str

    @property
    def length(self) -> int:
        return self.end - self.start


def encode_sequence(seq: str, header: str = "", fixed_length: int = FIXED_LENGTH) -> EncodedSeq:
    seq_bytes = seq.encode("ascii", "ignore")
    seq_idx = ENCODE[np.frombuffer(seq_bytes, dtype=np.uint8)]
    seq_len = min(len(seq_idx), fixed_length)
    seq_idx = seq_idx[:seq_len]
    start = max(0, (fixed_length - seq_len) // 2)
    end = start + seq_len
    canvas = np.full(fixed_length, 4, dtype=np.int64)  # default N
    canvas[start:end] = seq_idx
    return EncodedSeq(base_idx=canvas, start=start, end=end, header=header)


def to_onehot_mask(
    base_indices: np.ndarray, starts: Sequence[int], ends: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-sequence base index arrays into one-hot tensor + mask on device.

    base_indices: int64 array of shape (B, L). Values 0..4 (4 = N).
    starts, ends: per-sequence canvas bounds (used to set the mask).
    Returns: X (B, 5, L) float, mask (B, L) bool. mask = position is in [start,end) AND base != N.
    """
    base = torch.from_numpy(base_indices).to(device, non_blocking=True)  # (B, L)
    B, L = base.shape
    X = torch.zeros((B, 5, L), dtype=torch.float32, device=device)
    X.scatter_(1, base.unsqueeze(1), 1.0)
    in_window = torch.zeros((B, L), dtype=torch.bool, device=device)
    for i, (s, e) in enumerate(zip(starts, ends)):
        in_window[i, s:e] = True
    mask = in_window & (base != 4)
    return X, mask


# ============================================================================
# Forward helpers
# ============================================================================


@torch.no_grad()
def predict_logits(
    model: ImprovedRCCNN, encs: Sequence[EncodedSeq], device: torch.device, batch_size: int = 32
) -> np.ndarray:
    """Return (N, num_classes) logit array."""
    out: list[np.ndarray] = []
    for start in range(0, len(encs), batch_size):
        chunk = encs[start : start + batch_size]
        base = np.stack([e.base_idx for e in chunk], axis=0)
        starts = [e.start for e in chunk]
        ends = [e.end for e in chunk]
        X, mask = to_onehot_mask(base, starts, ends, device)
        logits, _ = model(X, mask)
        out.append(logits.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


# ============================================================================
# Gradient saliency
# ============================================================================


def compute_saliency(
    model: ImprovedRCCNN,
    enc: EncodedSeq,
    target_class: int,
    device: torch.device,
    method: str = "input_grad",
) -> np.ndarray:
    """Per-position scalar attribution over the canvas.

    `input_grad`: dot of input gradient with one-hot input, summed over channels.
                  This is the natural per-position projection of vanilla saliency
                  for one-hot inputs (equivalent to the gradient at the active
                  channel). Returned as float32, length = FIXED_LENGTH.
    """
    if method != "input_grad":
        raise ValueError(f"Unknown saliency method: {method}")
    base = enc.base_idx[None, :]
    X, mask = to_onehot_mask(base, [enc.start], [enc.end], device)
    X = X.detach().clone().requires_grad_(True)
    logits, _ = model(X, mask)
    score = logits[0, target_class]
    grad = torch.autograd.grad(score, X, retain_graph=False, create_graph=False)[0]
    sal = (grad * X).sum(dim=1).squeeze(0).detach().cpu().numpy()  # (L,)
    return sal.astype(np.float32)


def compute_integrated_gradients(
    model: ImprovedRCCNN,
    enc: EncodedSeq,
    target_class: int,
    device: torch.device,
    steps: int = 32,
    baseline: str = "Nmask",
) -> np.ndarray:
    """Integrated gradients with an N-baseline (uniform N over the same window).

    Returned shape = FIXED_LENGTH (per-position attribution after summing over
    channels). Note: still gradient-based; reported only as a comparison to
    vanilla saliency, not as a causal signal.
    """
    base = enc.base_idx[None, :]
    X1, mask = to_onehot_mask(base, [enc.start], [enc.end], device)
    if baseline == "Nmask":
        base0 = np.full_like(enc.base_idx, 4)[None, :]
        X0, _ = to_onehot_mask(base0, [enc.start], [enc.end], device)
    elif baseline == "zero":
        X0 = torch.zeros_like(X1)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    delta = X1 - X0  # (1, 5, L)
    accum = torch.zeros_like(X1)
    for k in range(1, steps + 1):
        alpha = k / steps
        X_k = (X0 + alpha * delta).detach().requires_grad_(True)
        logits, _ = model(X_k, mask)
        score = logits[0, target_class]
        grad = torch.autograd.grad(score, X_k)[0]
        accum = accum + grad
    avg_grad = accum / steps
    ig = (avg_grad * delta).sum(dim=1).squeeze(0).detach().cpu().numpy()
    return ig.astype(np.float32)


# ============================================================================
# Occlusion
# ============================================================================


def _apply_perturbation(
    base_idx: np.ndarray, start: int, end: int, mode: str, rng: np.random.Generator
) -> tuple[np.ndarray, bool]:
    """Return a perturbed copy of base_idx[start:end] (in canvas coords).

    Returns (new_base_idx, mask_disabled_in_window) where mask_disabled_in_window
    indicates whether the mask should be cleared in [start, end) (only for mode='N',
    since N's are also excluded by the `idx != 4` rule -- so we just rely on that).
    """
    out = base_idx.copy()
    if mode == "N":
        out[start:end] = 4
        return out, True
    if mode == "shuffle":
        # Permute only ACGT positions inside the window; N positions are left as N.
        window = out[start:end]
        acgt_pos = np.where(window != 4)[0]
        if acgt_pos.size > 1:
            perm = rng.permutation(acgt_pos.size)
            window[acgt_pos] = window[acgt_pos[perm]]
        out[start:end] = window
        return out, False
    if mode == "reverse":
        # Reverse-complement ACGT inside the window. N stays N.
        rc_lookup = np.array([3, 2, 1, 0, 4], dtype=np.int64)
        out[start:end] = rc_lookup[out[start:end][::-1]]
        return out, False
    raise ValueError(f"Unknown perturbation mode: {mode}")


def occlusion_profile(
    model: ImprovedRCCNN,
    enc: EncodedSeq,
    target_classes: Sequence[int],
    device: torch.device,
    window: int = 100,
    stride: int = 50,
    mode: str = "N",
    batch_size: int = 64,
    rng_seed: int = 0,
    region: tuple[int, int] | None = None,
) -> dict:
    """Slide a `window`-bp occlusion across the real-sequence portion.

    region: (start, end) in canvas coords; default = the sequence's [start, end).
    Returns dict with:
      centers: (P,) int  -- canvas position of each window center
      drops:   (T, P) float -- baseline_logit - perturbed_logit per target class.
                              Positive = the perturbation HURT that class.
      baseline_logits: (T,) float -- unperturbed logits for the requested classes.
    """
    rng = np.random.default_rng(rng_seed)
    s_lo, s_hi = region if region is not None else (enc.start, enc.end)
    starts = np.arange(s_lo, max(s_lo + 1, s_hi - window + 1), stride, dtype=np.int64)
    ends = np.minimum(starts + window, s_hi)
    centers = (starts + ends) // 2
    P = len(starts)
    T = len(target_classes)
    target_classes_t = torch.as_tensor(list(target_classes), dtype=torch.long, device=device)

    # Baseline forward (unperturbed)
    base_X, base_mask = to_onehot_mask(
        enc.base_idx[None, :], [enc.start], [enc.end], device
    )
    with torch.no_grad():
        base_logits, _ = model(base_X, base_mask)
    base_logits_sel = base_logits[0, target_classes_t].detach().cpu().numpy()  # (T,)

    drops = np.zeros((T, P), dtype=np.float32)
    for i in range(0, P, batch_size):
        bs = starts[i : i + batch_size]
        be = ends[i : i + batch_size]
        bsize = len(bs)
        batch_base = np.empty((bsize, FIXED_LENGTH), dtype=np.int64)
        s_list = []
        e_list = []
        for j, (a, b) in enumerate(zip(bs, be)):
            new_base, _ = _apply_perturbation(enc.base_idx, int(a), int(b), mode, rng)
            batch_base[j] = new_base
            s_list.append(enc.start)
            e_list.append(enc.end)
        Xb, maskb = to_onehot_mask(batch_base, s_list, e_list, device)
        with torch.no_grad():
            logits_b, _ = model(Xb, maskb)
        sel = logits_b[:, target_classes_t].detach().cpu().numpy()  # (bsize, T)
        drops[:, i : i + bsize] = (base_logits_sel[:, None] - sel.T)

    return {
        "centers": centers,
        "drops": drops,
        "baseline_logits": base_logits_sel,
        "starts": starts,
        "ends": ends,
        "window": window,
        "stride": stride,
        "mode": mode,
    }


def occlude_region(
    model: ImprovedRCCNN,
    enc: EncodedSeq,
    target_classes: Sequence[int],
    device: torch.device,
    start: int,
    end: int,
    mode: str = "N",
    rng_seed: int = 0,
) -> dict:
    """Single-region occlusion. Returns per-class logit drop."""
    rng = np.random.default_rng(rng_seed)
    target_classes_t = torch.as_tensor(list(target_classes), dtype=torch.long, device=device)
    base_X, base_mask = to_onehot_mask(
        enc.base_idx[None, :], [enc.start], [enc.end], device
    )
    with torch.no_grad():
        base_logits, _ = model(base_X, base_mask)
    base_sel = base_logits[0, target_classes_t].detach().cpu().numpy()
    new_base, _ = _apply_perturbation(enc.base_idx, int(start), int(end), mode, rng)
    Xb, maskb = to_onehot_mask(new_base[None, :], [enc.start], [enc.end], device)
    with torch.no_grad():
        logits_b, _ = model(Xb, maskb)
    pert_sel = logits_b[0, target_classes_t].detach().cpu().numpy()
    return {
        "drops": base_sel - pert_sel,  # (T,)
        "baseline_logits": base_sel,
        "perturbed_logits": pert_sel,
    }


def keep_only_window_profile(
    model: ImprovedRCCNN,
    enc: EncodedSeq,
    target_classes: Sequence[int],
    device: torch.device,
    window: int = 200,
    stride: int = 100,
    batch_size: int = 64,
) -> dict:
    """Mask EVERYTHING except a sliding `window` (sufficiency / B3).

    Returns surviving logits at each window position (not drops).
    """
    target_classes_t = torch.as_tensor(list(target_classes), dtype=torch.long, device=device)
    s_lo, s_hi = enc.start, enc.end
    starts = np.arange(s_lo, max(s_lo + 1, s_hi - window + 1), stride, dtype=np.int64)
    ends = np.minimum(starts + window, s_hi)
    centers = (starts + ends) // 2
    P = len(starts)
    T = len(target_classes)
    survived = np.zeros((T, P), dtype=np.float32)

    for i in range(0, P, batch_size):
        bs = starts[i : i + batch_size]
        be = ends[i : i + batch_size]
        bsize = len(bs)
        batch_base = np.full((bsize, FIXED_LENGTH), 4, dtype=np.int64)
        for j, (a, b) in enumerate(zip(bs, be)):
            batch_base[j, a:b] = enc.base_idx[a:b]
        Xb, maskb = to_onehot_mask(batch_base, [s_lo] * bsize, [s_hi] * bsize, device)
        with torch.no_grad():
            logits_b, _ = model(Xb, maskb)
        survived[:, i : i + bsize] = logits_b[:, target_classes_t].detach().cpu().numpy().T

    return {"centers": centers, "survived": survived, "starts": starts, "ends": ends}


# ============================================================================
# Aggregation / metrics
# ============================================================================


def deletion_curve(
    model: ImprovedRCCNN,
    enc: EncodedSeq,
    saliency: np.ndarray,
    target_class: int,
    device: torch.device,
    n_steps: int = 20,
    rng_seed: int = 0,
) -> dict:
    """Petsiuk-style deletion: progressively N-mask the top-k%-saliency positions
    and the k%-random positions; report true-class logit at each k."""
    rng = np.random.default_rng(rng_seed)
    s, e = enc.start, enc.end
    L = e - s
    region_sal = saliency[s:e]
    order_sal = np.argsort(-region_sal)  # most positive first
    perm_rand = rng.permutation(L)

    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    base_X, base_mask = to_onehot_mask(
        enc.base_idx[None, :], [s], [e], device
    )
    with torch.no_grad():
        base_logit = model(base_X, base_mask)[0][0, target_class].item()

    sal_logits = np.zeros(n_steps + 1, dtype=np.float32)
    rnd_logits = np.zeros(n_steps + 1, dtype=np.float32)
    sal_logits[0] = base_logit
    rnd_logits[0] = base_logit

    # Build all batched perturbed inputs (saliency + random), 2*n_steps total
    n_to_remove = (fractions[1:] * L).astype(np.int64)
    bases_sal = []
    bases_rnd = []
    for k in n_to_remove:
        b_sal = enc.base_idx.copy()
        b_sal[s + order_sal[:k]] = 4
        bases_sal.append(b_sal)
        b_rnd = enc.base_idx.copy()
        b_rnd[s + perm_rand[:k]] = 4
        bases_rnd.append(b_rnd)
    all_bases = np.stack(bases_sal + bases_rnd, axis=0)
    Xb, maskb = to_onehot_mask(
        all_bases,
        [s] * (2 * n_steps),
        [e] * (2 * n_steps),
        device,
    )
    with torch.no_grad():
        logits_b, _ = model(Xb, maskb)
    vals = logits_b[:, target_class].detach().cpu().numpy()
    sal_logits[1:] = vals[:n_steps]
    rnd_logits[1:] = vals[n_steps:]

    # Lower area under the saliency-deletion curve = saliency identifies
    # important positions better than random (a good causal-faithfulness signal).
    auc_sal = float(np.trapz(sal_logits, fractions))
    auc_rnd = float(np.trapz(rnd_logits, fractions))
    return {
        "fractions": fractions,
        "saliency_curve": sal_logits,
        "random_curve": rnd_logits,
        "auc_saliency": auc_sal,
        "auc_random": auc_rnd,
        "auc_gap": auc_rnd - auc_sal,  # >0 if saliency helps, <=0 if it doesn't
    }


def saliency_occlusion_correlation(
    saliency: np.ndarray,
    occlusion: dict,
    target_class_index: int = 0,
) -> float:
    """Spearman correlation between per-window mean saliency and per-window
    occlusion drop, restricted to the windows occlusion was evaluated on.

    Returns NaN if either side is constant (pathological short sequences).
    """
    starts = occlusion["starts"]
    ends = occlusion["ends"]
    drops = occlusion["drops"][target_class_index]  # (P,)
    win_sal = np.array(
        [saliency[s:e].mean() for s, e in zip(starts, ends)], dtype=np.float64
    )
    if np.std(win_sal) == 0 or np.std(drops) == 0:
        return float("nan")
    # Spearman = Pearson on ranks.
    r_sal = _rankdata(win_sal)
    r_drp = _rankdata(drops)
    return float(np.corrcoef(r_sal, r_drp)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank (handles ties)."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # Average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


# ============================================================================
# I/O
# ============================================================================


def read_fasta(path) -> tuple[list[str], list[str]]:
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


def load_multiclass_labels(label_path) -> dict[str, str]:
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


def load_tir_labels(label_path) -> dict[str, int]:
    """Parse the TIR presence/absence label file.

    Format observed in `vgp_feature_tir_v2.ipynb::load_tir_labels`: whitespace
    separated, first column the header name (optionally prefixed with '>'),
    second column the binary TIR flag.
    """
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


def parse_label_from_header(header: str) -> str | None:
    if "#" in header:
        return header.split("#")[-1]
    return None
