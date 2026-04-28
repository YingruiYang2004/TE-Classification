"""
Regenerate v3_confusion.png with corrected axis labels.
- "Transposase+" -> "DNA transposon"
- "None" -> "non-DNA"
Saves to thesis/figures/new_figures/v3_confusion.png
"""

import os, gc, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

# ── Constants ────────────────────────────────────────────────────────────────
FIXED_LENGTH = 40000
MIN_CLASS_COUNT = 100
LABEL_SMOOTHING = 0.1

FASTA_PATH = Path(__file__).parent.parent.parent / "data/vgp/all_vgp_tes.fa"
LABEL_PATH = Path(__file__).parent.parent.parent / "data/vgp/features-tpase"
CKPT_PATH  = Path(__file__).parent / "rc_cnn_hierarchical_v3.pt"
OUT_DIR    = Path(__file__).parent.parent.parent / "thesis/figures/new_figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Encoding ─────────────────────────────────────────────────────────────────
ENCODE = np.full(256, 4, dtype=np.int64)
for ch, idx in zip(b"ACGTNacgtn", [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]):
    ENCODE[ch] = idx
REV_COMP = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)

def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ── Data Loading ──────────────────────────────────────────────────────────────
def read_fasta(path):
    headers, sequences = [], []
    h, buf = None, []
    with open(path) as f:
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
    label_dict = {}
    binary_dict = {}
    with open(label_path) as f:
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

# ── Dataset ───────────────────────────────────────────────────────────────────
class SeqDatasetHierarchical(Dataset):
    def __init__(self, headers, sequences, binary_labels, class_labels, fixed_length=FIXED_LENGTH):
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
        seq_bytes = seq.encode("ascii", "ignore")
        seq_idx = ENCODE[np.frombuffer(seq_bytes, dtype=np.uint8)]
        max_start = max(0, self.fixed_length - seq_len)
        start_pos = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end_pos = start_pos + seq_len
        return (
            self.headers[idx], seq_idx,
            int(self.binary_labels[idx]), int(self.class_labels[idx]),
            start_pos, end_pos, seq_len
        )

def collate_hierarchical(batch, fixed_length=FIXED_LENGTH):
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
    Y_class  = torch.tensor(class_labels,  dtype=torch.long)
    starts_n = torch.tensor(starts, dtype=torch.float32) / fixed_length
    ends_n   = torch.tensor(ends,   dtype=torch.float32) / fixed_length
    lens_n   = torch.tensor(lengths,dtype=torch.float32) / fixed_length
    return list(headers), X, mask, Y_binary, Y_class, starts_n, ends_n, lens_n

# ── Model ─────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=9, dilation=1, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation)
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
        return self.dropout(F.gelu(self.batch_norm(y)))

class HierarchicalRCCNN(nn.Module):
    def __init__(self, num_superfamilies, width=128, motif_kernels=(7,15,21),
                 context_kernel=9, context_dilations=(1,2,4,8), dropout=0.15, rc_mode="late"):
        super().__init__()
        self.num_superfamilies = int(num_superfamilies)
        self.rc_mode = rc_mode

        if rc_mode == "early":
            self.motif_convs = nn.ModuleList([
                RCFirstConv1d(width, kernel_size=k, dropout=dropout) for k in motif_kernels
            ])
        else:
            self.motif_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(5, width, kernel_size=k, padding=k//2),
                    nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout)
                ) for k in motif_kernels
            ])

        in_ch = width * len(motif_kernels)
        self.mix = nn.Sequential(
            nn.Conv1d(in_ch, width, 1), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout)
        )
        self.context_blocks = nn.ModuleList([
            ConvBlock(width, width, kernel_size=context_kernel, dilation=d, dropout=dropout)
            for d in context_dilations
        ])
        self.pool = MaskedMaxPool1d(kernel_size=2, stride=2)

        self.binary_head = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 2)
        )
        self.superfamily_head = nn.Sequential(
            nn.Linear(width, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, self.num_superfamilies)
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(width, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 3), nn.Sigmoid()
        )

    @staticmethod
    def rc_transform(x, mask):
        x_rc   = x.index_select(1, REV_COMP.to(x.device)).flip(-1)
        mask_rc = None if mask is None else mask.flip(-1)
        return x_rc, mask_rc

    def encode(self, x, mask):
        feats = [conv(x) for conv in self.motif_convs]
        z = self.mix(torch.cat(feats, dim=1))
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
        return self.binary_head(pooled), self.superfamily_head(pooled), self.boundary_head(pooled)

# ── Rebuild Test Split (same params as original run) ─────────────────────────
def build_test_set():
    random_state  = 42
    subsample_none = 20000
    test_size      = 0.2

    print("Loading FASTA…")
    headers, sequences = read_fasta(FASTA_PATH)
    label_dict, binary_dict = load_hierarchical_labels(LABEL_PATH)

    all_h, all_s, all_tags, all_binary = [], [], [], []
    for h, s in zip(headers, sequences):
        if h not in label_dict:
            continue
        all_h.append(h); all_s.append(s)
        all_tags.append(label_dict[h])
        all_binary.append(binary_dict[h])
    del headers, sequences; gc.collect()
    print(f"Matched {len(all_h)} sequences")

    # Subsample None — same seed as training
    none_idx  = [i for i, b in enumerate(all_binary) if b == 0]
    tpase_idx = [i for i, b in enumerate(all_binary) if b == 1]
    if len(none_idx) > subsample_none:
        np.random.seed(random_state)
        sampled = np.random.choice(none_idx, subsample_none, replace=False)
        keep = sorted(list(tpase_idx) + list(sampled))
        all_h      = [all_h[i]      for i in keep]
        all_s      = [all_s[i]      for i in keep]
        all_tags   = [all_tags[i]   for i in keep]
        all_binary = [all_binary[i] for i in keep]

    # Build superfamily map
    tag_counts = Counter(t for t, b in zip(all_tags, all_binary) if b == 1)
    keep_sf    = {t for t, c in tag_counts.items() if c >= MIN_CLASS_COUNT}
    sf_names   = sorted(keep_sf)
    sf_to_id   = {t: i for i, t in enumerate(sf_names)}

    # Filter rare SF, build class ids
    fh, fs, ft, fb, fc = [], [], [], [], []
    for h, s, tag, binary in zip(all_h, all_s, all_tags, all_binary):
        if binary == 0:
            fh.append(h); fs.append(s); ft.append(tag); fb.append(0); fc.append(0)
        elif tag in sf_to_id:
            fh.append(h); fs.append(s); ft.append(tag); fb.append(1); fc.append(sf_to_id[tag])

    all_binary   = np.array(fb, dtype=np.int64)
    all_class_ids = np.array(fc, dtype=np.int64)

    _, idx_test = train_test_split(
        np.arange(len(fh)), test_size=test_size, stratify=all_binary, random_state=random_state
    )
    test_h   = [fh[i] for i in idx_test]
    test_s   = [fs[i] for i in idx_test]
    test_bin = all_binary[idx_test]
    test_cls = all_class_ids[idx_test]
    print(f"Test set: {len(test_h)} sequences  ({test_bin.sum()} DNA / {(test_bin==0).sum()} non-DNA)")
    return test_h, test_s, test_bin, test_cls, sf_names

# ── Run Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, test_h, test_s, test_bin, test_cls, device):
    ds     = SeqDatasetHierarchical(test_h, test_s, test_bin, test_cls)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=lambda b: collate_hierarchical(b, FIXED_LENGTH),
                        num_workers=0)
    all_bin_pred, all_bin_true = [], []
    all_sf_pred,  all_sf_true  = [], []

    model.eval()
    with torch.no_grad():
        for _, X, mask, Y_bin, Y_sf, *_ in loader:
            X    = X.to(device)
            mask = mask.to(device)
            bin_logits, sf_logits, _ = model(X, mask)
            bin_pred = bin_logits.argmax(1).cpu().numpy()
            sf_pred  = sf_logits.argmax(1).cpu().numpy()
            all_bin_pred.extend(bin_pred)
            all_bin_true.extend(Y_bin.numpy())
            tpase_mask = Y_bin == 1
            all_sf_pred.extend(sf_pred[tpase_mask.numpy()])
            all_sf_true.extend(Y_sf[tpase_mask].numpy())

    return (np.array(all_bin_pred), np.array(all_bin_true),
            np.array(all_sf_pred),  np.array(all_sf_true))

# ── Plot & Save ────────────────────────────────────────────────────────────────
def plot_and_save(bin_pred, bin_true, sf_pred, sf_true, sf_names, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Binary confusion matrix ---
    cm_bin = confusion_matrix(bin_true, bin_pred)
    # Normalise per row (true class)
    cm_bin_norm = cm_bin.astype(float) / cm_bin.sum(axis=1, keepdims=True)
    im1 = axes[0].imshow(cm_bin_norm, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["non-DNA", "DNA transposon"], fontsize=11)
    axes[0].set_yticklabels(["non-DNA", "DNA transposon"], fontsize=11)
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("True", fontsize=11)
    axes[0].set_title("Binary Confusion Matrix\n(DNA transposon vs non-DNA)", fontsize=12)
    for i in range(2):
        for j in range(2):
            pct  = cm_bin_norm[i, j]
            cnt  = cm_bin[i, j]
            col  = "white" if pct > 0.6 else "black"
            axes[0].text(j, i, f"{pct:.2f}\n({cnt})", ha="center", va="center",
                         fontsize=10, color=col)
    plt.colorbar(im1, ax=axes[0], label="Fraction")

    # --- Superfamily confusion matrix ---
    sf_short = [n.replace("DNA/", "") for n in sf_names]
    n_sf     = len(sf_short)
    cm_sf    = confusion_matrix(sf_true, sf_pred, labels=list(range(n_sf)))
    row_sums = cm_sf.sum(axis=1, keepdims=True)
    cm_sf_norm = np.where(row_sums > 0, cm_sf / row_sums, 0)
    im2 = axes[1].imshow(cm_sf_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(n_sf))
    axes[1].set_yticks(range(n_sf))
    axes[1].set_xticklabels(sf_short, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(sf_short, fontsize=8)
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("True", fontsize=11)
    axes[1].set_title("Superfamily Confusion Matrix\n(DNA transposons only)", fontsize=12)
    for i in range(n_sf):
        for j in range(n_sf):
            pct = cm_sf_norm[i, j]
            col = "white" if pct > 0.6 else "black"
            if cm_sf[i, j] > 0:
                axes[1].text(j, i, f"{pct:.2f}", ha="center", va="center", fontsize=7, color=col)
    plt.colorbar(im2, ax=axes[1], label="Fraction")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Print metrics
    bin_acc = accuracy_score(bin_true, bin_pred)
    bin_f1  = f1_score(bin_true, bin_pred, average="macro")
    sf_acc  = accuracy_score(sf_true, sf_pred)
    sf_f1   = f1_score(sf_true, sf_pred, average="macro")
    print(f"\nBinary:     acc={bin_acc:.4f}  macro-F1={bin_f1:.4f}")
    print(f"Superfamily: acc={sf_acc:.4f}  macro-F1={sf_f1:.4f}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = resolve_device()
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    sf_names_ckpt = ckpt["superfamily_names"]

    model = HierarchicalRCCNN(
        num_superfamilies = arch["num_superfamilies"],
        width             = arch["width"],
        motif_kernels     = arch["motif_kernels"],
        context_kernel    = arch["context_kernel"],
        context_dilations = arch["context_dilations"],
        dropout           = 0.15,
        rc_mode           = arch["rc_mode"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded v3 checkpoint (epoch {ckpt['best_epoch']}, rc_mode={arch['rc_mode']})")

    # Rebuild test split
    test_h, test_s, test_bin, test_cls, sf_names = build_test_set()

    # Verify SF names match checkpoint
    assert sf_names == sf_names_ckpt, f"SF mismatch:\n{sf_names}\nvs\n{sf_names_ckpt}"

    # Evaluate
    bin_pred, bin_true, sf_pred, sf_true = evaluate(model, test_h, test_s, test_bin, test_cls, device)

    # Plot and save
    out_path = OUT_DIR / "v3_confusion.png"
    plot_and_save(bin_pred, bin_true, sf_pred, sf_true, sf_names, out_path)

if __name__ == "__main__":
    main()
