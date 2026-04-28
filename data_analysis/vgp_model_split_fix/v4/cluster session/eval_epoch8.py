"""Re-evaluate the epoch-8 checkpoint of the CSD3 v4 run on the held-out test set.

Strategy: exec the GPU notebook's definition cells (0..23) so we get all
classes/functions, then do the data prep + test-loader build inline (mirroring
cell 21), then load `hybrid_v4_epoch8.pt`, then call the natural-prevalence
eval (cell 39) directly.
"""
import json, os, sys, gc, math, time
from pathlib import Path

THIS = Path(__file__).resolve().parent
os.chdir(THIS)
sys.path.insert(0, str(THIS))

NB = THIS / "vgp_hybrid_v4_gpu.ipynb"
CKPT = THIS / "hybrid_v4_epoch8.pt"
print(f"NB:   {NB}")
print(f"CKPT: {CKPT}")

# ---- Exec definition cells (skip training-driver cell 25, plotting, etc.)
nb = json.load(open(NB))
DEFINITION_CELLS = [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 23, 39]
ns = {"__name__": "__nbexec__", "__file__": str(NB)}
for i in DEFINITION_CELLS:
    src = "".join(nb["cells"][i]["source"])
    print(f"  exec cell {i}  ({len(src)} chars)")
    exec(compile(src, f"<cell {i}>", "exec"), ns)

# ---- Now mirror cell 21 to build the test loader ------------------------
import numpy as np, torch
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader

# Pull constants and fns from notebook namespace
# Override paths: notebook uses CSD3-relative paths; locally point to data/vgp/
WORKSPACE = THIS.parents[3]   # .../TE Classification
FASTA_PATH = str(WORKSPACE / "data" / "vgp" / "all_vgp_tes.fa")
LABEL_PATH = str(WORKSPACE / "data" / "vgp" / "20260120_features_sf")
assert Path(FASTA_PATH).is_file(), FASTA_PATH
assert Path(LABEL_PATH).is_file(), LABEL_PATH
print(f"FASTA: {FASTA_PATH}")
print(f"LABEL: {LABEL_PATH}")
FIXED_LENGTH      = ns["FIXED_LENGTH"]
MIN_CLASS_COUNT   = ns["MIN_CLASS_COUNT"]
MAX_PER_SF        = ns["MAX_PER_SF"]
KMER_K, KMER_DIM, KMER_WINDOW, KMER_STRIDE = ns["KMER_K"], ns["KMER_DIM"], ns["KMER_WINDOW"], ns["KMER_STRIDE"]
GNN_HIDDEN, GNN_LAYERS = ns["GNN_HIDDEN"], ns["GNN_LAYERS"]
CNN_WIDTH         = ns["CNN_WIDTH"]
MOTIF_KERNELS     = ns["MOTIF_KERNELS"]
CONTEXT_DILATIONS = ns["CONTEXT_DILATIONS"]
RC_FUSION_MODE    = ns["RC_FUSION_MODE"]
FUSION_DIM        = ns["FUSION_DIM"]
NUM_HEADS         = ns["NUM_HEADS"]
DROPOUT           = ns["DROPOUT"]

read_fasta              = ns["read_fasta"]
load_binary_dna_labels  = ns["load_binary_dna_labels"]
KmerWindowFeaturizer    = ns["KmerWindowFeaturizer"]
KmerWindowFeaturizerGPU = ns["KmerWindowFeaturizerGPU"]
HybridDataset           = ns["HybridDataset"]
collate_hybrid          = ns["collate_hybrid"]
HybridTEClassifierV4    = ns["HybridTEClassifierV4"]
eval_natural_prevalence = ns["eval_natural_prevalence"]

random_state = 42
test_size = 0.2

# Device: use MPS if available locally
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ---- Load + binarise -------------------------------------------------------
t0 = time.time()
print("\n=== Loading data ===")
headers, sequences = read_fasta(FASTA_PATH)
label_dict, class_dict = load_binary_dna_labels(LABEL_PATH)

all_h, all_s, all_tags, all_toplevel = [], [], [], []
for h, s in zip(headers, sequences):
    if h not in label_dict:
        continue
    all_h.append(h); all_s.append(s)
    all_tags.append(label_dict[h]); all_toplevel.append(class_dict[h])
del headers, sequences; gc.collect()
print(f"Matched {len(all_h)} sequences  [{time.time()-t0:.1f}s]")

# SF mapping from DNA only
dna_tags = [t for t, top in zip(all_tags, all_toplevel) if top == 1]
tag_counts = Counter(dna_tags)
keep_sf = {t for t, c in tag_counts.items() if c >= MIN_CLASS_COUNT}
superfamily_names = sorted(keep_sf)
superfamily_to_id = {t: i for i, t in enumerate(superfamily_names)}
n_superfamilies = len(superfamily_names)
print(f"Kept SFs ({n_superfamilies}): {superfamily_names}")

SF_SENTINEL = 0
fh, fs, ft, ftop, fsf = [], [], [], [], []
for h, s, tag, top in zip(all_h, all_s, all_tags, all_toplevel):
    if top == 0:
        fh.append(h); fs.append(s); ft.append(tag); ftop.append(0); fsf.append(SF_SENTINEL)
    elif tag in superfamily_to_id:
        fh.append(h); fs.append(s); ft.append(tag); ftop.append(1); fsf.append(superfamily_to_id[tag])
all_h, all_s, all_tags = fh, fs, ft
all_toplevel = np.array(ftop, dtype=np.int64)
all_sf = np.array(fsf, dtype=np.int64)
del fh, fs, ft, ftop, fsf; gc.collect()
print(f"Filtered: {len(all_h)} (DNA={int((all_toplevel==1).sum())}, non-DNA={int((all_toplevel==0).sum())})")

# Global non-DNA cap
if MAX_PER_SF is not None:
    rng_cap = np.random.RandomState(random_state)
    nd_idx = np.where(all_toplevel == 0)[0]
    d_idx = np.where(all_toplevel == 1)[0]
    by_tag = {}
    for i in nd_idx:
        by_tag.setdefault(all_tags[i], []).append(int(i))
    keep_nd = []
    for tag, idxs in by_tag.items():
        if len(idxs) > MAX_PER_SF:
            idxs = rng_cap.choice(idxs, MAX_PER_SF, replace=False).tolist()
        keep_nd.extend(idxs)
    keep = sorted(d_idx.tolist() + keep_nd)
    all_h = [all_h[i] for i in keep]
    all_s = [all_s[i] for i in keep]
    all_tags = [all_tags[i] for i in keep]
    all_toplevel = all_toplevel[keep]
    all_sf = all_sf[keep]
    print(f"After non-DNA cap: {len(all_h)} (DNA={int((all_toplevel==1).sum())}, non-DNA={int((all_toplevel==0).sum())})")

# K-mer featurization (THE EXPENSIVE STEP)
print("\n=== K-mer featurization ===")
if device.type in ("cuda", "mps"):
    feat = KmerWindowFeaturizerGPU(k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW,
                                   stride=KMER_STRIDE, add_pos=True, l2_normalize=True,
                                   device=device)
    print(f"  using GPU featurizer on {device}")
else:
    feat = KmerWindowFeaturizer(k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW,
                                stride=KMER_STRIDE, add_pos=True, l2_normalize=True)
    print("  using CPU featurizer")
all_kmer = []
n_total = len(all_s)
print_every = max(2000, n_total // 20)
t1 = time.time()
for i, seq in enumerate(all_s, 1):
    X, _ = feat.featurize_sequence(seq)
    all_kmer.append(X.astype(np.float16, copy=False))
    if i % print_every == 0 or i == n_total:
        elapsed = time.time() - t1
        rate = i / max(elapsed, 1e-6)
        eta = (n_total - i) / max(rate, 1e-6)
        print(f"  [{i}/{n_total}] {rate:.1f} seq/s  ETA {eta/60:.1f} min", flush=True)
print(f"Done featurizing: {time.time()-t1:.1f}s")

# Stratification + species grouping (must EXACTLY match cell 21)
def _species_from_header(h):
    return h.lstrip(">").split("#", 1)[0].rsplit("-", 1)[1]
all_species = np.array([_species_from_header(h) for h in all_h])
all_strat = all_toplevel * (n_superfamilies + 1) + all_sf

gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
idx_trainval, idx_test = next(gss.split(np.arange(len(all_h)), y=all_strat, groups=all_species))
print(f"\nSplit: trainval={len(idx_trainval)}, test={len(idx_test)}")

test_h = [all_h[i] for i in idx_test]
test_s = [all_s[i] for i in idx_test]
test_top = all_toplevel[idx_test]
test_sf  = all_sf[idx_test]
test_kmer = [all_kmer[i] for i in idx_test]

ds_test = HybridDataset(test_h, test_s, test_top, test_sf, test_kmer)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False, collate_fn=collate_hybrid, num_workers=0)
print(f"Test loader: {len(ds_test)} samples")

# ---- Build model + load epoch 8 -------------------------------------------
class_names = ["non-DNA", "DNA"]
n_classes = 2

print("\n=== Loading epoch-8 checkpoint ===")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"  epoch       : {ckpt['epoch']}")
print(f"  score       : {ckpt['score']:.4f}")
print(f"  arch keys   : {list(ckpt['arch'].keys())}")
sf_names_ckpt = ckpt["superfamily_names"]
print(f"  SF names    : {sf_names_ckpt}")
assert sf_names_ckpt == superfamily_names, "SF mismatch!"

arch = ckpt["arch"]
model = HybridTEClassifierV4(
    num_classes=arch["num_classes"],
    num_superfamilies=arch["num_superfamilies"],
    cnn_width=arch["cnn_width"],
    motif_kernels=tuple(arch["motif_kernels"]),
    context_dilations=tuple(arch["context_dilations"]),
    rc_mode=arch["rc_mode"],
    gnn_in_dim=arch["gnn_in_dim"],
    gnn_hidden=arch["gnn_hidden"],
    gnn_layers=arch["gnn_layers"],
    fusion_dim=arch["fusion_dim"],
    num_heads=arch["num_heads"],
    dropout=DROPOUT,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model on {device}, params={sum(p.numel() for p in model.parameters()):,}")

# ---- Run eval --------------------------------------------------------------
print("\n=== Running natural-prevalence eval (epoch 8) ===")
eval_natural_prevalence(model, loader_test, device, class_names, superfamily_names)
print(f"\nTOTAL TIME: {(time.time()-t0)/60:.1f} min")
