"""
Evaluate v4.2 epoch28, v4.3 epoch40 (rotating-CV), and v4.3-singlefold epoch28
on the THREE held-out genomes in data/mini_benchmark/:
  - bTaeGut: taeniopygia_guttata_lib.fa  (zebra finch)
  - mOrnAna: platypus_curated_lib.fa     (platypus)
  - rAllMis: Aligator_mississippiensis_lib.fa  (alligator)

These genomes are EXCLUDED from the VGP corpus, so this is a true held-out
benchmark on out-of-distribution species/lineages.

Header format (mini_benchmark labelled libs):
    >{name}#{class}[/{superfamily}] @{taxon} [S:{...}]
e.g.  >Eulor1#DNA @Amniota [S:50]
      >Academ-1_AMi#DNA/Academ-1 @Crocodylia [S:40,50]

Outputs:
  - eval_mini_benchmark.log   (stdout tee)
  - eval_mini_benchmark.json  (full per-(model,genome) metrics)
  - eval_mini_benchmark.md    (compact comparison table)
"""

from __future__ import annotations
import json, sys, time, types as _types, re, gc
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # workspace root

# ---- Reference notebook for helper/model code (any v4 variant works -- arch identical) ----
REF_NOTEBOOK = HERE / "v4.2/cluster session/vgp_hybrid_v4.2.ipynb"
NEEDED_CELLS = [0, 1, 3, 5, 7, 9, 10, 12, 14, 16, 18]

# ---- Three checkpoints ----
CHECKPOINTS = [
    ("v4.2_epoch28",            HERE / "v4.2/cluster session/hybrid_v4.2_epoch28.pt"),
    ("v4.3_rotating_epoch40",   HERE / "v4.3/cluster session/hybrid_v4.3_epoch40.pt"),
    ("v4.3_singlefold_epoch28", HERE / "v4.3/cluster session single fold/hybrid_v4.3_singlefold_epoch28.pt"),
]

# ---- Three held-out genomes ----
GENOMES = [
    ("bTaeGut", ROOT / "data/mini_benchmark/bTaeGut/taeniopygia_guttata_lib.fa"),
    ("mOrnAna", ROOT / "data/mini_benchmark/mOrnAna/platypus_curated_lib.fa"),
    ("rAllMis", ROOT / "data/mini_benchmark/rAllMis/Aligator_mississippiensis_lib.fa"),
]

OUT_LOG  = HERE / "eval_mini_benchmark.log"
OUT_JSON = HERE / "eval_mini_benchmark.json"
OUT_MD   = HERE / "eval_mini_benchmark.md"

KEEP_CLASSES = ('DNA', 'LTR', 'LINE')
CLASS_TO_ID  = {c: i for i, c in enumerate(KEEP_CLASSES)}


# -- log tee --------------------------------------------------------------
class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for st in self.s: st.write(x); st.flush()
    def flush(self):
        for st in self.s: st.flush()
_log = open(OUT_LOG, "w")
sys.stdout = _Tee(sys.__stdout__, _log)
sys.stderr = _Tee(sys.__stderr__, _log)


# -- helper code -----------------------------------------------------------
print(f"Loading helpers from {REF_NOTEBOOK}")
with open(REF_NOTEBOOK) as f:
    nb = json.load(f)

_helper_mod = _types.ModuleType("v4_mini_eval_helpers")
sys.modules["v4_mini_eval_helpers"] = _helper_mod
ns: dict = _helper_mod.__dict__
ns["__name__"] = "v4_mini_eval_helpers"
for ci in NEEDED_CELLS:
    cell = nb["cells"][ci]
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    exec(compile(src, f"<v4 cell {ci}>", "exec"), ns)
print("  helpers loaded.")

KmerWindowFeaturizer    = ns["KmerWindowFeaturizer"]
HybridDataset           = ns["HybridDataset"]
collate_hybrid          = ns["collate_hybrid"]
HybridTEClassifierV4    = ns["HybridTEClassifierV4"]
FIXED_LENGTH            = ns["FIXED_LENGTH"]
resolve_device          = ns["resolve_device"]


# -- mini-benchmark FASTA + parser ----------------------------------------
HEADER_RX = re.compile(r"^>(?P<name>[^#\s]+)#(?P<cls>[^/\s]+)(?:/(?P<sf>[^\s]+))?")

def read_minibench_fasta(path: Path):
    """Return list of (header, seq, class_str, sf_str_or_None).

    Skips entries whose class is not in KEEP_CLASSES (DNA/LTR/LINE).
    """
    entries = []
    cur_header, cur_seq = None, []
    def flush():
        if cur_header is None: return
        m = HEADER_RX.match(cur_header)
        if not m: return
        cls = m.group("cls")
        sf  = m.group("sf")  # may be None
        if cls not in KEEP_CLASSES:
            return
        seq = "".join(cur_seq).upper()
        entries.append((cur_header[1:], seq, cls, sf))
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                cur_header = line.split()[0]  # drop @taxon [S:..]
                # Re-grab full header for parsing (already only first token, fine)
                cur_seq = []
            else:
                cur_seq.append(line)
        flush()
    return entries


def to_sf_tag(cls: str, sf: str | None) -> str:
    """Map (class, sf) to the model tag form used in superfamily_to_id."""
    if sf is None or sf == "":
        return cls           # e.g. 'DNA'
    return f"{cls}/{sf}"     # e.g. 'DNA/hAT'


# -- evaluation core ------------------------------------------------------
@torch.no_grad()
def run_inference(model, device, headers, seqs, kmer_feats, sf_targets, cls_targets):
    ds = HybridDataset(
        headers=headers, sequences=seqs,
        binary_labels=np.asarray(sf_targets, dtype=np.int64),
        class_labels=np.asarray(cls_targets, dtype=np.int64),
        kmer_features=kmer_feats, fixed_length=FIXED_LENGTH,
    )
    loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=0,
        collate_fn=partial(collate_hybrid, fixed_length=FIXED_LENGTH),
    )
    sf_p, cls_p = [], []
    for batch in loader:
        _, X_cnn, mask, _Y_sf, _Y_cls, x_gnn, edge_index, batch_vec = batch
        X_cnn = X_cnn.to(device); mask = mask.to(device)
        x_gnn = x_gnn.to(device); edge_index = edge_index.to(device)
        batch_vec = batch_vec.to(device)
        cls_logits, sf_logits, _ = model(X_cnn, mask, x_gnn, edge_index, batch_vec)
        sf_p.extend(sf_logits.argmax(dim=1).cpu().tolist())
        cls_p.extend(cls_logits.argmax(dim=1).cpu().tolist())
    return np.asarray(cls_p), np.asarray(sf_p)


def metrics_block(cls_t, cls_p, sf_t, sf_p, sf_mask, sf_names):
    """sf_mask: bool array selecting entries whose true SF is known to model."""
    out = {
        "n_total": int(len(cls_t)),
        "n_sf_evaluable": int(sf_mask.sum()),
    }
    # top-class metrics on ALL entries (class is always known)
    out["class_acc"]          = float(accuracy_score(cls_t, cls_p))
    out["class_balanced_acc"] = float(balanced_accuracy_score(cls_t, cls_p))
    out["class_macroF1"]      = float(f1_score(cls_t, cls_p, average="macro", zero_division=0))
    out["class_weightedF1"]   = float(f1_score(cls_t, cls_p, average="weighted", zero_division=0))
    # per-class
    f1s = f1_score(cls_t, cls_p, average=None, labels=list(range(len(KEEP_CLASSES))), zero_division=0)
    sup = np.bincount(cls_t, minlength=len(KEEP_CLASSES))
    from sklearn.metrics import precision_score, recall_score
    precs = precision_score(cls_t, cls_p, average=None, labels=list(range(len(KEEP_CLASSES))), zero_division=0)
    recs  = recall_score(cls_t, cls_p, average=None, labels=list(range(len(KEEP_CLASSES))), zero_division=0)
    out["class_per"] = {
        c: {"f1": float(f1s[i]), "prec": float(precs[i]), "rec": float(recs[i]), "support": int(sup[i])}
        for i, c in enumerate(KEEP_CLASSES)
    }
    # SF metrics on subset where label is in model's vocabulary
    if sf_mask.sum() > 0:
        sft = sf_t[sf_mask]; sfp = sf_p[sf_mask]
        out["sf_acc"]          = float(accuracy_score(sft, sfp))
        out["sf_balanced_acc"] = float(balanced_accuracy_score(sft, sfp))
        out["sf_macroF1"]      = float(f1_score(sft, sfp, average="macro", zero_division=0))
        out["sf_weightedF1"]   = float(f1_score(sft, sfp, average="weighted", zero_division=0))
        # per-SF (only those with support>0)
        n_sf = len(sf_names)
        sf_f1s = f1_score(sft, sfp, average=None, labels=list(range(n_sf)), zero_division=0)
        sf_sup = np.bincount(sft, minlength=n_sf)
        out["sf_per"] = {
            sf_names[i]: {"f1": float(sf_f1s[i]), "support": int(sf_sup[i])}
            for i in range(n_sf) if sf_sup[i] > 0
        }
    else:
        out["sf_acc"] = out["sf_balanced_acc"] = out["sf_macroF1"] = out["sf_weightedF1"] = None
        out["sf_per"] = {}
    return out


# -- pre-load all genomes (parsing only; featurization is per-model since
#    arch params identical we could share, but we re-featurize per-model
#    for safety: arch dict does carry kmer_k/dim/window/stride identical) --
DEVICE = resolve_device()
print(f"Device: {DEVICE}")
print(f"FIXED_LENGTH={FIXED_LENGTH}")

print("\n=== Parsing mini-benchmark FASTAs ===")
genome_data = {}  # gname -> (headers, seqs, cls_arr, sf_tag_list)
for gname, gpath in GENOMES:
    entries = read_minibench_fasta(gpath)
    headers = [e[0] for e in entries]
    seqs    = [e[1] for e in entries]
    classes = np.array([CLASS_TO_ID[e[2]] for e in entries], dtype=np.int64)
    sf_tags = [to_sf_tag(e[2], e[3]) for e in entries]
    genome_data[gname] = (headers, seqs, classes, sf_tags)
    cls_counts = {c: int((classes == i).sum()) for i, c in enumerate(KEEP_CLASSES)}
    print(f"  {gname:<8s} n={len(entries):4d}  by class: {cls_counts}")

# -- main loop -----------------------------------------------------------
results = {"checkpoints": {}, "_meta": {"device": str(DEVICE), "fixed_length": int(FIXED_LENGTH)}}

for ck_name, ck_path in CHECKPOINTS:
    print(f"\n{'='*70}\n=== Checkpoint: {ck_name}\n=== {ck_path}\n{'='*70}")
    ckpt = torch.load(ck_path, map_location=DEVICE, weights_only=False)
    arch       = ckpt["arch"]
    sf_names   = list(ckpt["superfamily_names"])
    sf_to_id   = dict(ckpt["superfamily_to_id"])
    print(f"  epoch={ckpt['epoch']} score={ckpt['score']:.4f} n_sf={len(sf_names)}")

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
        dropout=arch.get("dropout", 0.15),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    featurizer = KmerWindowFeaturizer(
        k=arch.get("kmer_k", 7),
        dim=arch.get("kmer_dim", arch["gnn_in_dim"] - 1),
        window=arch.get("kmer_window", 512),
        stride=arch.get("kmer_stride", 256),
        add_pos=True, l2_normalize=True,
    )

    per_genome = {}
    # also accumulate cross-genome arrays for overall summary
    all_cls_t, all_cls_p, all_sf_t, all_sf_p, all_sf_mask = [], [], [], [], []

    for gname, _ in GENOMES:
        headers, seqs, cls_t, sf_tags = genome_data[gname]
        # SF mask + integer targets (use 0 for unknown; masked out for SF metrics)
        sf_mask = np.array([t in sf_to_id for t in sf_tags], dtype=bool)
        sf_t = np.array([sf_to_id[t] if t in sf_to_id else 0 for t in sf_tags], dtype=np.int64)

        print(f"\n  -- {gname}: n={len(headers)}, sf-known={int(sf_mask.sum())} --")

        # featurize on the fly
        t0 = time.time()
        kmer = []
        for i, s in enumerate(seqs):
            X, _ = featurizer.featurize_sequence(s)
            kmer.append(X)
        print(f"     featurized in {time.time()-t0:.1f}s")

        t0 = time.time()
        cls_p, sf_p = run_inference(model, DEVICE, headers, seqs, kmer, sf_t, cls_t)
        print(f"     inferred  in {time.time()-t0:.1f}s")

        m = metrics_block(cls_t, cls_p, sf_t, sf_p, sf_mask, sf_names)
        per_genome[gname] = m
        print(f"     class macroF1={m['class_macroF1']:.4f}  "
              f"DNA F1/rec={m['class_per']['DNA']['f1']:.3f}/{m['class_per']['DNA']['rec']:.3f} "
              f"LTR F1={m['class_per']['LTR']['f1']:.3f}  "
              f"LINE F1={m['class_per']['LINE']['f1']:.3f}")
        if m["sf_macroF1"] is not None:
            print(f"     sf    macroF1={m['sf_macroF1']:.4f}  bal_acc={m['sf_balanced_acc']:.4f}  "
                  f"(n_sf_eval={m['n_sf_evaluable']})")

        all_cls_t.append(cls_t); all_cls_p.append(cls_p)
        all_sf_t.append(sf_t); all_sf_p.append(sf_p); all_sf_mask.append(sf_mask)

    # cross-genome overall
    cls_t_all = np.concatenate(all_cls_t); cls_p_all = np.concatenate(all_cls_p)
    sf_t_all  = np.concatenate(all_sf_t);  sf_p_all  = np.concatenate(all_sf_p)
    sf_m_all  = np.concatenate(all_sf_mask)
    overall = metrics_block(cls_t_all, cls_p_all, sf_t_all, sf_p_all, sf_m_all, sf_names)
    print(f"\n  ** OVERALL across 3 genomes: cls macroF1={overall['class_macroF1']:.4f}  "
          f"sf macroF1={overall['sf_macroF1']:.4f}  "
          f"DNA F1={overall['class_per']['DNA']['f1']:.3f}")

    results["checkpoints"][ck_name] = {
        "path": str(ck_path),
        "epoch": int(ckpt["epoch"]),
        "internal_score": float(ckpt["score"]),
        "per_genome": per_genome,
        "overall": overall,
    }

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote → {OUT_JSON}")


# ----- compact markdown table -----
def _fmt(x, p=3):
    return "—" if x is None else f"{x:.{p}f}"

lines = []
lines.append("# Mini-benchmark held-out genome evaluation\n")
lines.append("Three v4 variants evaluated on three genomes excluded from VGP training "
             "(`bTaeGut`, `mOrnAna`, `rAllMis`).\n")
lines.append("Top-level (3-class) metrics use ALL DNA/LTR/LINE entries. "
             "Superfamily metrics restrict to entries whose true SF tag is in the "
             "model's 23-SF vocabulary.\n\n")

for ck_name, info in results["checkpoints"].items():
    lines.append(f"## `{ck_name}`\n\n")
    lines.append("| genome | n | n_sf | cls_macroF1 | cls_bal_acc | DNA F1 | DNA rec | LTR F1 | LINE F1 | sf_macroF1 | sf_bal_acc |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for gname, _ in GENOMES:
        m = info["per_genome"][gname]
        lines.append(
            f"| {gname} | {m['n_total']} | {m['n_sf_evaluable']} | "
            f"{_fmt(m['class_macroF1'])} | {_fmt(m['class_balanced_acc'])} | "
            f"{_fmt(m['class_per']['DNA']['f1'])} | {_fmt(m['class_per']['DNA']['rec'])} | "
            f"{_fmt(m['class_per']['LTR']['f1'])} | {_fmt(m['class_per']['LINE']['f1'])} | "
            f"{_fmt(m['sf_macroF1'])} | {_fmt(m['sf_balanced_acc'])} |\n"
        )
    o = info["overall"]
    lines.append(
        f"| **overall** | **{o['n_total']}** | **{o['n_sf_evaluable']}** | "
        f"**{_fmt(o['class_macroF1'])}** | **{_fmt(o['class_balanced_acc'])}** | "
        f"**{_fmt(o['class_per']['DNA']['f1'])}** | **{_fmt(o['class_per']['DNA']['rec'])}** | "
        f"**{_fmt(o['class_per']['LTR']['f1'])}** | **{_fmt(o['class_per']['LINE']['f1'])}** | "
        f"**{_fmt(o['sf_macroF1'])}** | **{_fmt(o['sf_balanced_acc'])}** |\n\n"
    )

with open(OUT_MD, "w") as f:
    f.writelines(lines)
print(f"Wrote → {OUT_MD}")
print("DONE.")
