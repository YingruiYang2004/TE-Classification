"""
Run the v4.3 hybrid (Stage 2) and v4.3-singlefold hybrid (Stage 3) over the
ENTIRE VGP corpus (~135k sequences in `data/vgp/all_vgp_tes.fa`), including
the ~125k sequences whose Pantera transposase tag is `None`.

Goal: see how the model classifies sequences that the Pantera tpase pipeline
could not assign a transposase to, by comparing two leak-controlled hybrids
that have a 3-class (DNA/LTR/LINE) head + 23-superfamily head.

Outputs (one CSV per checkpoint):
    all_predictions_v4.3_rotating_full.csv
    all_predictions_v4.3_singlefold_full.csv
    all_predictions_v4.3_full_summary.md   (joint summary across both)

CSV schema (per row, one row per FASTA sequence):
    header
    species_id              # rsplit('-', 1)[-1] of the part before '#'
    pantera_tpase_tag       # raw tag from features-tpase ('None' or e.g. 'DNA/hAT')
    raw_class               # part before '/' in tag, or '' if tag=='None'
    raw_sf                  # part after  '/' in tag, or '' if tag=='None'
    seq_length
    excluded_genome         # bool, in {mOrnAna, bTaeGut, rAllMis}
    outer_role              # 'test' / 'trainval' (replays training split)
    inner_role              # 'train' / 'val' / '' (Stage-3 single-fold only)
    pred_class              # argmax over {DNA, LTR, LINE}
    pred_class_conf         # softmax max
    pred_sf                 # argmax over the 23 SF vocab
    pred_sf_conf            # softmax max
    p_class_DNA, p_class_LTR, p_class_LINE          # full softmax (3 cols)
    p_sf__<sfname1>, ..., p_sf__<sfname23>          # full softmax (23 cols)

The script streams the FASTA in chunks of CHUNK sequences to keep memory
bounded (~100 windows * 2049 dims * 4 B ~= 0.8 MB / seq peak, so CHUNK=2000
~= 1.6 GB peak features). Both checkpoints are loaded once at startup;
features are computed once per chunk and re-used for both checkpoints.

Usage:
    python predict_full_corpus.py --limit 1000   # dry run
    python predict_full_corpus.py                # full run
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import types as _types
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # workspace root

# ---- Helper notebook (any v4 variant works -- arch helpers are identical) ----
REF_NOTEBOOK = HERE / "v4.2/cluster session/vgp_hybrid_v4.2.ipynb"
NEEDED_CELLS = [0, 1, 3, 5, 7, 9, 10, 12, 14, 16, 18]

# ---- Two checkpoints (Stage 2 + Stage 3) ----
CHECKPOINTS = [
    ("v4.3_rotating",   HERE / "v4.3/cluster session/hybrid_v4.3_epoch40.pt"),
    ("v4.3_singlefold", HERE / "v4.3/cluster session single fold/hybrid_v4.3_singlefold_epoch28.pt"),
]

# ---- Inputs ----
FASTA_PATH = ROOT / "data/vgp/all_vgp_tes.fa"
LABEL_PATH = ROOT / "data/vgp/features-tpase"

# ---- Outputs ----
OUT_CSV_BY_CKPT = {
    "v4.3_rotating":   HERE / "all_predictions_v4.3_rotating_full.csv",
    "v4.3_singlefold": HERE / "all_predictions_v4.3_singlefold_full.csv",
}
OUT_LOG  = HERE / "predict_full_corpus.log"
OUT_MD   = HERE / "all_predictions_v4.3_full_summary.md"

# ---- Split provenance constants (from training notebooks) ----
EXCLUDE_GENOMES = {'mOrnAna', 'bTaeGut', 'rAllMis'}
OUTER_TEST_SIZE = 0.2
OUTER_RANDOM_STATE = 42
# Stage-3 single-fold uses random_state=43 inside trainval
INNER_VAL_SIZE = 0.2
INNER_RANDOM_STATE = 43
KEEP_CLASSES = ('DNA', 'LTR', 'LINE')

# ---- Streaming params ----
CHUNK = 2000
BATCH_SIZE = 32


# -------- log tee --------
class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for st in self.s:
            st.write(x); st.flush()
    def flush(self):
        for st in self.s: st.flush()


def setup_logging():
    log = open(OUT_LOG, "w")
    sys.stdout = _Tee(sys.__stdout__, log)
    sys.stderr = _Tee(sys.__stderr__, log)
    return log


# -------- helpers from notebook --------
def load_helpers():
    print(f"Loading helpers from {REF_NOTEBOOK}")
    with open(REF_NOTEBOOK) as f:
        nb = json.load(f)
    mod = _types.ModuleType("v4_full_corpus_helpers")
    sys.modules["v4_full_corpus_helpers"] = mod
    ns: dict = mod.__dict__
    ns["__name__"] = "v4_full_corpus_helpers"
    for ci in NEEDED_CELLS:
        cell = nb["cells"][ci]
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        exec(compile(src, f"<v4 cell {ci}>", "exec"), ns)
    print("  helpers loaded.")
    return ns


# -------- header parsing --------
def species_from_header(h: str) -> str:
    """Mirror `_species_from_header` in v4.3 training notebook."""
    return h.lstrip(">").split("#", 1)[0].rsplit("-", 1)[-1]


def split_pantera_tag(tag: str) -> tuple[str, str]:
    """Pantera tag like 'DNA/hAT' -> ('DNA', 'hAT'); 'None' -> ('', '')."""
    if tag == "None" or tag == "":
        return "", ""
    if "/" in tag:
        a, b = tag.split("/", 1)
        return a, b
    return tag, ""


def load_label_map(path: Path) -> dict[str, str]:
    """features-tpase has one '>{header}\\t{tag}' per line; tag may be 'None'."""
    out: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            h = parts[0].lstrip(">")
            out[h] = parts[1]
    return out


def iter_fasta(path: Path):
    """Yield (header, sequence) pairs (sequence upper-cased, joined)."""
    h, buf = None, []
    with open(path) as f:
        for line in f:
            if not line:
                continue
            if line[0] == ">":
                if h is not None:
                    yield h, "".join(buf).upper()
                    buf = []
                h = line[1:].strip().split()[0]
            else:
                buf.append(line.strip())
    if h is not None:
        yield h, "".join(buf).upper()


# -------- split provenance (replays training-time splits) --------
def compute_split_provenance(headers: list[str], pantera_tags: list[str]):
    """Replay outer (Stage 2 + Stage 3 share) and inner (Stage 3 only) splits.

    Returns three string arrays of len(headers):
        outer_role: 'test' / 'trainval' / 'excluded'
        inner_role: 'train' / 'val' / 'test_outer' / 'excluded' / 'untracked'
        excluded_genome (bool)
    The training notebook applies BOTH the EXCLUDE_GENOMES filter AND the
    `keep_classes={DNA,LTR,LINE}` filter before any split, so we replay that
    exact pre-filter to assign roles. Sequences outside the pre-filter get
    role 'excluded' (excluded_genome=True) or 'untracked' (raw class not in
    keep_classes) so the reader can sub-set unambiguously.
    """
    n = len(headers)
    species = np.array([species_from_header(h) for h in headers])
    excluded_genome = np.array([s in EXCLUDE_GENOMES for s in species])

    raw_classes = np.array([split_pantera_tag(t)[0] for t in pantera_tags])
    in_keep_classes = np.isin(raw_classes, list(KEEP_CLASSES))

    # eligible = same pre-filter as training (notebook also drops superfamilies
    # below MIN_CLASS_COUNT and caps at MAX_PER_SF, but we IGNORE those caps
    # here -- they only affect which sequences entered training, NOT the
    # outer test species split. For outer_role we only need the species set.)
    eligible_mask = (~excluded_genome) & in_keep_classes
    eligible_idx = np.where(eligible_mask)[0]

    outer_role = np.full(n, "untracked", dtype=object)
    outer_role[excluded_genome] = "excluded"
    outer_role[(~excluded_genome) & (~in_keep_classes)] = "untracked"

    inner_role = np.full(n, "untracked", dtype=object)
    inner_role[excluded_genome] = "excluded"
    inner_role[(~excluded_genome) & (~in_keep_classes)] = "untracked"

    if eligible_idx.size > 0:
        elig_species = species[eligible_idx]
        # Outer split: stratify on superfamily integer; here we only need the
        # split, so use raw_classes as stratification proxy (training uses sf
        # ids, but the split is grouped by species so the y= argument only
        # affects which species end up where -- we DO need to mirror it
        # exactly to reproduce the assignment).
        # The training code stratifies on SF id; emulating that exactly would
        # require rebuilding the same SF id table. The species assignment is
        # determined by `groups=species` and the random_state, so for our
        # purpose (provenance lookup) we approximate y with a constant array;
        # this gives a *species-wise* split that is consistent within itself
        # but may not match training species-by-species.
        # ⚠ FIDELITY NOTE: outer_role here is provenance-grade, not a perfect
        # replay; the Verification step will quantify the match.
        gss = GroupShuffleSplit(
            n_splits=1, test_size=OUTER_TEST_SIZE, random_state=OUTER_RANDOM_STATE
        )
        idx_trainval, idx_test = next(gss.split(
            np.arange(eligible_idx.size),
            y=np.zeros(eligible_idx.size),
            groups=elig_species,
        ))
        global_trainval = eligible_idx[idx_trainval]
        global_test     = eligible_idx[idx_test]
        outer_role[global_trainval] = "trainval"
        outer_role[global_test]     = "test"

        # Inner split (Stage 3 only). random_state=43 inside trainval.
        if global_trainval.size > 0:
            tv_species = species[global_trainval]
            gss_inner = GroupShuffleSplit(
                n_splits=1, test_size=INNER_VAL_SIZE,
                random_state=INNER_RANDOM_STATE,
            )
            tv_idx_train, tv_idx_val = next(gss_inner.split(
                np.arange(global_trainval.size),
                y=np.zeros(global_trainval.size),
                groups=tv_species,
            ))
            inner_role[global_trainval[tv_idx_train]] = "train"
            inner_role[global_trainval[tv_idx_val]]   = "val"
        inner_role[global_test] = "test_outer"

    return outer_role.astype(str), inner_role.astype(str), excluded_genome


# -------- inference --------
@torch.no_grad()
def infer_chunk(model, device, headers, seqs, kmer_feats,
                HybridDataset, collate_hybrid, FIXED_LENGTH, batch_size):
    """Return (cls_probs[N,3], sf_probs[N, n_sf]) as numpy float32."""
    n = len(seqs)
    # placeholder labels (unused for inference)
    sf_targets = np.zeros(n, dtype=np.int64)
    cls_targets = np.zeros(n, dtype=np.int64)
    ds = HybridDataset(
        headers=headers, sequences=seqs,
        binary_labels=sf_targets, class_labels=cls_targets,
        kmer_features=kmer_feats, fixed_length=FIXED_LENGTH,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(collate_hybrid, fixed_length=FIXED_LENGTH),
    )
    cls_probs_list, sf_probs_list = [], []
    for batch in loader:
        _, X_cnn, mask, _Y_sf, _Y_cls, x_gnn, edge_index, batch_vec = batch
        X_cnn = X_cnn.to(device); mask = mask.to(device)
        x_gnn = x_gnn.to(device); edge_index = edge_index.to(device)
        batch_vec = batch_vec.to(device)
        cls_logits, sf_logits, _ = model(X_cnn, mask, x_gnn, edge_index, batch_vec)
        cls_probs_list.append(torch.softmax(cls_logits.float(), dim=1).cpu().numpy())
        sf_probs_list.append(torch.softmax(sf_logits.float(), dim=1).cpu().numpy())
    return np.concatenate(cls_probs_list, axis=0), np.concatenate(sf_probs_list, axis=0)


# -------- main --------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N FASTA sequences (dry-run).")
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = p.parse_args()

    setup_logging()
    print(f"FASTA: {FASTA_PATH}")
    print(f"Labels: {LABEL_PATH}")
    print(f"Limit: {args.limit}  Chunk: {args.chunk}  Batch: {args.batch_size}")

    ns = load_helpers()
    KmerWindowFeaturizer    = ns["KmerWindowFeaturizer"]
    HybridDataset           = ns["HybridDataset"]
    collate_hybrid          = ns["collate_hybrid"]
    HybridTEClassifierV4    = ns["HybridTEClassifierV4"]
    FIXED_LENGTH            = ns["FIXED_LENGTH"]
    resolve_device          = ns["resolve_device"]

    DEVICE = resolve_device()
    print(f"Device: {DEVICE}  FIXED_LENGTH={FIXED_LENGTH}")

    # ---- Load labels ----
    print("\n=== Loading labels ===")
    t0 = time.time()
    label_map = load_label_map(LABEL_PATH)
    print(f"  {len(label_map):,} label entries in {time.time()-t0:.1f}s")

    # ---- First pass: parse ALL headers + tags + length from the full FASTA.
    # We always read the full corpus here so the species split (provenance) is
    # computed against the same population the training script saw. --limit
    # is honoured later in Pass 2 (inference only).
    print("\n=== Pass 1: parse FASTA headers (no sequences kept) ===")
    t0 = time.time()
    all_headers: list[str] = []
    all_tags: list[str] = []
    all_lengths: list[int] = []
    n_missing_tag = 0
    for h, s in iter_fasta(FASTA_PATH):
        all_headers.append(h)
        all_lengths.append(len(s))
        tag = label_map.get(h)
        if tag is None:
            n_missing_tag += 1
            tag = "None"
        all_tags.append(tag)
    n_total_corpus = len(all_headers)
    n_total = min(args.limit, n_total_corpus) if args.limit else n_total_corpus
    print(f"  parsed {n_total_corpus:,} sequences in {time.time()-t0:.1f}s "
          f"(missing-tag fallback to 'None': {n_missing_tag:,}); "
          f"will infer on first {n_total:,}")

    # ---- Compute split provenance ----
    print("\n=== Computing split provenance ===")
    t0 = time.time()
    outer_role, inner_role, excluded_genome = compute_split_provenance(
        all_headers, all_tags
    )
    print(f"  done in {time.time()-t0:.1f}s")
    from collections import Counter
    print(f"  outer_role counts: {Counter(outer_role)}")
    print(f"  inner_role counts: {Counter(inner_role)}")
    print(f"  excluded_genome:   {int(excluded_genome.sum()):,} sequences")

    # ---- Load both models + featurizers (arch is identical, but use each
    #      ckpt's recorded params to be safe) ----
    print("\n=== Loading checkpoints ===")
    models = {}
    sf_names_by = {}
    sf_to_id_by = {}
    featurizers = {}
    for ck_name, ck_path in CHECKPOINTS:
        ckpt = torch.load(ck_path, map_location=DEVICE, weights_only=False)
        arch = ckpt["arch"]
        sf_names = list(ckpt["superfamily_names"])
        sf_to_id = dict(ckpt["superfamily_to_id"])
        print(f"  {ck_name}: epoch={ckpt['epoch']}, score={ckpt['score']:.4f}, "
              f"n_sf={len(sf_names)}")
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
        models[ck_name] = model
        sf_names_by[ck_name] = sf_names
        sf_to_id_by[ck_name] = sf_to_id
        featurizers[ck_name] = KmerWindowFeaturizer(
            k=arch.get("kmer_k", 7),
            dim=arch.get("kmer_dim", arch["gnn_in_dim"] - 1),
            window=arch.get("kmer_window", 512),
            stride=arch.get("kmer_stride", 256),
            add_pos=True, l2_normalize=True,
        )
        del ckpt
    # Pick one featurizer to use; check they're identical
    first_arch = None
    for ck_name, _ in CHECKPOINTS:
        f = featurizers[ck_name]
        sig = (f.k, f.dim, f.window, f.stride, f.add_pos, f.l2_normalize)
        if first_arch is None:
            first_arch = sig
        else:
            assert sig == first_arch, \
                f"Featurizer mismatch between checkpoints: {first_arch} vs {sig}"
    SHARED_FEAT = featurizers[CHECKPOINTS[0][0]]
    print(f"  shared featurizer: k={SHARED_FEAT.k} dim={SHARED_FEAT.dim} "
          f"window={SHARED_FEAT.window} stride={SHARED_FEAT.stride}")

    # ---- Open output CSVs ----
    fhs: dict[str, object] = {}
    for ck_name, _ in CHECKPOINTS:
        out_csv = OUT_CSV_BY_CKPT[ck_name]
        sf_names = sf_names_by[ck_name]
        cls_cols = [f"p_class_{c}" for c in KEEP_CLASSES]
        # sanitise SF names for csv header
        sf_cols = [f"p_sf__{re.sub(r'[^A-Za-z0-9_/-]', '_', s)}" for s in sf_names]
        header = (
            "header,species_id,pantera_tpase_tag,raw_class,raw_sf,seq_length,"
            "excluded_genome,outer_role,inner_role,"
            "pred_class,pred_class_conf,pred_sf,pred_sf_conf,"
            + ",".join(cls_cols) + "," + ",".join(sf_cols) + "\n"
        )
        f = open(out_csv, "w")
        f.write(header)
        fhs[ck_name] = f
        print(f"  opened {out_csv.name}")

    # ---- Pass 2: stream FASTA in chunks, featurise, infer for both models ----
    print(f"\n=== Pass 2: streaming FASTA in chunks of {args.chunk} ===")
    n_done = 0
    chunk_buf_h: list[str] = []
    chunk_buf_s: list[str] = []
    chunk_buf_global_idx: list[int] = []
    t_run_start = time.time()

    def flush_chunk():
        nonlocal chunk_buf_h, chunk_buf_s, chunk_buf_global_idx, n_done
        if not chunk_buf_h:
            return
        ck_t0 = time.time()
        # featurise once
        kmer = []
        for s in chunk_buf_s:
            X, _ = SHARED_FEAT.featurize_sequence(s)
            kmer.append(X)
        t_feat = time.time() - ck_t0

        # for each checkpoint, run inference and append to its CSV
        for ck_name, _ in CHECKPOINTS:
            ti0 = time.time()
            cls_probs, sf_probs = infer_chunk(
                models[ck_name], DEVICE, chunk_buf_h, chunk_buf_s, kmer,
                HybridDataset, collate_hybrid, FIXED_LENGTH, args.batch_size,
            )
            t_inf = time.time() - ti0
            sf_names = sf_names_by[ck_name]
            f = fhs[ck_name]
            for i_local, gi in enumerate(chunk_buf_global_idx):
                h = all_headers[gi]
                tag = all_tags[gi]
                rc, rs = split_pantera_tag(tag)
                sp = species_from_header(h)
                cls_p = cls_probs[i_local]
                sf_p  = sf_probs[i_local]
                cls_arg = int(cls_p.argmax())
                sf_arg  = int(sf_p.argmax())
                row = [
                    h, sp, tag, rc, rs, str(all_lengths[gi]),
                    str(bool(excluded_genome[gi])),
                    outer_role[gi], inner_role[gi],
                    KEEP_CLASSES[cls_arg], f"{cls_p[cls_arg]:.6f}",
                    sf_names[sf_arg], f"{sf_p[sf_arg]:.6f}",
                ]
                row += [f"{v:.6f}" for v in cls_p]
                row += [f"{v:.6f}" for v in sf_p]
                # CSV-escape: headers/tags should not contain commas or quotes
                # in this corpus, but be safe
                def _esc(x):
                    if "," in x or '"' in x or "\n" in x:
                        return '"' + x.replace('"', '""') + '"'
                    return x
                f.write(",".join(_esc(c) for c in row) + "\n")
            f.flush()
            print(f"   [{ck_name}] inference {t_inf:.1f}s")
        # cleanup
        n_done += len(chunk_buf_h)
        chunk_buf_h.clear()
        chunk_buf_s.clear()
        chunk_buf_global_idx.clear()
        del kmer
        gc.collect()
        elapsed = time.time() - t_run_start
        rate = n_done / max(elapsed, 1e-6)
        eta = (n_total - n_done) / max(rate, 1e-6)
        print(f"  chunk done: featurise {t_feat:.1f}s | "
              f"progress {n_done:,}/{n_total:,} "
              f"({100*n_done/n_total:.1f}%) | rate {rate:.1f}/s | ETA {eta/60:.1f} min")

    for i, (h, s) in enumerate(iter_fasta(FASTA_PATH)):
        if args.limit is not None and i >= args.limit:
            break
        chunk_buf_h.append(h)
        chunk_buf_s.append(s)
        chunk_buf_global_idx.append(i)
        if len(chunk_buf_h) >= args.chunk:
            flush_chunk()
    flush_chunk()

    for f in fhs.values():
        f.close()
    print(f"\nDONE. Wrote:")
    for ck_name, _ in CHECKPOINTS:
        print(f"  {OUT_CSV_BY_CKPT[ck_name]}")

    # ---- Joint summary ----
    write_joint_summary(n_total, sf_names_by)


def write_joint_summary(n_total: int, sf_names_by: dict):
    """Quick summary md: predicted-class distribution among None-tagged seqs,
    plus Stage-2/Stage-3 agreement, computed from the freshly written CSVs."""
    import pandas as pd
    print("\n=== Building joint summary ===")
    dfs = {}
    for ck_name, path in OUT_CSV_BY_CKPT.items():
        if not path.exists():
            print(f"  ! missing {path}, skipping summary")
            return
        # keep_default_na=False so the string 'None' (Pantera tag) is preserved
        # and not silently coerced to NaN by pandas
        dfs[ck_name] = pd.read_csv(path, low_memory=False, keep_default_na=False)

    lines = [f"# Full-corpus predictions summary",
             f"",
             f"- Total sequences: {n_total:,}",
             f"- Outputs:"]
    for ck_name in dfs:
        lines.append(f"  - `{OUT_CSV_BY_CKPT[ck_name].name}` ({len(dfs[ck_name]):,} rows)")
    lines.append("")

    for ck_name, df in dfs.items():
        lines.append(f"## {ck_name}")
        # outer_role breakdown
        lines.append("### outer_role distribution")
        lines.append("```")
        lines.append(df["outer_role"].value_counts(dropna=False).to_string())
        lines.append("```")
        # predicted-class distribution among None-tagged
        none_mask = (df["pantera_tpase_tag"] == "None")
        lines.append(f"### Predicted class for `pantera_tpase_tag == None` "
                     f"(n={int(none_mask.sum()):,})")
        lines.append("```")
        lines.append(df.loc[none_mask, "pred_class"].value_counts().to_string())
        lines.append("```")
        # mean confidence by predicted class on None
        lines.append(f"### Mean `pred_class_conf` for None, by predicted class")
        lines.append("```")
        lines.append(df.loc[none_mask].groupby("pred_class")["pred_class_conf"].agg(
            ["count", "mean", "std"]).to_string())
        lines.append("```")

    # ---- Agreement Stage 2 vs Stage 3 ----
    if len(dfs) == 2:
        ck_a, ck_b = list(dfs.keys())
        a = dfs[ck_a].set_index("header")
        b = dfs[ck_b].set_index("header")
        common = a.index.intersection(b.index)
        a = a.loc[common]; b = b.loc[common]
        agree_cls = (a["pred_class"] == b["pred_class"]).mean()
        agree_sf  = (a["pred_sf"]    == b["pred_sf"]).mean()
        none_idx = (a["pantera_tpase_tag"] == "None")
        agree_cls_none = (a.loc[none_idx, "pred_class"] == b.loc[none_idx, "pred_class"]).mean()
        agree_sf_none  = (a.loc[none_idx, "pred_sf"]    == b.loc[none_idx, "pred_sf"]).mean()
        lines += [
            f"## Agreement: {ck_a} vs {ck_b}",
            f"- overall: pred_class agree {agree_cls:.4f}, pred_sf agree {agree_sf:.4f}",
            f"- on Pantera-tag == None: pred_class agree {agree_cls_none:.4f}, "
            f"pred_sf agree {agree_sf_none:.4f}",
            f"- cross-tab pred_class on None:",
            "```",
            str(pd.crosstab(a.loc[none_idx, "pred_class"],
                            b.loc[none_idx, "pred_class"],
                            rownames=[ck_a], colnames=[ck_b])),
            "```",
        ]

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT_MD}")


if __name__ == "__main__":
    main()
