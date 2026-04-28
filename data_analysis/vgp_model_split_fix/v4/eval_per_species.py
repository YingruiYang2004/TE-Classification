"""Per-species diagnostic for the v4 hybrid checkpoints.

Extends ``cluster session/eval_epoch8.py`` to break down predictions by
species so we can answer the question that drives the cross-species
strategy ranking:

    Is hAT collapse concentrated on 1-2 outlier test species (sampling
    artefact, fix with rebalanced sampling) or uniform across all test
    species (architectural problem, fix with regulariser/adversary)?

What this script does
---------------------
1. Re-uses the notebook-exec pattern from ``eval_epoch8.py`` to import
   ``HybridTEClassifierV4`` and friends.
2. Builds the species-disjoint test loader exactly as in cell 21 of
   ``vgp_hybrid_v4_gpu.ipynb`` (seed 42, 80/20 group split).
3. Runs inference for one or more checkpoints, capturing for every
   sample: predicted class, predicted SF, gate weights ``(w_cnn, w_gnn)``.
4. Writes one CSV per checkpoint with columns:
       header, species, top_true, top_pred, sf_true, sf_pred,
       w_cnn, w_gnn
   to ``per_species_<ckpt_stem>.csv`` next to the script.
5. Prints to stdout:
   - DNA-only per-species precision/recall/F1 (focus: hAT, TcMar-Tc1)
   - For hAT: count of test species with hAT support >= 5, sorted by
     recall ascending. Reveals concentration vs uniform failure.
   - Mean gate weights cross-tabbed by (species, predicted SF).
   - Train/test species set check (must be disjoint).

Run::

    cd "data_analysis/vgp_model_split_fix/v4"
    python eval_per_species.py \\
        --ckpt "cluster session/hybrid_v4_epoch8.pt" \\
        --ckpt "cluster session/hybrid_v4_epoch30.pt"

Notes
-----
- Cost: dominated by the one-time k-mer featurization of the full filtered
  dataset (same as eval_epoch8). On MPS this is ~10-15 minutes; the
  inference passes themselves are <1 minute each.
- The script does *not* depend on the rolling-fold training driver;
  only on the definition cells.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve().parent
NB = THIS / "cluster session" / "vgp_hybrid_v4_gpu.ipynb"
WORKSPACE = THIS.parents[2]
FASTA_PATH = str(WORKSPACE / "data" / "vgp" / "all_vgp_tes.fa")
LABEL_PATH = str(WORKSPACE / "data" / "vgp" / "20260120_features_sf")

# Cells from the GPU notebook that contain only definitions / configs.
# Same list as eval_epoch8.py.
DEFINITION_CELLS = [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 23, 39]


def _exec_notebook_defs() -> dict:
    nb = json.load(open(NB))
    import types
    mod = types.ModuleType("__nbexec__")
    mod.__file__ = str(NB)
    sys.modules["__nbexec__"] = mod
    ns: dict = mod.__dict__
    ns["__name__"] = "__nbexec__"
    ns["__file__"] = str(NB)
    for i in DEFINITION_CELLS:
        src = "".join(nb["cells"][i]["source"])
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    return ns


def _species_from_header(h: str) -> str:
    return h.lstrip(">").split("#", 1)[0].rsplit("-", 1)[1]


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_test_split(ns: dict, device: torch.device, random_state: int = 42, test_size: float = 0.2):
    """Mirror cell 21: load -> filter -> non-DNA cap -> featurize -> split."""
    read_fasta = ns["read_fasta"]
    load_binary_dna_labels = ns["load_binary_dna_labels"]
    KmerWindowFeaturizer = ns["KmerWindowFeaturizer"]
    KmerWindowFeaturizerGPU = ns["KmerWindowFeaturizerGPU"]
    HybridDataset = ns["HybridDataset"]
    collate_hybrid = ns["collate_hybrid"]
    MIN_CLASS_COUNT = ns["MIN_CLASS_COUNT"]
    MAX_PER_SF = ns["MAX_PER_SF"]
    KMER_K = ns["KMER_K"]
    KMER_DIM = ns["KMER_DIM"]
    KMER_WINDOW = ns["KMER_WINDOW"]
    KMER_STRIDE = ns["KMER_STRIDE"]

    t0 = time.time()
    print("=== Loading data ===", flush=True)
    headers, sequences = read_fasta(FASTA_PATH)
    label_dict, class_dict = load_binary_dna_labels(LABEL_PATH)

    all_h, all_s, all_tags, all_top = [], [], [], []
    for h, s in zip(headers, sequences):
        if h not in label_dict:
            continue
        all_h.append(h); all_s.append(s)
        all_tags.append(label_dict[h]); all_top.append(class_dict[h])
    del headers, sequences; gc.collect()
    print(f"Matched {len(all_h)} sequences  [{time.time()-t0:.1f}s]")

    # SF mapping from DNA only
    dna_tags = [t for t, top in zip(all_tags, all_top) if top == 1]
    keep_sf = {t for t, c in Counter(dna_tags).items() if c >= MIN_CLASS_COUNT}
    superfamily_names = sorted(keep_sf)
    superfamily_to_id = {t: i for i, t in enumerate(superfamily_names)}
    print(f"Kept SFs ({len(superfamily_names)}): {superfamily_names}")

    SF_SENTINEL = 0
    fh, fs, ft, ftop, fsf = [], [], [], [], []
    for h, s, tag, top in zip(all_h, all_s, all_tags, all_top):
        if top == 0:
            fh.append(h); fs.append(s); ft.append(tag); ftop.append(0); fsf.append(SF_SENTINEL)
        elif tag in superfamily_to_id:
            fh.append(h); fs.append(s); ft.append(tag); ftop.append(1); fsf.append(superfamily_to_id[tag])
    all_h, all_s, all_tags = fh, fs, ft
    all_top = np.array(ftop, dtype=np.int64)
    all_sf = np.array(fsf, dtype=np.int64)
    del fh, fs, ft, ftop, fsf; gc.collect()
    print(f"Filtered: {len(all_h)} (DNA={int((all_top==1).sum())}, non-DNA={int((all_top==0).sum())})")

    # Global non-DNA cap (matches eval_epoch8.py)
    if MAX_PER_SF is not None:
        rng_cap = np.random.RandomState(random_state)
        nd_idx = np.where(all_top == 0)[0]
        d_idx = np.where(all_top == 1)[0]
        by_tag = defaultdict(list)
        for i in nd_idx:
            by_tag[all_tags[i]].append(int(i))
        keep_nd = []
        for tag, idxs in by_tag.items():
            if len(idxs) > MAX_PER_SF:
                idxs = rng_cap.choice(idxs, MAX_PER_SF, replace=False).tolist()
            keep_nd.extend(idxs)
        keep = sorted(d_idx.tolist() + keep_nd)
        all_h = [all_h[i] for i in keep]
        all_s = [all_s[i] for i in keep]
        all_tags = [all_tags[i] for i in keep]
        all_top = all_top[keep]
        all_sf = all_sf[keep]
        print(f"After non-DNA cap: {len(all_h)} (DNA={int((all_top==1).sum())}, non-DNA={int((all_top==0).sum())})")

    # Featurize
    print("\n=== K-mer featurization ===", flush=True)
    if device.type in ("cuda", "mps"):
        feat = KmerWindowFeaturizerGPU(
            k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW, stride=KMER_STRIDE,
            add_pos=True, l2_normalize=True, device=device,
        )
        print(f"  GPU featurizer on {device}")
    else:
        feat = KmerWindowFeaturizer(
            k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW, stride=KMER_STRIDE,
            add_pos=True, l2_normalize=True,
        )
        print("  CPU featurizer")
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
    print(f"Featurize done: {time.time()-t1:.1f}s")

    # Split
    species = np.array([_species_from_header(h) for h in all_h])
    strat = all_top * (len(superfamily_names) + 1) + all_sf

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_trainval, idx_test = next(gss.split(np.arange(len(all_h)), y=strat, groups=species))
    sp_tv, sp_te = set(species[idx_trainval]), set(species[idx_test])
    assert not (sp_tv & sp_te), f"Species leak trainval/test: {sorted(sp_tv & sp_te)}"
    print(f"\nSplit: trainval={len(idx_trainval)} ({len(sp_tv)} sp) | test={len(idx_test)} ({len(sp_te)} sp)")
    print(f"Test species: {sorted(sp_te)}")

    test_h = [all_h[i] for i in idx_test]
    test_s = [all_s[i] for i in idx_test]
    test_top = all_top[idx_test]
    test_sf = all_sf[idx_test]
    test_kmer = [all_kmer[i] for i in idx_test]
    test_species = species[idx_test]

    ds_test = HybridDataset(test_h, test_s, test_top, test_sf, test_kmer)
    loader_test = DataLoader(ds_test, batch_size=64, shuffle=False, collate_fn=collate_hybrid, num_workers=0)
    return loader_test, test_h, test_species, test_top, test_sf, superfamily_names


def run_inference(ns: dict, loader_test, ckpt_path: Path, device, superfamily_names):
    HybridTEClassifierV4 = ns["HybridTEClassifierV4"]
    DROPOUT = ns["DROPOUT"]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sf_names_ckpt = ckpt["superfamily_names"]
    assert sf_names_ckpt == superfamily_names, "SF mismatch with checkpoint!"

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
    print(f"  loaded {ckpt_path.name} (epoch={ckpt.get('epoch','?')}, score={ckpt.get('score',float('nan')):.4f})")

    cls_pred, sf_pred, gate_w = [], [], []
    with torch.no_grad():
        for _, X_cnn, mask, Y_cls, Y_sf, x_gnn, edge_index, batch_vec in loader_test:
            X_cnn = X_cnn.to(device); mask = mask.to(device)
            x_gnn = x_gnn.to(device); edge_index = edge_index.to(device)
            batch_vec = batch_vec.to(device)
            class_logits, sf_logits, gw = model(X_cnn, mask, x_gnn, edge_index, batch_vec)
            cls_pred.append(class_logits.argmax(1).cpu().numpy())
            sf_pred.append(sf_logits.argmax(1).cpu().numpy())
            gate_w.append(gw.cpu().numpy())
    cls_pred = np.concatenate(cls_pred)
    sf_pred = np.concatenate(sf_pred)
    gate_w = np.concatenate(gate_w, axis=0)
    return cls_pred, sf_pred, gate_w


def report_per_species(df: pd.DataFrame, superfamily_names, label: str):
    print("\n" + "=" * 72)
    print(f"PER-SPECIES DIAGNOSTIC: {label}")
    print("=" * 72)

    dna = df[df["top_true"] == 1]
    print(f"\nDNA test samples: {len(dna)} across {dna['species'].nunique()} species")

    # Overall SF macro F1 (recompute for sanity)
    p, r, f, s = precision_recall_fscore_support(
        dna["sf_true"], dna["sf_pred"],
        labels=list(range(len(superfamily_names))), zero_division=0,
    )
    print("\nOverall SF (DNA-only):")
    print(f"  {'sf':<20s} {'P':>6s} {'R':>6s} {'F1':>6s} {'n':>6s}")
    for n, pi, ri, fi, si in zip(superfamily_names, p, r, f, s):
        print(f"  {n:<20s} {pi:6.3f} {ri:6.3f} {fi:6.3f} {si:6d}")
    print(f"  macro-F1 = {np.mean(f):.4f}")

    # Per-species breakdown for each SF
    for sf_id, sf_name in enumerate(superfamily_names):
        sub = dna[dna["sf_true"] == sf_id]
        if sub.empty:
            continue
        per_sp = []
        for sp, g in sub.groupby("species"):
            n_true = len(g)
            if n_true < 5:
                continue
            tp = int((g["sf_pred"] == sf_id).sum())
            recall = tp / n_true
            # precision over all preds == sf_id within this species
            pred_pos = dna[(dna["species"] == sp) & (dna["sf_pred"] == sf_id)]
            prec = (int((pred_pos["sf_true"] == sf_id).sum()) / len(pred_pos)) if len(pred_pos) else float("nan")
            per_sp.append((sp, n_true, recall, prec))
        if not per_sp:
            continue
        per_sp.sort(key=lambda r: r[2])  # ascending recall
        print(f"\n  >>> SF={sf_name}  (species with >=5 true positives) <<<")
        print(f"      {'species':<20s} {'n_true':>7s} {'recall':>7s} {'prec':>7s}")
        for sp, n_true, recall, prec in per_sp:
            print(f"      {sp:<20s} {n_true:7d} {recall:7.3f} {prec:7.3f}")
        # Concentration: fraction of misses from worst 2 species
        misses = [(sp, int(n * (1 - r))) for sp, n, r, _ in per_sp]
        total_miss = sum(m for _, m in misses)
        if total_miss:
            misses.sort(key=lambda x: -x[1])
            top2 = sum(m for _, m in misses[:2])
            print(f"      [concentration] worst-2 species account for "
                  f"{top2}/{total_miss} = {top2/total_miss:.0%} of misses")

    # Gate weight cross-tab
    print("\nMean gate weight (w_gnn) by (species, predicted SF) for DNA samples:")
    pivot = (dna.assign(sf_pred_name=dna["sf_pred"].map({i: n for i, n in enumerate(superfamily_names)}))
                .pivot_table(index="species", columns="sf_pred_name", values="w_gnn", aggfunc="mean"))
    with pd.option_context("display.float_format", "{:.2f}".format,
                           "display.max_columns", 20,
                           "display.width", 200):
        print(pivot.fillna(np.nan).round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="Path to a hybrid_v4_*.pt checkpoint (may be repeated).")
    ap.add_argument("--out-dir", default=str(THIS), help="Where to write CSVs.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint paths BEFORE we chdir into "cluster session"
    ckpt_paths = [Path(c).expanduser().resolve() for c in args.ckpt]

    sys.path.insert(0, str(THIS / "cluster session"))
    os.chdir(THIS / "cluster session")

    device = _resolve_device()
    print(f"Device: {device}")

    ns = _exec_notebook_defs()
    loader_test, test_h, test_species, test_top, test_sf, superfamily_names = build_test_split(ns, device)

    for ckpt_path in ckpt_paths:
        if not ckpt_path.is_file():
            print(f"[skip] missing checkpoint: {ckpt_path}")
            continue
        print(f"\n=== Inference: {ckpt_path.name} ===")
        cls_pred, sf_pred, gate_w = run_inference(ns, loader_test, ckpt_path, device, superfamily_names)

        df = pd.DataFrame({
            "header": test_h,
            "species": test_species,
            "top_true": test_top,
            "top_pred": cls_pred,
            "sf_true": test_sf,
            "sf_pred": sf_pred,
            "w_cnn": gate_w[:, 0],
            "w_gnn": gate_w[:, 1],
        })
        out_csv = out_dir / f"per_species_{ckpt_path.stem}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  wrote {out_csv}  ({len(df)} rows)")

        report_per_species(df, superfamily_names, label=ckpt_path.name)


if __name__ == "__main__":
    main()
