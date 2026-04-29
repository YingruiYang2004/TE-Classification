"""Round-2 data prep: same species-disjoint splits as Round 1, plus
clade-prefix and group-id arrays needed by Group-DRO and the phylo
sampler.

Mirrors `smoke_aug.run_smoke_aug.prepare_smoke_data` but augmented with:
  * per-sample clade prefix
  * per-sample group id (clade x sf for DNA, clade for non-DNA)
  * per-sample species id (label-encoded for the DANN species head)

The test set is the FROZEN seed-42 species-disjoint split — keep the
same `RANDOM_STATE`, `TEST_SIZE`, and `VAL_SIZE` constants as Round 1.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from .phylo_sampler import species_clade

THIS = Path(__file__).resolve()
ROUND2_DIR = THIS.parents[1]
V4_DIR = ROUND2_DIR.parent
WORKSPACE = V4_DIR.parents[2]

# Default paths (workstation layout). Override via env vars on CSD3 / clusters
# where round2/ is deployed standalone (e.g. ~/TEs/round2/) and the
# data_analysis tree is not mirrored:
#   TE_NOTEBOOK_PATH = absolute path to vgp_hybrid_v4_gpu.ipynb
#   TE_FASTA_PATH    = absolute path to all_vgp_tes.fa
#   TE_LABEL_PATH    = absolute path to features label file
import os as _os
NB = Path(_os.environ.get(
    "TE_NOTEBOOK_PATH",
    str(V4_DIR / "cluster session" / "vgp_hybrid_v4_gpu.ipynb"),
))
FASTA_PATH = _os.environ.get(
    "TE_FASTA_PATH",
    str(WORKSPACE / "data" / "vgp" / "all_vgp_tes.fa"),
)
LABEL_PATH = _os.environ.get(
    "TE_LABEL_PATH",
    str(WORKSPACE / "data" / "vgp" / "20260120_features_sf"),
)

# Same definition cells as Round 1.
DEFINITION_CELLS = [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 23, 39]


def exec_notebook_defs() -> dict:
    """Execute the v4 notebook's definition cells into a fresh module ns."""
    nb = json.load(open(NB))
    mod = types.ModuleType("__nbexec_round2__")
    mod.__file__ = str(NB)
    sys.modules["__nbexec_round2__"] = mod
    ns: dict = mod.__dict__
    ns["__name__"] = "__nbexec_round2__"
    ns["__file__"] = str(NB)
    for i in DEFINITION_CELLS:
        src = "".join(nb["cells"][i]["source"])
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    return ns


def species_from_header(h: str) -> str:
    return h.lstrip(">").split("#", 1)[0].rsplit("-", 1)[1]


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_data(
    ns: dict,
    device: torch.device,
    *,
    subset_size: int | None,
    random_state: int = 42,
    test_size: float = 0.20,
    val_size: float = 0.25,
):
    """Load + featurize + split. Returns a dict with everything tracks need."""
    read_fasta = ns["read_fasta"]
    load_binary_dna_labels = ns["load_binary_dna_labels"]
    KmerWindowFeaturizer = ns["KmerWindowFeaturizer"]
    KmerWindowFeaturizerGPU = ns["KmerWindowFeaturizerGPU"]
    HybridDataset = ns["HybridDataset"]
    MIN_CLASS_COUNT = ns["MIN_CLASS_COUNT"]
    MAX_PER_SF = ns["MAX_PER_SF"]
    KMER_K = ns["KMER_K"]
    KMER_DIM = ns["KMER_DIM"]
    KMER_WINDOW = ns["KMER_WINDOW"]
    KMER_STRIDE = ns["KMER_STRIDE"]

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

    dna_tags = [t for t, top in zip(all_tags, all_top) if top == 1]
    keep_sf = {t for t, c in Counter(dna_tags).items() if c >= MIN_CLASS_COUNT}
    superfamily_names = sorted(keep_sf)
    superfamily_to_id = {t: i for i, t in enumerate(superfamily_names)}
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
    print(f"After cap: {len(all_h)} (DNA={int((all_top==1).sum())}, "
          f"non-DNA={int((all_top==0).sum())})", flush=True)

    if subset_size is not None and subset_size < len(all_h):
        rng_sub = np.random.default_rng(random_state)
        sel = np.sort(rng_sub.choice(len(all_h), subset_size, replace=False))
        all_h = [all_h[i] for i in sel]
        all_s = [all_s[i] for i in sel]
        all_top = all_top[sel]
        all_sf = all_sf[sel]
        print(f"Subset: {len(all_h)} sequences", flush=True)

    species_arr = np.array([species_from_header(h) for h in all_h])
    clade_arr = np.array([species_clade(sp) for sp in species_arr])
    n_species = len(set(species_arr))
    n_clades = len(set(clade_arr))
    print(f"Species in subset: {n_species} | clades: {sorted(set(clade_arr))}",
          flush=True)

    # Featurize.
    print("=== K-mer featurization ===", flush=True)
    if device.type in ("cuda", "mps"):
        feat = KmerWindowFeaturizerGPU(
            k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW, stride=KMER_STRIDE,
            add_pos=True, l2_normalize=True, device=device,
        )
    else:
        feat = KmerWindowFeaturizer(
            k=KMER_K, dim=KMER_DIM, window=KMER_WINDOW, stride=KMER_STRIDE,
            add_pos=True, l2_normalize=True,
        )
    all_kmer = []
    n_total = len(all_s)
    pe = max(500, n_total // 10)
    t0 = time.time()
    for i, seq in enumerate(all_s, 1):
        X, _ = feat.featurize_sequence(seq)
        all_kmer.append(X.astype(np.float16, copy=False))
        if i % pe == 0 or i == n_total:
            print(f"  [{i}/{n_total}] {(time.time()-t0):.1f}s", flush=True)

    # Splits (frozen seed-42 species-disjoint test, same as Round 1).
    strat = all_top * (len(superfamily_names) + 1) + all_sf
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_trainval, idx_te = next(gss_outer.split(np.arange(len(all_h)), y=strat, groups=species_arr))
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    inner_tr, inner_val = next(gss_inner.split(
        idx_trainval, y=strat[idx_trainval], groups=species_arr[idx_trainval]))
    idx_tr = idx_trainval[inner_tr]
    idx_val = idx_trainval[inner_val]

    sp_tr, sp_val, sp_te = set(species_arr[idx_tr]), set(species_arr[idx_val]), set(species_arr[idx_te])
    assert not (sp_tr & sp_val), "leak train/val"
    assert not (sp_tr & sp_te), "leak train/test"
    assert not (sp_val & sp_te), "leak val/test"
    print(f"Splits: train={len(idx_tr)} ({len(sp_tr)}sp) | "
          f"val={len(idx_val)} ({len(sp_val)}sp) | "
          f"test={len(idx_te)} ({len(sp_te)}sp)", flush=True)

    # Train-only species/clade label encoders (val/test species are unseen,
    # so the species head is only trained / evaluated on train species; for
    # val/test we will pass a sentinel id and ignore the species loss there).
    train_species = sorted(set(species_arr[idx_tr]))
    sp_to_id = {sp: i for i, sp in enumerate(train_species)}
    train_clades = sorted(set(clade_arr[idx_tr]))
    cl_to_id = {c: i for i, c in enumerate(train_clades)}
    SPECIES_SENTINEL = -1
    CLADE_SENTINEL = -1

    species_id_arr = np.array(
        [sp_to_id.get(sp, SPECIES_SENTINEL) for sp in species_arr], dtype=np.int64
    )
    clade_id_arr = np.array(
        [cl_to_id.get(cl, CLADE_SENTINEL) for cl in clade_arr], dtype=np.int64
    )

    def _make(idxs):
        return HybridDataset(
            [all_h[i] for i in idxs],
            [all_s[i] for i in idxs],
            all_top[idxs],
            all_sf[idxs],
            [all_kmer[i] for i in idxs],
        )
    ds_tr = _make(idx_tr)
    ds_val = _make(idx_val)
    ds_te = _make(idx_te)

    return {
        "ds_tr": ds_tr, "ds_val": ds_val, "ds_te": ds_te,
        "idx_tr": idx_tr, "idx_val": idx_val, "idx_te": idx_te,
        "all_top": all_top, "all_sf": all_sf,
        "species_arr": species_arr, "clade_arr": clade_arr,
        "species_id_arr": species_id_arr, "clade_id_arr": clade_id_arr,
        "n_train_species": len(train_species),
        "n_train_clades": len(train_clades),
        "train_clades": train_clades,
        "superfamily_names": superfamily_names,
    }
