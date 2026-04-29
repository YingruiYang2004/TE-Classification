"""Round-2 unified smoke driver.

Three competing tracks share one driver. Per-track wiring lives in
`_common/models.py`; per-track behaviour is selected with `--track`.

  * --track T1 : V4 (unchanged) + invariance bolt-ons
  * --track T2 : CNN-only (no GNN) + invariance bolt-ons
  * --track T3 : V4 with GNN replaced by CompResEncoder + invariance bolt-ons

`--no-stack` disables DANN + Group-DRO + phylo sampler (control arm).
`--ab` runs both arms (with-stack then no-stack) and writes a delta JSON.

Smoke params are deliberately small so all 6 runs (3 tracks x 2 arms)
fit in <8h on Apple Silicon MPS.

Usage::

    cd data_analysis/vgp_model_split_fix/v4/round2
    python run_smoke.py --track T1 --ab       # ~1h on MPS
    python run_smoke.py --track T2 --no-stack
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

# Allow `python run_smoke.py` from inside the round2 dir.
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from _common.data_prep import prepare_data, exec_notebook_defs, resolve_device  # noqa: E402
from _common.dann import SpeciesHead, lambda_warmup  # noqa: E402
from _common.group_dro import GroupDROLoss  # noqa: E402
from _common.phylo_sampler import make_clade_sf_sampler  # noqa: E402
from _common.models import V4Wrapper, CnnOnlyClassifier, CompResEncoder, V4CompResWrapper  # noqa: E402

# ------------------------------------------------------------------ #
# SMOKE PARAMS (defaults; overridable via CLI for Round-2.5 sweep)    #
# ------------------------------------------------------------------ #
EPOCHS = 3
SUBSET_SIZE = 5000
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-4
SF_DROPOUT = 0.3
LABEL_SMOOTHING_SF = 0.1
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.25
DANN_MAX_LAMBDA = 0.5
DANN_WARMUP_EPOCHS = 1
GDRO_ETA = 0.05
USE_COSINE = False
TAG = ""  # suffix appended to output filenames (e.g. "_6ep", "_weak1")


# ----------------------------------------------------------------- helpers
def make_collate_with_groups(base_collate, header_to_species, header_to_clade,
                             *, fixed_length: int):
    """Wrap collate_hybrid: also return species_ids and clade_ids per sample."""
    def _coll(batch):
        out = base_collate(batch, fixed_length=fixed_length)
        headers = out[0]
        sp_ids = torch.tensor([header_to_species.get(h, -1) for h in headers],
                              dtype=torch.long)
        cl_ids = torch.tensor([header_to_clade.get(h, -1) for h in headers],
                              dtype=torch.long)
        return (*out, sp_ids, cl_ids)
    return _coll


def build_model(track: str, ns: dict, n_classes: int, n_sf: int, device,
                sf_dropout: float, n_clades: int):
    """Instantiate the per-track model wrapper."""
    HybridTEClassifierV4 = ns["HybridTEClassifierV4"]
    CNNTower = ns["CNNTower"]
    CNN_WIDTH = ns["CNN_WIDTH"]
    MOTIF_KERNELS = ns["MOTIF_KERNELS"]
    CONTEXT_DILATIONS = ns["CONTEXT_DILATIONS"]
    RC_FUSION_MODE = ns["RC_FUSION_MODE"]
    KMER_DIM = ns["KMER_DIM"]
    GNN_HIDDEN = ns["GNN_HIDDEN"]
    GNN_LAYERS = ns["GNN_LAYERS"]
    FUSION_DIM = ns["FUSION_DIM"]
    NUM_HEADS = ns["NUM_HEADS"]
    DROPOUT = ns["DROPOUT"]

    def _new_v4_base():
        return HybridTEClassifierV4(
            num_classes=n_classes,
            num_superfamilies=n_sf,
            cnn_width=CNN_WIDTH,
            motif_kernels=tuple(MOTIF_KERNELS),
            context_dilations=tuple(CONTEXT_DILATIONS),
            rc_mode=RC_FUSION_MODE,
            gnn_in_dim=KMER_DIM + 1,
            gnn_hidden=GNN_HIDDEN,
            gnn_layers=GNN_LAYERS,
            fusion_dim=FUSION_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
        )

    def _bump_sf_dropout(model):
        for m in model.superfamily_head.modules():
            if isinstance(m, nn.Dropout):
                m.p = sf_dropout

    if track == "T1":
        base = _new_v4_base()
        _bump_sf_dropout(base)
        wrap = V4Wrapper(base)
    elif track == "T2":
        cnn = CNNTower(
            width=CNN_WIDTH,
            motif_kernels=tuple(MOTIF_KERNELS),
            context_dilations=tuple(CONTEXT_DILATIONS),
            dropout=DROPOUT,
            rc_mode=RC_FUSION_MODE,
        )
        wrap = CnnOnlyClassifier(cnn, n_classes, n_sf,
                                 cnn_dim=CNN_WIDTH, fusion_dim=FUSION_DIM,
                                 dropout=DROPOUT)
        _bump_sf_dropout(wrap)
    elif track == "T3":
        base = _new_v4_base()
        _bump_sf_dropout(base)
        compres = CompResEncoder(
            in_dim=KMER_DIM + 1,
            hidden=GNN_HIDDEN,
            clade_centroid_momentum=0.0,  # per-batch mean for smoke
            n_clades=n_clades,
            dropout=DROPOUT,
        )
        wrap = V4CompResWrapper(base, compres)
    else:
        raise ValueError(f"unknown track {track!r}")

    return wrap.to(device)


def evaluate(wrap, loader, device, n_sf, superfamily_names, *, track):
    wrap.eval()
    cls_p, cls_t, sf_p, sf_t, gw, hdrs = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            (headers, X_cnn, mask, Y_cls, Y_sf, x_gnn, edge_index, batch_vec,
             sp_ids, cl_ids) = batch
            X_cnn = X_cnn.to(device); mask = mask.to(device)
            x_gnn = x_gnn.to(device); edge_index = edge_index.to(device)
            batch_vec = batch_vec.to(device)
            if track == "T3":
                # Use per-batch centroid in eval (centroid_momentum=0).
                wrap.set_clade_ids(None)
            out = wrap(X_cnn, mask, x_gnn, edge_index, batch_vec)
            cls_p.append(out["class"].argmax(1).cpu().numpy())
            cls_t.append(Y_cls.numpy())
            sf_p.append(out["sf"].argmax(1).cpu().numpy())
            sf_t.append(Y_sf.numpy())
            if out["gate"] is not None:
                gw.append(out["gate"].cpu().numpy())
            hdrs.extend(headers)
    cls_p = np.concatenate(cls_p); cls_t = np.concatenate(cls_t)
    sf_p = np.concatenate(sf_p); sf_t = np.concatenate(sf_t)
    cls_macro = f1_score(cls_t, cls_p, average="macro", zero_division=0)
    dna = (cls_t == 1)
    if dna.any():
        p, r, f, s = precision_recall_fscore_support(
            sf_t[dna], sf_p[dna],
            labels=list(range(n_sf)), zero_division=0,
        )
        sf_macro = float(np.mean(f))
        per_sf = {n: {"P": float(pi), "R": float(ri), "F1": float(fi), "n": int(si)}
                  for n, pi, ri, fi, si in zip(superfamily_names, p, r, f, s)}
    else:
        sf_macro = float("nan"); per_sf = {}
    gate_mean = (np.concatenate(gw, 0).mean(0).tolist() if gw else None)
    return {
        "cls_macro_f1": float(cls_macro),
        "sf_macro_f1": sf_macro,
        "per_sf": per_sf,
        "gate_mean": gate_mean,
        "n": int(len(cls_p)),
        "headers": hdrs,
        "cls_pred": cls_p.tolist(),
        "cls_true": cls_t.tolist(),
        "sf_pred": sf_p.tolist(),
        "sf_true": sf_t.tolist(),
    }


# ----------------------------------------------------------------- arm
def run_one_arm(*, track: str, with_stack: bool, ns: dict, data: dict,
                device, log_path: Path, out_dir: Path):
    print(f"\n>>> arm track={track} stack={'ON' if with_stack else 'OFF'} "
          f"device={device}", flush=True)

    superfamily_names = data["superfamily_names"]
    n_sf = len(superfamily_names)
    n_classes = 2
    n_clades = data["n_train_clades"]
    n_species = data["n_train_species"]

    ds_tr = data["ds_tr"]; ds_val = data["ds_val"]; ds_te = data["ds_te"]

    # Header -> species/clade-id lookup for the augmented collate.
    species_id_arr = data["species_id_arr"]
    clade_id_arr = data["clade_id_arr"]
    # idx_tr/val/te are positions into the post-cap arrays; ds_tr.headers has
    # the matching subset of headers in the same order.
    idx_tr = data["idx_tr"]; idx_val = data["idx_val"]; idx_te = data["idx_te"]
    header_to_species = {}
    header_to_clade = {}
    for ds, idxs in [(ds_tr, idx_tr), (ds_val, idx_val), (ds_te, idx_te)]:
        for h, gi in zip(ds.headers, idxs):
            header_to_species[h] = int(species_id_arr[gi])
            header_to_clade[h] = int(clade_id_arr[gi])

    ENCODE = ns["ENCODE"]; FIXED_LENGTH = ns["FIXED_LENGTH"]
    collate_hybrid = ns["collate_hybrid"]
    coll = make_collate_with_groups(collate_hybrid, header_to_species,
                                    header_to_clade, fixed_length=FIXED_LENGTH)

    # Sampler.
    if with_stack:
        train_clades = np.array([header_to_clade[h] for h in ds_tr.headers])
        train_sfs = ds_tr.class_labels
        train_top = ds_tr.binary_labels
        # Note: phylo sampler takes the string clade; we just give the
        # int clade ids (cast to str) since it only does Counter buckets.
        sampler = make_clade_sf_sampler(
            clades=np.array([str(c) for c in train_clades]),
            sfs=train_sfs,
            binary_labels=train_top,
        )
        loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler,
                               collate_fn=coll, num_workers=0)
    else:
        loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=coll, num_workers=0)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=coll, num_workers=0)
    loader_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=coll, num_workers=0)

    wrap = build_model(track, ns, n_classes, n_sf, device, SF_DROPOUT, n_clades)
    n_params = sum(p.numel() for p in wrap.parameters())
    print(f"  params: {n_params:,}", flush=True)

    # Class weights (from train).
    tr_top = ds_tr.binary_labels
    cls_w = torch.tensor(
        [len(tr_top) / (2 * max((tr_top == c).sum(), 1)) for c in range(n_classes)],
        dtype=torch.float32, device=device,
    )
    # Effective-number-of-samples weighting for SF head (Cui et al. 2019).
    # Sharper than plain inverse-frequency; lifts gradient on rare SFs
    # (Academ-1, PIF-Harbinger, CMC) which dominate the macro-F1 bottleneck.
    tr_sf_dna = ds_tr.class_labels[ds_tr.binary_labels == 1]
    SF_BETA = 0.999
    sf_counts = np.array([max((tr_sf_dna == c).sum(), 1) for c in range(n_sf)],
                         dtype=np.float64)
    eff_num = 1.0 - np.power(SF_BETA, sf_counts)
    sf_w_np = (1.0 - SF_BETA) / eff_num
    sf_w_np = sf_w_np / sf_w_np.sum() * n_sf  # normalise so mean weight = 1
    sf_w = torch.tensor(sf_w_np, dtype=torch.float32, device=device)
    print(f"  sf class weights (effective-number, beta={SF_BETA}):", flush=True)
    for nm, ct, w in zip(superfamily_names, sf_counts.astype(int), sf_w_np):
        print(f"    {nm:<22s} n={int(ct):<6d} w={w:.3f}", flush=True)

    cls_loss_fn = nn.CrossEntropyLoss(weight=cls_w)
    # SF loss is per-sample (reduction='none') so Group-DRO can reweight.
    sf_loss_fn = nn.CrossEntropyLoss(weight=sf_w, label_smoothing=LABEL_SMOOTHING_SF,
                                     reduction="none")

    species_head = SpeciesHead(in_dim=ns["FUSION_DIM"], n_species=n_species).to(device)
    sp_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    gdro = GroupDROLoss(n_groups=n_clades, eta=GDRO_ETA, device=device)

    params = list(wrap.parameters())
    if with_stack:
        params += list(species_head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    # Cosine LR decay to LR/10 over EPOCHS (settles the gate oscillation
    # we saw in the 25ep run where loss was flat but cls/SF F1 swung
    # ~0.05 between epochs at constant LR).
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(EPOCHS, 1), eta_min=LR * 0.1,
    ) if USE_COSINE else None

    history = []
    log_lines: list[str] = []
    best_val_sf = -1.0
    best_state = None
    best_epoch = -1

    def _log(m: str):
        print(m, flush=True); log_lines.append(m)

    _log(f"=== Training (track={track}, stack={with_stack}, epochs={EPOCHS}) ===")
    for ep in range(1, EPOCHS + 1):
        if with_stack:
            species_head.set_lambda(lambda_warmup(ep, DANN_MAX_LAMBDA, DANN_WARMUP_EPOCHS))
        wrap.train(); species_head.train()
        t0 = time.time(); running = 0.0; n_seen = 0
        for batch in loader_tr:
            (_, X_cnn, mask, Y_cls, Y_sf, x_gnn, edge_index, batch_vec,
             sp_ids, cl_ids) = batch
            X_cnn = X_cnn.to(device); mask = mask.to(device)
            x_gnn = x_gnn.to(device); edge_index = edge_index.to(device)
            batch_vec = batch_vec.to(device)
            Y_cls = Y_cls.to(device); Y_sf = Y_sf.to(device)
            sp_ids = sp_ids.to(device); cl_ids = cl_ids.to(device)

            if track == "T3":
                wrap.set_clade_ids(cl_ids.clamp_min(0))  # sentinel-safe

            out = wrap(X_cnn, mask, x_gnn, edge_index, batch_vec)
            loss_cls = cls_loss_fn(out["class"], Y_cls)
            dna = (Y_cls == 1)
            if dna.any():
                per_sample = sf_loss_fn(out["sf"][dna], Y_sf[dna])
                if with_stack:
                    valid = cl_ids[dna] >= 0
                    if valid.any():
                        loss_sf = gdro.weighted_mean(per_sample[valid], cl_ids[dna][valid])
                    else:
                        loss_sf = per_sample.mean()
                else:
                    loss_sf = per_sample.mean()
            else:
                loss_sf = torch.zeros((), device=device)

            if with_stack:
                sp_logits = species_head(out["fused"])
                loss_sp = sp_loss_fn(sp_logits, sp_ids)
            else:
                loss_sp = torch.zeros((), device=device)

            loss = loss_cls + loss_sf + loss_sp
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            bs = X_cnn.size(0)
            running += float(loss.item()) * bs; n_seen += bs

        train_loss = running / max(n_seen, 1)
        val_metrics = evaluate(wrap, loader_val, device, n_sf, superfamily_names, track=track)
        elapsed = time.time() - t0
        gate_str = (f"({val_metrics['gate_mean'][0]:.2f},{val_metrics['gate_mean'][1]:.2f})"
                    if val_metrics["gate_mean"] else "n/a")
        cur_lr = opt.param_groups[0]["lr"]
        _log(f"  ep {ep}/{EPOCHS} | lr {cur_lr:.2e} | loss {train_loss:.4f} "
             f"| val cls F1 {val_metrics['cls_macro_f1']:.4f} "
             f"| val SF F1 {val_metrics['sf_macro_f1']:.4f} "
             f"| gate {gate_str} | {elapsed:.0f}s")
        if sched is not None:
            sched.step()
        if "DNA/hAT" in val_metrics["per_sf"]:
            h = val_metrics["per_sf"]["DNA/hAT"]
            _log(f"      hAT P={h['P']:.3f} R={h['R']:.3f} F1={h['F1']:.3f} n={h['n']}")
        # Drop big arrays before stashing in history (keep summary only).
        slim = {k: v for k, v in val_metrics.items() if k not in
                ("headers", "cls_pred", "cls_true", "sf_pred", "sf_true")}
        history.append({"epoch": ep, "train_loss": train_loss, **slim})

        # Track best by val SF macro F1 (the actual target metric).
        cur_sf = val_metrics["sf_macro_f1"]
        if cur_sf == cur_sf and cur_sf > best_val_sf:  # NaN-safe
            best_val_sf = float(cur_sf)
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in wrap.state_dict().items()}
            _log(f"      [best] val SF F1 {best_val_sf:.4f} @ ep {ep}")

    if best_state is not None:
        wrap.load_state_dict(best_state)
        _log(f"\n=== Loaded best checkpoint (ep {best_epoch}, val SF F1 {best_val_sf:.4f}) ===")
    else:
        _log("\n=== WARNING: no best checkpoint captured; using last epoch ===")

    test_metrics = evaluate(wrap, loader_te, device, n_sf, superfamily_names, track=track)
    _log("\n=== Held-out test (species-disjoint) ===")
    _log(f"  cls macro F1: {test_metrics['cls_macro_f1']:.4f}")
    _log(f"  SF macro F1 : {test_metrics['sf_macro_f1']:.4f}")
    if test_metrics["gate_mean"]:
        _log(f"  gate mean   : (cnn={test_metrics['gate_mean'][0]:.3f}, "
             f"gnn={test_metrics['gate_mean'][1]:.3f})")
    for name, m in test_metrics["per_sf"].items():
        _log(f"  {name:<22s} P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} n={m['n']}")

    # Per-clade hAT recall (the diagnostic-driven smoke acceptance signal).
    test_headers = test_metrics["headers"]
    sf_t = np.array(test_metrics["sf_true"])
    sf_p = np.array(test_metrics["sf_pred"])
    cls_t = np.array(test_metrics["cls_true"])
    if "DNA/hAT" in superfamily_names:
        hat_id = superfamily_names.index("DNA/hAT")
        is_hat = (cls_t == 1) & (sf_t == hat_id)
        clades = np.array([header_to_clade.get(h, -1) for h in test_headers])
        per_clade_hat = {}
        for c in sorted(set(clades[is_hat].tolist())):
            sel = is_hat & (clades == c)
            n = int(sel.sum())
            r = float((sf_p[sel] == hat_id).mean()) if n else float("nan")
            per_clade_hat[str(c)] = {"n": n, "recall": r}
        _log("  per-clade hAT recall (test, clade-id):")
        for c, m in per_clade_hat.items():
            _log(f"    clade={c} n={m['n']} R={m['recall']:.3f}")
        test_metrics["per_clade_hat"] = per_clade_hat

    log_path.write_text("\n".join(log_lines) + "\n")
    arm_tag = "stack" if with_stack else "nostack"
    out_path = out_dir / f"smoke_{track}_{arm_tag}{TAG}.json"
    out_path.write_text(json.dumps({
        "track": track, "stack": with_stack,
        "epochs": EPOCHS, "subset_size": SUBSET_SIZE,
        "best_epoch": best_epoch,
        "best_val_sf_macro_f1": best_val_sf,
        "history": history,
        "test": {k: v for k, v in test_metrics.items() if k not in
                 ("headers",)},
    }, indent=2, default=float))
    print(f"  wrote {out_path}", flush=True)
    return test_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["T1", "T2", "T3"], required=True)
    ap.add_argument("--no-stack", action="store_true",
                    help="control arm: no DANN/DRO/sampler.")
    ap.add_argument("--ab", action="store_true",
                    help="run both arms (stack ON then OFF) and write delta.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset-size", type=int, default=None)
    ap.add_argument("--dann-lambda", type=float, default=None,
                    help="DANN max lambda (default 0.5).")
    ap.add_argument("--dann-warmup", type=int, default=None)
    ap.add_argument("--gdro-eta", type=float, default=None,
                    help="Group-DRO step size (default 0.05).")
    ap.add_argument("--cosine", action="store_true",
                    help="Use cosine LR schedule decaying to LR/10 over EPOCHS.")
    ap.add_argument("--tag", default="",
                    help="Suffix for output filenames, e.g. '_6ep'.")
    args = ap.parse_args()

    global EPOCHS, SUBSET_SIZE, DANN_MAX_LAMBDA, DANN_WARMUP_EPOCHS, GDRO_ETA, USE_COSINE, TAG
    if args.epochs is not None: EPOCHS = args.epochs
    if args.subset_size is not None:
        # Treat 0 / negative as "use full dataset" (passed as None to prepare_data).
        SUBSET_SIZE = args.subset_size if args.subset_size > 0 else None
    if args.dann_lambda is not None: DANN_MAX_LAMBDA = args.dann_lambda
    if args.dann_warmup is not None: DANN_WARMUP_EPOCHS = args.dann_warmup
    if args.gdro_eta is not None: GDRO_ETA = args.gdro_eta
    if args.cosine: USE_COSINE = True
    if args.tag:
        TAG = args.tag if args.tag.startswith("_") else "_" + args.tag
    print(f"[params] EPOCHS={EPOCHS} SUBSET_SIZE={SUBSET_SIZE} "
          f"DANN_LAMBDA={DANN_MAX_LAMBDA} DANN_WARMUP={DANN_WARMUP_EPOCHS} "
          f"GDRO_ETA={GDRO_ETA} COSINE={USE_COSINE} TAG={TAG!r}", flush=True)

    device = resolve_device()
    print(f"Device: {device}", flush=True)
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}  "
          f"MPS: {hasattr(torch.backends,'mps') and torch.backends.mps.is_available()}",
          flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else THIS / "results" / args.track
    out_dir.mkdir(parents=True, exist_ok=True)

    ns = exec_notebook_defs()
    data = prepare_data(ns, device, subset_size=SUBSET_SIZE,
                        random_state=RANDOM_STATE,
                        test_size=TEST_SIZE, val_size=VAL_SIZE)

    arms: list[bool] = []
    if args.ab:
        arms = [True, False]
    elif args.no_stack:
        arms = [False]
    else:
        arms = [True]

    results = {}
    for with_stack in arms:
        tag = "stack" if with_stack else "nostack"
        log_path = out_dir / f"smoke_{args.track}_{tag}{TAG}.log"
        results[tag] = run_one_arm(
            track=args.track, with_stack=with_stack,
            ns=ns, data=data, device=device,
            log_path=log_path, out_dir=out_dir,
        )

    if args.ab:
        d_sf = results["stack"]["sf_macro_f1"] - results["nostack"]["sf_macro_f1"]
        d_cls = results["stack"]["cls_macro_f1"] - results["nostack"]["cls_macro_f1"]
        delta = {
            "track": args.track,
            "delta_sf_macro_f1": d_sf,
            "delta_cls_macro_f1": d_cls,
            "stack_sf_macro_f1": results["stack"]["sf_macro_f1"],
            "nostack_sf_macro_f1": results["nostack"]["sf_macro_f1"],
            "stack_per_clade_hat": results["stack"].get("per_clade_hat"),
            "nostack_per_clade_hat": results["nostack"].get("per_clade_hat"),
        }
        (out_dir / f"smoke_{args.track}_delta{TAG}.json").write_text(
            json.dumps(delta, indent=2, default=float))
        print(f"\nA/B delta (stack - nostack) for {args.track}: "
              f"SF={d_sf:+.4f}  cls={d_cls:+.4f}", flush=True)


if __name__ == "__main__":
    main()
