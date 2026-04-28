"""CLI runner for causal saliency experiments on the v4.3 hybrid CNN+GNN
model (`hybrid_v4.3_epoch40.pt`).

Saliency / occlusion / sufficiency are computed against the **superfamily
head** (23 classes), since that is the granularity the thesis Section 3.6
saliency story is at (LTR/Gypsy <-> LTR/Pao confusions etc.).

Outputs follow the same layout as `run_causal_saliency.py`: one .npz per
sequence under `model_result_interp/interpretation_results/causal_saliency_hybrid/raw/`
plus a single `index.json`.

Resumable: existing .npz files are skipped and re-summarised into the index.

Usage:
    ./.venv/bin/python model_result_interp/run_causal_saliency_hybrid.py \\
        --max-per-superfamily 50 --window 200 --stride 100 \\
        --device auto

Quick dry run / test:
    ./.venv/bin/python model_result_interp/run_causal_saliency_hybrid.py \\
        --max-per-superfamily 3 --limit 30 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import causal_saliency_hybrid as csh  # noqa: E402

REPO = HERE.parent
DEFAULT_FASTA = REPO / "data" / "vgp" / "all_vgp_tes.fa"
DEFAULT_LABELS = REPO / "data" / "vgp" / "features-tpase"
DEFAULT_TIRS = REPO / "data" / "vgp" / "features"
DEFAULT_CKPT = (REPO / "data_analysis" / "vgp_model_data_tpase_multi" / "v4.3"
                / "hybrid_v4.3_epoch40.pt")
DEFAULT_OUT = HERE / "interpretation_results" / "causal_saliency_hybrid"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--tirs", type=Path, default=DEFAULT_TIRS)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", type=str, default="auto",
                   help="auto | mps | cuda | cpu")
    p.add_argument("--max-per-superfamily", type=int, default=50,
                   help="Cap per (superfamily, tir) cell.")
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--ig-fraction", type=float, default=0.1)
    p.add_argument("--ig-steps", type=int, default=8)
    p.add_argument("--keep-only-window", type=int, default=600)
    p.add_argument("--keep-only-stride", type=int, default=300)
    p.add_argument("--deletion-steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-superfamily", action="append",
                   help="If passed (one or more times), restrict sampling to "
                        "these superfamily names. Default: all.")
    p.add_argument("--head", choices=["sf", "class"], default="sf",
                   help="Which head to attribute against. Defaults to "
                        "superfamily (matches thesis Fig 9).")
    return p.parse_args()


def load_inputs(args, device):
    print(f"loading model from {args.ckpt} ...", flush=True)
    model, class_names, sf_names = csh.load_hybrid_checkpoint(args.ckpt, device)
    sf_to_id = {n: i for i, n in enumerate(sf_names)}
    print(f"  {len(class_names)} classes | {len(sf_names)} superfamilies", flush=True)

    print(f"loading superfamily labels from {args.labels} ...", flush=True)
    labels = csh.load_multiclass_labels(args.labels)
    print(f"  {len(labels)} entries", flush=True)
    print(f"loading TIR flags from {args.tirs} ...", flush=True)
    tirs = csh.load_tir_labels(args.tirs)
    print(f"  {len(tirs)} entries", flush=True)

    print(f"reading fasta {args.fasta} ...", flush=True)
    headers, sequences = csh.read_fasta(args.fasta)
    print(f"  {len(headers)} sequences", flush=True)
    return model, class_names, sf_names, sf_to_id, headers, sequences, labels, tirs


def stratified_sample(headers, labels, tirs, sf_to_id, args, rng):
    keep_sfs = set(args.include_superfamily) if args.include_superfamily else None
    by_cell: dict[tuple[str, int], list[int]] = defaultdict(list)
    skipped_no_label = 0
    skipped_unknown_sf = 0
    skipped_no_tir = 0
    for i, h in enumerate(headers):
        sf = labels.get(h)
        if sf is None:
            skipped_no_label += 1; continue
        if sf not in sf_to_id:
            skipped_unknown_sf += 1; continue
        if keep_sfs is not None and sf not in keep_sfs:
            continue
        tir = tirs.get(h)
        if tir is None:
            skipped_no_tir += 1; continue
        by_cell[(sf, int(tir))].append(i)
    print(f"  skipped: no_label={skipped_no_label}, unknown_sf={skipped_unknown_sf}, "
          f"no_tir={skipped_no_tir}", flush=True)
    print(f"  cells (superfamily, tir): {len(by_cell)}", flush=True)

    selected: list[int] = []
    for cell, idxs in sorted(by_cell.items()):
        idxs = np.asarray(idxs, dtype=np.int64)
        if args.max_per_superfamily > 0 and len(idxs) > args.max_per_superfamily:
            chosen = rng.choice(idxs, size=args.max_per_superfamily, replace=False)
        else:
            chosen = idxs
        selected.extend(int(x) for x in chosen)
        print(f"    {cell}: pool={len(idxs)} chosen={len(chosen)}", flush=True)
    if args.limit > 0 and len(selected) > args.limit:
        rng.shuffle(selected)
        selected = selected[: args.limit]
    print(f"  total selected: {len(selected)}", flush=True)
    return selected


def main() -> int:
    args = parse_args()
    device = csh.resolve_device(None if args.device == "auto" else args.device)
    print(f"device: {device}", flush=True)
    rng = np.random.default_rng(args.seed)

    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    (model, class_names, sf_names, sf_to_id,
     headers, sequences, labels, tirs) = load_inputs(args, device)

    selected = stratified_sample(headers, labels, tirs, sf_to_id, args, rng)
    if args.dry_run:
        print(f"DRY RUN: stopping after sampling. Would process {len(selected)} sequences. "
              f"First 3 headers: {[headers[i] for i in selected[:3]]}", flush=True)
        return 0

    n_ig = max(1, int(args.ig_fraction * len(selected)))
    rng_ig = np.random.default_rng(args.seed + 1)
    ig_set = set(rng_ig.choice(selected, size=min(n_ig, len(selected)), replace=False).tolist())

    featurizer = csh.KmerFeaturizer()
    head = csh.HEAD_SUPERFAMILY if args.head == "sf" else csh.HEAD_CLASS

    index_rows = []
    t_start = time.time()
    n_done = 0
    n_skipped = 0

    for k, i in enumerate(selected):
        header = headers[i]
        seq = sequences[i]
        sf = labels[header]
        tir = int(tirs[header])
        true_id = sf_to_id[sf]  # superfamily id when head=sf

        out_path = raw_dir / f"{i:07d}.npz"
        if out_path.exists():
            n_skipped += 1
            try:
                with np.load(out_path) as z:
                    index_rows.append(dict(
                        idx=int(i), header=header, superfamily=sf, tir=tir,
                        true_id=int(z["true_id"]), pred_id=int(z["pred_id"]),
                        true_prob=float(z["true_prob"]), pred_prob=float(z["pred_prob"]),
                        correct=bool(z["correct"]),
                        rho_N=float(z["rho_N"]), rho_shuffle=float(z["rho_shuffle"]),
                        rho_reverse=float(z["rho_reverse"]),
                        deletion_auc_gap=float(z["deletion_auc_gap"]),
                        seq_len=int(z["seq_len"]),
                        has_ig=int((raw_dir / f"{i:07d}_ig.npy").exists()),
                    ))
            except Exception:
                pass
            continue

        enc = csh.encode_sequence(seq, header)

        # Forward pass for confidence (single sequence batch).
        logits = csh.predict_logits(model, [enc], featurizer, device,
                                    batch_size=1, head=head)[0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        pred_id = int(probs.argmax())
        true_prob = float(probs[true_id])
        pred_prob = float(probs[pred_id])
        correct = pred_id == true_id

        # Saliency w.r.t. TRUE id.
        sal = csh.compute_saliency(model, enc, true_id, featurizer, device, head=head)

        occ_results = {}
        for mode in ("N", "shuffle", "reverse"):
            occ_results[mode] = csh.occlusion_profile(
                model, enc, [true_id], featurizer, device,
                window=args.window, stride=args.stride, mode=mode,
                batch_size=args.batch_size,
                rng_seed=args.seed + 100 + abs(hash(mode)) % 10_000,
                head=head,
            )

        suf = csh.keep_only_window_profile(
            model, enc, [true_id, pred_id], featurizer, device,
            window=args.keep_only_window, stride=args.keep_only_stride,
            batch_size=args.batch_size, head=head,
        )

        dc = csh.deletion_curve(
            model, enc, sal, true_id, featurizer, device,
            n_steps=args.deletion_steps, rng_seed=args.seed + 200, head=head,
        )

        ig = None
        if i in ig_set:
            ig = csh.compute_integrated_gradients(
                model, enc, true_id, featurizer, device,
                steps=args.ig_steps, head=head,
            )

        rho_N = csh.saliency_occlusion_correlation(sal, occ_results["N"], 0)
        rho_shuffle = csh.saliency_occlusion_correlation(sal, occ_results["shuffle"], 0)
        rho_reverse = csh.saliency_occlusion_correlation(sal, occ_results["reverse"], 0)

        np.savez_compressed(
            out_path,
            header=np.array(header),
            superfamily=np.array(sf),
            tir=np.int8(tir),
            true_id=np.int32(true_id),
            pred_id=np.int32(pred_id),
            true_prob=np.float32(true_prob),
            pred_prob=np.float32(pred_prob),
            correct=np.bool_(correct),
            seq_start=np.int32(enc.start),
            seq_end=np.int32(enc.end),
            seq_len=np.int32(enc.length),
            saliency=sal.astype(np.float32),
            occ_centers=occ_results["N"]["centers"].astype(np.int32),
            occ_starts=occ_results["N"]["starts"].astype(np.int32),
            occ_ends=occ_results["N"]["ends"].astype(np.int32),
            occ_drops_N=occ_results["N"]["drops"][0].astype(np.float32),
            occ_drops_shuffle=occ_results["shuffle"]["drops"][0].astype(np.float32),
            occ_drops_reverse=occ_results["reverse"]["drops"][0].astype(np.float32),
            occ_baseline_logit=np.float32(occ_results["N"]["baseline_logits"][0]),
            sufficiency_centers=suf["centers"].astype(np.int32),
            sufficiency_true_logit=suf["survived"][0].astype(np.float32),
            sufficiency_pred_logit=suf["survived"][1].astype(np.float32),
            deletion_fractions=dc["fractions"].astype(np.float32),
            deletion_saliency_curve=dc["saliency_curve"].astype(np.float32),
            deletion_random_curve=dc["random_curve"].astype(np.float32),
            deletion_auc_gap=np.float32(dc["auc_gap"]),
            rho_N=np.float32(rho_N),
            rho_shuffle=np.float32(rho_shuffle),
            rho_reverse=np.float32(rho_reverse),
        )
        if ig is not None:
            np.save(raw_dir / f"{i:07d}_ig.npy", ig.astype(np.float32))

        index_rows.append(dict(
            idx=int(i), header=header, superfamily=sf, tir=tir,
            true_id=true_id, pred_id=pred_id,
            true_prob=true_prob, pred_prob=pred_prob, correct=correct,
            rho_N=float(rho_N), rho_shuffle=float(rho_shuffle), rho_reverse=float(rho_reverse),
            deletion_auc_gap=float(dc["auc_gap"]),
            seq_len=int(enc.length),
            has_ig=int(ig is not None),
        ))
        n_done += 1
        if n_done % 5 == 0 or n_done <= 5:
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 1e-6)
            remaining = (len(selected) - n_skipped - n_done) / max(rate, 1e-6)
            print(f"  [{k+1}/{len(selected)}] done={n_done} skipped={n_skipped} "
                  f"rate={rate:.2f}/s ETA={remaining/60:.1f}min", flush=True)

    index_path = args.out / "index.json"
    with index_path.open("w") as f:
        json.dump({"rows": index_rows,
                   "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   "class_names": list(class_names),
                   "superfamily_names": list(sf_names)}, f, indent=2)
    print(f"wrote {index_path} ({len(index_rows)} rows)", flush=True)

    elapsed = time.time() - t_start
    print(f"DONE. {n_done} new + {n_skipped} skipped in {elapsed/60:.1f}min "
          f"({elapsed/max(n_done,1):.2f}s/seq new).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
