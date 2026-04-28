"""CLI runner for causal saliency experiments.

For a stratified sample of sequences, computes per-sequence:
  - vanilla input-gradient saliency (true-class)
  - occlusion profile in modes {N, shuffle, reverse} (true-class)
  - keep-only-window sufficiency profile (true-class)
  - deletion curve (true-class)
  - integrated gradients (true-class) on a smaller subsample

Writes results into one .npz per sequence under
  model_result_interp/interpretation_results/causal_saliency/raw/
plus a single index.parquet (or csv if pandas missing) summarising metadata.

Designed to be killable / resumable: skips sequences whose .npz already exists.

Usage:
  python model_result_interp/run_causal_saliency.py \
      --max-per-superfamily 100 --window 100 --stride 50 --device auto

For a quick dry run:
  python model_result_interp/run_causal_saliency.py --max-per-superfamily 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import causal_saliency as cs

REPO = HERE.parent
DEFAULT_FASTA = REPO / "data" / "vgp" / "all_vgp_tes.fa"
DEFAULT_LABELS = REPO / "data" / "vgp" / "features-tpase"
DEFAULT_TIRS = REPO / "data" / "vgp" / "features"
DEFAULT_CKPT = REPO / "data_analysis" / "vgp_model_data_tpase_multi" / "improved_rc_cnn_best.pt"
DEFAULT_OUT = HERE / "interpretation_results" / "causal_saliency"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--tirs", type=Path, default=DEFAULT_TIRS)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", type=str, default="auto",
                   help="auto | mps | cuda | cpu")
    p.add_argument("--max-per-superfamily", type=int, default=100,
                   help="Cap per (superfamily, tir) cell. Negative = no cap.")
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="Drop sequences whose true-class softmax prob < this. "
                        "0 disables the filter; saliency is computed even for "
                        "low-confidence cases but they are flagged in the index.")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--ig-fraction", type=float, default=0.2,
                   help="Fraction of selected sequences to also run IG on.")
    p.add_argument("--ig-steps", type=int, default=16)
    p.add_argument("--keep-only-window", type=int, default=300)
    p.add_argument("--keep-only-stride", type=int, default=150)
    p.add_argument("--deletion-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=-1,
                   help="Hard cap on total selected sequences (after stratified sampling). "
                        "Negative = no cap.")
    p.add_argument("--dry-run", action="store_true",
                   help="Stop after stratified sampling + first 3 sequences.")
    p.add_argument("--include-non-dna", action="store_true",
                   help="Also include LTR/* and LINE/* sequences (not just DNA/*). "
                        "Useful for B1 cross-class panels; default is DNA-only since "
                        "the TIR labels are most informative there.")
    return p.parse_args()


def load_inputs(args, device):
    print(f"loading model from {args.ckpt} ...", flush=True)
    model, class_names, tag_to_id = cs.load_checkpoint(args.ckpt, device)
    print(f"  classes ({len(class_names)}): {class_names}", flush=True)

    print(f"loading labels from {args.labels} ...", flush=True)
    labels = cs.load_multiclass_labels(args.labels)
    print(f"  {len(labels)} entries", flush=True)
    print(f"loading TIR flags from {args.tirs} ...", flush=True)
    tirs = cs.load_tir_labels(args.tirs)
    print(f"  {len(tirs)} entries", flush=True)

    print(f"reading fasta {args.fasta} ...", flush=True)
    headers, sequences = cs.read_fasta(args.fasta)
    print(f"  {len(headers)} sequences", flush=True)
    return model, class_names, tag_to_id, headers, sequences, labels, tirs


def stratified_sample(headers, sequences, labels, tirs, tag_to_id, args, rng):
    """Group by (superfamily, tir) and cap each cell."""
    by_cell: dict[tuple[str, int], list[int]] = defaultdict(list)
    skipped_no_label = 0
    skipped_unknown_class = 0
    skipped_no_tir = 0
    for i, h in enumerate(headers):
        sf = labels.get(h)
        if sf is None:
            skipped_no_label += 1
            continue
        if sf not in tag_to_id:
            skipped_unknown_class += 1
            continue
        tir = tirs.get(h)
        if tir is None:
            skipped_no_tir += 1
            continue
        if not args.include_non_dna and not sf.startswith("DNA"):
            continue
        by_cell[(sf, int(tir))].append(i)

    print(f"  skipped: no_label={skipped_no_label}, unknown_class={skipped_unknown_class}, no_tir={skipped_no_tir}", flush=True)
    print(f"  cells (superfamily, tir): {len(by_cell)}", flush=True)

    selected: list[int] = []
    for cell, idxs in sorted(by_cell.items()):
        idxs = np.asarray(idxs, dtype=np.int64)
        if args.max_per_superfamily > 0 and len(idxs) > args.max_per_superfamily:
            chosen = rng.choice(idxs, size=args.max_per_superfamily, replace=False)
        else:
            chosen = idxs
        selected.extend(int(x) for x in chosen)
        print(f"    {cell}: pool={len(idxs)}, chosen={len(chosen)}", flush=True)

    if args.limit > 0 and len(selected) > args.limit:
        rng.shuffle(selected)
        selected = selected[: args.limit]

    print(f"  total selected: {len(selected)}", flush=True)
    return selected


def main() -> int:
    args = parse_args()
    device = cs.resolve_device(None if args.device == "auto" else args.device)
    print(f"device: {device}", flush=True)
    rng = np.random.default_rng(args.seed)

    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    model, class_names, tag_to_id, headers, sequences, labels, tirs = load_inputs(args, device)

    selected = stratified_sample(headers, sequences, labels, tirs, tag_to_id, args, rng)
    if args.dry_run:
        print(f"DRY RUN: stopping after sampling. Would process {len(selected)} sequences. "
              f"First 3: {[headers[i] for i in selected[:3]]}", flush=True)
        return 0

    # Decide which sequences also get IG (deterministic from seed).
    n_ig = max(1, int(args.ig_fraction * len(selected)))
    rng_ig = np.random.default_rng(args.seed + 1)
    ig_set = set(rng_ig.choice(selected, size=min(n_ig, len(selected)), replace=False).tolist())

    index_rows = []
    t_start = time.time()
    n_done = 0
    n_skipped = 0
    n_lowconf = 0

    for k, i in enumerate(selected):
        header = headers[i]
        seq = sequences[i]
        sf = labels[header]
        tir = int(tirs[header])
        true_class = tag_to_id[sf]

        out_path = raw_dir / f"{i:07d}.npz"
        if out_path.exists():
            n_skipped += 1
            # still record in index so we don't lose metadata across re-runs
            try:
                with np.load(out_path) as z:
                    pred_class = int(z["pred_class"])
                    true_prob = float(z["true_prob"])
                    pred_prob = float(z["pred_prob"])
                    correct = bool(z["correct"])
                    rho_N = float(z["rho_N"])
                    rho_shuffle = float(z["rho_shuffle"])
                    rho_reverse = float(z["rho_reverse"])
                    auc_gap = float(z["deletion_auc_gap"])
                index_rows.append(dict(
                    idx=i, header=header, superfamily=sf, tir=tir,
                    true_class=true_class, pred_class=pred_class,
                    true_prob=true_prob, pred_prob=pred_prob, correct=correct,
                    rho_N=rho_N, rho_shuffle=rho_shuffle, rho_reverse=rho_reverse,
                    deletion_auc_gap=auc_gap,
                    seq_len=min(len(seq), cs.FIXED_LENGTH),
                    has_ig=int((raw_dir / f"{i:07d}_ig.npy").exists()),
                ))
            except Exception:
                pass
            continue

        enc = cs.encode_sequence(seq, header)

        # Forward pass for true/predicted-class confidence.
        logits = cs.predict_logits(model, [enc], device)[0]
        # softmax in float64 for numerical safety
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        pred_class = int(probs.argmax())
        true_prob = float(probs[true_class])
        pred_prob = float(probs[pred_class])
        correct = pred_class == true_class
        if true_prob < args.min_confidence:
            n_lowconf += 1

        # Saliency w.r.t. TRUE class (this is the canonical attribution for
        # comparing against curator features, regardless of whether the model
        # got it right).
        sal = cs.compute_saliency(model, enc, true_class, device)

        # Occlusion in 3 modes, true-class.
        occ_results = {}
        for mode in ("N", "shuffle", "reverse"):
            occ = cs.occlusion_profile(
                model, enc, [true_class], device,
                window=args.window, stride=args.stride, mode=mode,
                batch_size=args.batch_size,
                rng_seed=args.seed + 100 + hash(mode) % 10_000,
            )
            occ_results[mode] = occ

        # Sufficiency (keep-only window)
        suf = cs.keep_only_window_profile(
            model, enc, [true_class, pred_class], device,
            window=args.keep_only_window, stride=args.keep_only_stride,
            batch_size=args.batch_size,
        )

        # Deletion curve (saliency vs random)
        dc = cs.deletion_curve(
            model, enc, sal, true_class, device,
            n_steps=args.deletion_steps, rng_seed=args.seed + 200,
        )

        # Optional IG
        ig = None
        if i in ig_set:
            ig = cs.compute_integrated_gradients(
                model, enc, true_class, device, steps=args.ig_steps,
            )

        # Spearman rho per mode
        rho_N = cs.saliency_occlusion_correlation(sal, occ_results["N"], 0)
        rho_shuffle = cs.saliency_occlusion_correlation(sal, occ_results["shuffle"], 0)
        rho_reverse = cs.saliency_occlusion_correlation(sal, occ_results["reverse"], 0)

        np.savez_compressed(
            out_path,
            header=np.array(header),
            superfamily=np.array(sf),
            tir=np.int8(tir),
            true_class=np.int32(true_class),
            pred_class=np.int32(pred_class),
            true_prob=np.float32(true_prob),
            pred_prob=np.float32(pred_prob),
            correct=np.bool_(correct),
            seq_start=np.int32(enc.start),
            seq_end=np.int32(enc.end),
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
            idx=i, header=header, superfamily=sf, tir=tir,
            true_class=true_class, pred_class=pred_class,
            true_prob=true_prob, pred_prob=pred_prob, correct=correct,
            rho_N=float(rho_N), rho_shuffle=float(rho_shuffle), rho_reverse=float(rho_reverse),
            deletion_auc_gap=float(dc["auc_gap"]),
            seq_len=enc.length,
            has_ig=int(ig is not None),
        ))
        n_done += 1
        if n_done % 10 == 0 or n_done <= 5:
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 1e-6)
            remaining = (len(selected) - n_skipped - n_done) / max(rate, 1e-6)
            print(f"  [{k+1}/{len(selected)}] done={n_done} skipped={n_skipped} lowconf={n_lowconf} "
                  f"rate={rate:.2f}/s ETA={remaining/60:.1f}min", flush=True)

    # Write index.
    index_path = args.out / "index.json"
    with index_path.open("w") as f:
        json.dump({"rows": index_rows,
                   "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   "class_names": list(class_names)}, f, indent=2)
    print(f"wrote {index_path} ({len(index_rows)} rows)", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDONE. processed {n_done} new + {n_skipped} skipped (resumed) sequences "
          f"in {elapsed/60:.1f}min ({elapsed/max(n_done,1):.2f}s/seq new).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
