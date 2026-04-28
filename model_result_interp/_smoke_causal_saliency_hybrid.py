"""Smoke test for causal_saliency_hybrid: load v4.3, validate predictions on a
handful of headers in all_test_predictions_v4.3.csv, run the attribution
primitives end-to-end.
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_saliency_hybrid as cs  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CKPT = os.path.join(ROOT, "data_analysis", "vgp_model_data_tpase_multi", "v4.3",
                    "hybrid_v4.3_epoch40.pt")
FASTA = os.path.join(ROOT, "data", "vgp", "all_vgp_tes.fa")
PRED_CSV = os.path.join(ROOT, "data_analysis", "vgp_model_data_tpase_multi", "v4.3",
                        "all_test_predictions_v4.3.csv")
SF_LABELS = os.path.join(ROOT, "data", "vgp", "features-tpase")
TIR_LABELS = os.path.join(ROOT, "data", "vgp", "features")


def main() -> int:
    device = cs.resolve_device()
    print(f"[device] {device}")
    t0 = time.time()
    model, class_names, sf_names = cs.load_hybrid_checkpoint(CKPT, device)
    print(f"[load] model in {time.time() - t0:.1f}s "
          f"| {len(class_names)} classes | {len(sf_names)} superfamilies")
    name_to_id = {n: i for i, n in enumerate(sf_names)}

    # --- pick a handful of test predictions and look up sequences -----------
    print("[csv] reading 12 sample preds from all_test_predictions_v4.3.csv")
    samples: list[dict] = []
    seen_sf: set[int] = set()
    with open(PRED_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sf_true = int(row["true_superfamily_id"])
            if sf_true in seen_sf:
                continue
            samples.append(row)
            seen_sf.add(sf_true)
            if len(samples) >= 12:
                break
    headers_wanted = {r["header"] for r in samples}

    print(f"[fasta] streaming {FASTA} for {len(headers_wanted)} headers")
    seq_by_h: dict[str, str] = {}
    cur_h, cur_buf = None, []
    with open(FASTA) as f:
        for line in f:
            if not line:
                continue
            if line[0] == ">":
                if cur_h is not None and cur_h in headers_wanted:
                    seq_by_h[cur_h] = "".join(cur_buf).upper()
                cur_h = line[1:].strip()
                cur_buf = []
                if len(seq_by_h) == len(headers_wanted):
                    break
            else:
                if cur_h in headers_wanted:
                    cur_buf.append(line.strip())
        else:
            if cur_h in headers_wanted and cur_h not in seq_by_h:
                seq_by_h[cur_h] = "".join(cur_buf).upper()

    print(f"[fasta] retrieved {len(seq_by_h)} / {len(headers_wanted)} headers")

    # --- encode + predict ---------------------------------------------------
    encs = [cs.encode_sequence(seq_by_h[r["header"]], r["header"])
            for r in samples if r["header"] in seq_by_h]
    print(f"[encode] {len(encs)} sequences (lengths "
          f"{[e.length for e in encs[:5]]}...)")
    featurizer = cs.KmerFeaturizer()
    t0 = time.time()
    sf_logits = cs.predict_logits(model, encs, featurizer, device,
                                  batch_size=8, head=cs.HEAD_SUPERFAMILY)
    cl_logits = cs.predict_logits(model, encs, featurizer, device,
                                  batch_size=8, head=cs.HEAD_CLASS)
    print(f"[predict] superfamily logits {sf_logits.shape}, class logits "
          f"{cl_logits.shape} in {time.time() - t0:.2f}s")

    sf_pred = sf_logits.argmax(axis=1)
    cl_pred = cl_logits.argmax(axis=1)
    n_sf_match = 0
    n_cl_match = 0
    rows = [r for r in samples if r["header"] in seq_by_h]
    for i, r in enumerate(rows):
        sf_t = int(r["pred_superfamily_id"])
        cl_t = int(r["pred_class_id"])
        ok_sf = (sf_pred[i] == sf_t)
        ok_cl = (cl_pred[i] == cl_t)
        n_sf_match += int(ok_sf)
        n_cl_match += int(ok_cl)
        if i < 6:
            print(f"  [{i}] {r['header'][:60]:<60} | "
                  f"true_sf={sf_names[int(r['true_superfamily_id'])]:<22} "
                  f"csv_pred_sf={sf_names[sf_t]:<22} ours={sf_names[int(sf_pred[i])]:<22} "
                  f"{'OK' if ok_sf else 'MISMATCH'}")
    print(f"[match] superfamily pred = csv pred on {n_sf_match}/{len(rows)} | "
          f"class pred = csv pred on {n_cl_match}/{len(rows)}")
    if n_sf_match < int(0.8 * len(rows)):
        print("[smoke] FAIL: forward pass disagrees with logged predictions.")
        return 1

    # --- attribution primitives --------------------------------------------
    e = encs[0]
    target_sf = int(sf_pred[0])
    t0 = time.time()
    sal = cs.compute_saliency(model, e, target_sf, featurizer, device,
                              head=cs.HEAD_SUPERFAMILY)
    ig = cs.compute_integrated_gradients(model, e, target_sf, featurizer, device,
                                         steps=8, head=cs.HEAD_SUPERFAMILY)
    print(f"[saliency] |sal| stats abs.mean={np.abs(sal).mean():.4e} "
          f"finite={np.isfinite(sal).all()} | IG abs.mean={np.abs(ig).mean():.4e} "
          f"in {time.time() - t0:.2f}s")

    t0 = time.time()
    occN = cs.occlusion_profile(model, e, [target_sf], featurizer, device,
                                window=200, stride=200, mode="N", batch_size=8,
                                head=cs.HEAD_SUPERFAMILY)
    occS = cs.occlusion_profile(model, e, [target_sf], featurizer, device,
                                window=200, stride=200, mode="shuffle", batch_size=8,
                                head=cs.HEAD_SUPERFAMILY)
    occR = cs.occlusion_profile(model, e, [target_sf], featurizer, device,
                                window=200, stride=200, mode="reverse", batch_size=8,
                                head=cs.HEAD_SUPERFAMILY)
    print(f"[occlusion] N drops range [{occN['drops'].min():.3f}, {occN['drops'].max():.3f}] "
          f"({occN['drops'].shape[1]} positions) in {time.time() - t0:.2f}s")
    print(f"[occlusion] shuffle drops range [{occS['drops'].min():.3f}, {occS['drops'].max():.3f}]")
    print(f"[occlusion] reverse drops range [{occR['drops'].min():.3f}, {occR['drops'].max():.3f}]")

    rho_n = cs.saliency_occlusion_correlation(sal, occN)
    rho_s = cs.saliency_occlusion_correlation(sal, occS)
    rho_r = cs.saliency_occlusion_correlation(sal, occR)
    print(f"[rho] sal vs occlusion: N={rho_n:.3f} shuffle={rho_s:.3f} reverse={rho_r:.3f}")

    t0 = time.time()
    suff = cs.keep_only_window_profile(model, e, [target_sf], featurizer, device,
                                       window=600, stride=600, batch_size=8,
                                       head=cs.HEAD_SUPERFAMILY)
    print(f"[sufficiency] kept-only logit max-min "
          f"[{suff['survived'].min():.3f}, {suff['survived'].max():.3f}] "
          f"({suff['survived'].shape[1]} windows) in {time.time() - t0:.2f}s")

    t0 = time.time()
    delete = cs.deletion_curve(model, e, sal, target_sf, featurizer, device,
                               n_steps=8, head=cs.HEAD_SUPERFAMILY)
    print(f"[deletion] auc gap (random - saliency) = {delete['auc_gap']:.3f} "
          f"in {time.time() - t0:.2f}s")

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
