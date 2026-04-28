"""Smoke test for causal_saliency.py.

Loads the v3 ImprovedRCCNN checkpoint, picks 5 sequences, and runs:
- predict_logits
- compute_saliency
- compute_integrated_gradients (small step count)
- occlusion_profile in all 3 modes
- keep_only_window_profile
- deletion_curve
- saliency_occlusion_correlation

Prints shapes, finiteness, and a couple of summary numbers. Exit code 0 on success.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import causal_saliency as cs

REPO = HERE.parent
FASTA = REPO / "data" / "vgp" / "all_vgp_tes.fa"
LABELS = REPO / "data" / "vgp" / "features-tpase"
TIR_FILE = REPO / "data" / "vgp" / "features"
CKPT = REPO / "data_analysis" / "vgp_model_data_tpase_multi" / "improved_rc_cnn_best.pt"


def main() -> int:
    assert FASTA.exists() and LABELS.exists() and CKPT.exists(), "missing input files"

    device = cs.resolve_device()
    print(f"device: {device}")

    t0 = time.time()
    model, class_names, tag_to_id = cs.load_checkpoint(CKPT, device)
    print(f"loaded model in {time.time()-t0:.2f}s, {len(class_names)} classes")
    print(f"first 5 classes: {class_names[:5]}")

    # Read a small slice of the FASTA so we don't pay the full IO cost.
    headers, sequences = [], []
    h, buf = None, []
    with open(FASTA) as f:
        for line in f:
            if line.startswith(">"):
                if h is not None:
                    headers.append(h)
                    sequences.append("".join(buf).upper())
                    buf = []
                h = line[1:].strip()
                if len(headers) >= 5:
                    break
            else:
                buf.append(line.strip())
    print(f"read {len(headers)} sequences; lengths={[len(s) for s in sequences]}")

    labels = cs.load_multiclass_labels(LABELS)
    tirs = cs.load_tir_labels(TIR_FILE)
    print(f"label table: {len(labels)} entries; TIR table: {len(tirs)} entries")
    for h in headers:
        print(f"  {h[:50]}...  label={labels.get(h)}  tir={tirs.get(h)}")

    encs = [cs.encode_sequence(s, h) for h, s in zip(headers, sequences)]
    for i, e in enumerate(encs):
        assert e.length == min(len(sequences[i]), cs.FIXED_LENGTH)
        assert e.base_idx.shape == (cs.FIXED_LENGTH,)

    # --- predict
    t0 = time.time()
    logits = cs.predict_logits(model, encs, device)
    print(f"predict_logits: shape={logits.shape}, finite={np.all(np.isfinite(logits))}, time={time.time()-t0:.2f}s")
    pred = logits.argmax(axis=1)
    print(f"predictions: {[class_names[p] for p in pred]}")

    # --- saliency on first sequence
    enc0 = encs[0]
    target_class = int(pred[0])
    t0 = time.time()
    sal = cs.compute_saliency(model, enc0, target_class, device)
    print(f"saliency: shape={sal.shape}, finite={np.all(np.isfinite(sal))}, |max|={np.max(np.abs(sal)):.4f}, time={time.time()-t0:.2f}s")
    in_window_max = np.max(np.abs(sal[enc0.start:enc0.end]))
    out_window_max = np.max(np.abs(np.concatenate([sal[:enc0.start], sal[enc0.end:]])))
    print(f"  |sal| in-window max={in_window_max:.4f}, out-of-window max={out_window_max:.4f}")

    # --- IG (small steps to keep smoke fast)
    t0 = time.time()
    ig = cs.compute_integrated_gradients(model, enc0, target_class, device, steps=8)
    print(f"IG (8 steps): shape={ig.shape}, finite={np.all(np.isfinite(ig))}, time={time.time()-t0:.2f}s")

    # --- occlusion profile, three modes
    for mode in ("N", "shuffle", "reverse"):
        t0 = time.time()
        occ = cs.occlusion_profile(
            model, enc0, [target_class], device,
            window=100, stride=200, mode=mode, batch_size=32,
        )
        drops = occ["drops"][0]
        n_pos = drops.size
        print(f"occlusion[{mode}]: P={n_pos}, mean drop={drops.mean():.4f}, max drop={drops.max():.4f}, "
              f"min drop={drops.min():.4f}, finite={np.all(np.isfinite(drops))}, time={time.time()-t0:.2f}s")

        # ρ vs saliency for this profile
        rho = cs.saliency_occlusion_correlation(sal, occ, target_class_index=0)
        print(f"  spearman(sal, drop[{mode}]) = {rho:.3f}")

    # --- keep-only sufficiency
    t0 = time.time()
    suf = cs.keep_only_window_profile(model, enc0, [target_class], device, window=200, stride=400, batch_size=32)
    surv = suf["survived"][0]
    print(f"sufficiency: P={surv.size}, peak_survived_logit={surv.max():.3f}, baseline={logits[0, target_class]:.3f}, time={time.time()-t0:.2f}s")

    # --- deletion curve
    t0 = time.time()
    dc = cs.deletion_curve(model, enc0, sal, target_class, device, n_steps=10)
    print(f"deletion: auc_sal={dc['auc_saliency']:.3f}, auc_rnd={dc['auc_random']:.3f}, gap={dc['auc_gap']:.3f}, time={time.time()-t0:.2f}s")
    print(f"  saliency_curve: {np.array2string(dc['saliency_curve'], precision=2)}")
    print(f"  random_curve:   {np.array2string(dc['random_curve'], precision=2)}")

    # --- determinism: re-run shuffle occlusion with same seed; expect identical drops
    occ_a = cs.occlusion_profile(model, enc0, [target_class], device, window=100, stride=400, mode="shuffle", rng_seed=7, batch_size=32)
    occ_b = cs.occlusion_profile(model, enc0, [target_class], device, window=100, stride=400, mode="shuffle", rng_seed=7, batch_size=32)
    print(f"determinism (shuffle same seed): max diff = {np.max(np.abs(occ_a['drops'] - occ_b['drops'])):.2e}")

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
