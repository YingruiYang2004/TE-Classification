# Smoke test: cross-species augmentation (strategy 1)

This directory contains the runnable smoke test for **strategy 1** in
[../cross_species_strategies.md](../cross_species_strategies.md):
training-time sequence augmentation (RC + crop + low-rate point
mutation + GNN window dropout) plus SF-head label smoothing and raised
SF-head dropout.

It exists to give a fast yes/no signal on whether the augmentation
recipe meaningfully shifts the v4 hybrid's behaviour on the
species-disjoint test set, before committing to a full A100 run.

## Files

- `run_smoke_aug.py` — single-file training script. Reuses the v4
  hybrid notebook's definition cells via `exec` (same pattern as
  `../cluster session/eval_epoch8.py`).
- `smoke_<aug|noaug>.log` — produced after a run.
- `smoke_<aug|noaug>.pt` — final-epoch state + history.
- `smoke_summary.json` — short JSON with test metrics + A/B delta.

## Launch (local, MPS / CPU)

```bash
cd "data_analysis/vgp_model_split_fix/v4/smoke_aug"

# Aug arm only (default):
python run_smoke_aug.py

# Control arm only:
python run_smoke_aug.py --no-aug

# Both back-to-back (recommended for an A/B comparison):
python run_smoke_aug.py --both
```

Each arm of the smoke takes ~30–60 minutes on Apple Silicon MPS with
the default `EPOCHS=3, SUBSET_SIZE=5000, BATCH_SIZE=16`.

## What to look for

Open `smoke_aug.log` and `smoke_noaug.log` side by side, or read the
`smoke_summary.json`. Three quantities matter:

1. **Held-out (species-disjoint) test SF macro F1** — main outcome.
   Augmentation should **not be worse than the control** at 3 epochs
   on a 5000-sample subset; ideally already a small positive delta.
2. **hAT F1 on the held-out test** — the worst class on the full v4
   epoch-30 run; if augmentation helps, it should help here first.
3. **Mean gate weights `(w_cnn, w_gnn)` on the held-out test** —
   baseline v4 epoch 30 collapsed to ~(0.91, 0.09). With augmentation,
   the gate should sit closer to (0.5, 0.5) because the GNN tower is
   no longer rewarded for memorising exact 7-mers.

A "smoke pass" is **delta SF macro F1 ≥ +0.02** at 3 epochs / 5000
samples. The full-scale signal target (≥+0.05) is not expected from
the smoke alone; the smoke only checks the recipe is plausible.

## Scaling up

To turn this into a real experiment on CSD3:

1. Edit the `# SMOKE TEST PARAMS` block at the top of
   `run_smoke_aug.py`:

   ```python
   EPOCHS         = 25
   SUBSET_SIZE    = None    # full data
   BATCH_SIZE     = 64      # A100
   ```

2. Submit using the existing slurm template
   `models/slurm_submit_hybrid_v5.sh`. Replace the inner training
   command with:

   ```bash
   python data_analysis/vgp_model_split_fix/v4/smoke_aug/run_smoke_aug.py --both
   ```

3. The full run trains ~36k samples × 25 epochs × 2 arms; on an A100
   this is ~10–14 h total. The on-the-fly augmentation re-featurization
   (CPU `KmerWindowFeaturizer`) is the main extra cost vs. the cached
   v4 baseline; it is parallelisable via `num_workers > 0` if needed
   (the script currently uses `num_workers=0` to keep MPS happy —
   bump it on CSD3).

4. After training, run the per-species diagnostic on the saved
   checkpoints:

   ```bash
   python data_analysis/vgp_model_split_fix/v4/eval_per_species.py \
       --ckpt data_analysis/vgp_model_split_fix/v4/smoke_aug/smoke_aug.pt \
       --ckpt data_analysis/vgp_model_split_fix/v4/smoke_aug/smoke_noaug.pt
   ```

   Note that `eval_per_species.py` currently expects the `arch` /
   `model_state_dict` checkpoint format used by the v4 training driver,
   not the smoke-test checkpoint format. If you scale up, switch the
   smoke save to that format (mirror the keys saved in the GPU notebook
   training driver) so the diagnostic accepts the file.

## Implementation notes

- **Augmentation is training-time only.** Validation and test sets use
  the cached, unmodified k-mer features so metrics remain comparable
  across runs.
- **Window dropout is implemented at the dataset/collate boundary**:
  we drop windows from each sample's k-mer feature tensor before the
  base `collate_hybrid` builds the chain edge index. This avoids
  reimplementing edge-index construction. Run-level p_run=0.5 means
  half of all batches see no window dropout — keeps the signal
  trainable.
- **Label smoothing applies to the SF head only**, not the binary
  head. The binary head is working (macro F1 ~0.93 on
  species-disjoint test) and we explicitly do not want to perturb it.
- **Random-crop and random placement in the canvas** are independent
  random sources. The crop happens *before* the canvas placement, so
  the placement only operates on the augmented (possibly RC, possibly
  cropped) sequence.

## Failure modes / troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Aug arm cls macro F1 drops by >0.02 vs control | mutation rate too high | drop `P_MUT` from 0.005 to 0.002 |
| Aug arm SF macro F1 not different from control | augmentation too weak at smoke scale | increase `SUBSET_SIZE` to 10000 or `EPOCHS` to 5 |
| Featurization step blows up RAM | featurizer caching too many windows | confirm `add_pos=True, l2_normalize=True` in `KmerWindowFeaturizerGPU` and that `MAX_PER_SF` cap is applied |
| MPS step crashes mid-epoch | known MPS edge case in BatchNorm with very small batches | drop `BATCH_SIZE` to 8 or run on CPU for the smoke |
