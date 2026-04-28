# DANN Smoke Test — README

**Strategy:** Domain-Adversarial Neural Network (DANN) for species-disjoint SF generalisation.  
**Target metric:** Species-disjoint SF macro F1 (baseline: 0.41 at epoch 8).  
**Expected improvement:** +0.05–0.10 SF macro F1 after 20–25 epochs on full data.

---

## Files

| File | Purpose |
|------|---------|
| `dann_smoke_test.py` | Standalone training script (DANN + Hybrid V4 + species-disjoint split) |
| `cross_species_strategies.md` | Full prioritised shortlist with rationale and implementation sketches |

---

## Quick Start — Smoke Test (3 epochs, 5000 samples)

The script defaults to `EPOCHS=3` and `SUBSET_SIZE=5000` so you can verify it runs
in ~10–30 minutes on MPS or ~5 minutes on A100.

```bash
# From repo root:
cd data_analysis/vgp_model_split_fix/v4/

# Smoke test (fast, default parameters)
python dann_smoke_test.py \
  --fasta  ../../../data/vgp/all_vgp_tes.fa \
  --labels ../../../data/vgp/20260120_features_sf \
  --output ./dann_smoke_results \
  --epochs 3 \
  --subset 5000
```

Expected output (smoke test, ~10 min on MPS):
```
Species: 82
Superfamilies (8): ['Academ-1', 'CMC', 'DNA sentinel', 'Maverick', ...]
After filtering: 5000 sequences
Train: 4001 seqs, 65 species
Test:  999 seqs, 17 species (disjoint)
Ep  1: loss=1.4312  bin_f1=0.72  sf_f1=0.18  gate_CNN=0.55  lam=0.000
Ep  2: loss=1.1204  bin_f1=0.80  sf_f1=0.24  gate_CNN=0.61  lam=0.500
Ep  3: loss=0.9871  bin_f1=0.84  sf_f1=0.28  gate_CNN=0.63  lam=0.731
```

The numbers will be noisy at 5k samples and 3 epochs — the key signal is:
- `sf_f1` increasing across epochs (learning is happening)
- `gate_CNN` staying in 0.50–0.75 range (DANN preventing gate collapse)
- No NaN losses

---

## Scaling Up — Full Training Run

Remove `--subset` (or set to 0) and increase epochs for the real experiment:

```bash
# Full run on MPS (~6–8 hours for 25 epochs on full VGP dataset)
python dann_smoke_test.py \
  --fasta  ../../../data/vgp/all_vgp_tes.fa \
  --labels ../../../data/vgp/20260120_features_sf \
  --output ./dann_full_results \
  --epochs 25 \
  --subset 0

# Full run on CSD3 A100 (~2–3 hours for 25 epochs)
python dann_smoke_test.py \
  --device cuda \
  --fasta  /path/to/all_vgp_tes.fa \
  --labels /path/to/20260120_features_sf \
  --output ./dann_full_results \
  --epochs 25 \
  --subset 0
```

---

## SLURM Submission (CSD3)

```bash
#!/bin/bash
#SBATCH --job-name=dann_v4
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G

module load python/3.11 cuda/11.8
source /path/to/venv/bin/activate

python /path/to/dann_smoke_test.py \
  --device cuda \
  --fasta  /rds/user/$USER/all_vgp_tes.fa \
  --labels /rds/user/$USER/20260120_features_sf \
  --output /rds/user/$USER/dann_results \
  --epochs 25 \
  --subset 0
```

---

## Tuning DANN_ALPHA

The species adversarial loss weight `--dann-alpha` (default 0.10) controls the strength of
domain confusion. If results are poor:

| Symptom | Action |
|---------|--------|
| `sf_f1` on test < baseline (0.41) after 10 epochs | Reduce `--dann-alpha` to 0.05 |
| `gate_CNN` still > 0.88 at epoch 10 | Increase `--dann-alpha` to 0.20–0.30 |
| Species adversary loss drops to near-0 quickly | DANN is working; leave as-is |
| Species adversary loss stays high | Species are too similar; reduce `n_layers` in adversary |

---

## Outputs

All results are saved to `--output` (default `./dann_smoke_results/`):

| File | Contents |
|------|---------|
| `dann_epoch{N}_sf{F1}.pt` | Best checkpoint (by held-out SF F1) |
| `dann_metrics.json` | Per-epoch `{train_loss, val_binary_f1, val_sf_f1, gate_cnn, dann_lam}` |
| `dann_species_f1.csv` | Per-species hAT precision/recall and mean gate weight on held-out test |

---

## What to Report

Compare against baseline numbers (epoch 8, no DANN, species-disjoint split):

| Metric | Baseline | DANN (expected) |
|--------|----------|-----------------|
| Binary macro F1 | 0.93 | ~0.93 (should not change) |
| SF macro F1 | 0.41 | **≥ 0.46** |
| hAT F1 | 0.53 | ≥ 0.55 |
| gate CNN/GNN | 0.91 / 0.09 | ~0.65 / 0.35 |

If the gate stabilises in the 0.60–0.70 range (vs. 0.91 bimodal), this confirms the
DANN is suppressing GNN-species memorisation.

---

## Comparing to Other Strategies

See `cross_species_strategies.md` for the full shortlist. If DANN alone gives < +0.03:

1. **Experiment 2 (Two-stage):** Load `hybrid_v4_epoch39.pt`, freeze GNN + fusion,
   finetune only the SF head on the species-disjoint split for 10 epochs.

2. **Experiment 3 (Augmentation):** Add k-mer dropout (15% bins per window) to
   `HybridDataset.__getitem__` and retrain with DANN.

The combination of DANN + k-mer dropout is expected to be the strongest signal.
