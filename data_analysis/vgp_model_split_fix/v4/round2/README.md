# Round-2 cross-species smokes

Three competing tracks against the seed-42 species-disjoint test set.
See `../cross_species_strategies.md` and `../RESULTS.md` for the
diagnostic that motivates these.

## Tracks
| Track | Architecture                              | Tests                                            |
|-------|-------------------------------------------|--------------------------------------------------|
| T1    | Unchanged V4 + DANN + Group-DRO + sampler | Can the existing V4 learn species-invariant SF?  |
| T2    | V4 CNN tower only (no GNN, no fusion)     | Is the GNN a *net liability* under species shift?|
| T3    | V4 with GNN replaced by CompResEncoder    | Can a centroid-debiased compositional encoder beat the GNN? |

All three share the same invariance stack (DANN species-adversarial
head + Group-DRO over clades + phylo-rebalanced sampler over
`(clade, sf)`).

## Smoke params (in `run_smoke.py`)
EPOCHS=3, SUBSET_SIZE=5000, BATCH_SIZE=16, LR=3e-4, SF_DROPOUT=0.3,
LABEL_SMOOTHING_SF=0.1, DANN_MAX_LAMBDA=0.5, GDRO_ETA=0.05.

Each `--ab` run trains both arms (stack ON, then OFF) so the comparison
is paired on the same data subset. ~1h per arm on Apple Silicon MPS.

## Launch
```bash
cd data_analysis/vgp_model_split_fix/v4/round2
bash _run_smoke.sh T1
bash _run_smoke.sh T2
bash _run_smoke.sh T3
```

Outputs land in `results/<T>/smoke_<T>_<arm>.{log,json}` plus a
`smoke_<T>_delta.json` summarising A/B.

## Acceptance criteria (per track)
- Δ(SF macro F1, stack − nostack) ≥ +0.03 on val.
- aPelLes-style hAT recall (clade `a` in `per_clade_hat`) ≥ 0.40 on test.

The track that maxes both wins; that one gets the full-scale CSD3 run
(`models/slurm_submit_hybrid_v5.sh` adapted for whichever wrapper
won).

## Scale-up to full data
Edit the `# SMOKE PARAMS` block at the top of `run_smoke.py`:
`EPOCHS=10, SUBSET_SIZE=None, BATCH_SIZE=64`. The diagnostic showed
late epochs (>10) actively *forget* held-out clades, so do NOT
increase `EPOCHS` past 10–12 for the full run.
