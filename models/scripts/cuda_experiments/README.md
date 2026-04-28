# CUDA Experiments — Imbalance-Handling Ablation

Three self-contained notebooks targeting a rented CUDA GPU. Each notebook is
a complete experiment (no shared imports beyond stdlib + torch + sklearn) and
ends with an analysis block tied to its hypothesis.

All three keep the **proven v4.3 hybrid architecture** (~8M params: CNN tower
with motif kernels [7,15,21], 4 dilated context blocks, RC late-fusion + GNN
tower with k=7 / dim=2048 / window=512 + Cross-Modal Attention Fusion). The
only things that change between notebooks are: (a) the data subsampling policy,
(b) the loss function, and (c) the analysis block.

| # | File | Question | Loss | Sampling |
|---|------|----------|------|----------|
| 1 | `binary_dna_natural_focal.ipynb` | Can focal loss + weighted sampler recover DNA recall under the natural ~8/92 prior? | `FocalLoss(γ=2, α=inv_freq)` | `WeightedRandomSampler` |
| 2 | `binary_dna_natural_threshold_tuned.ipynb` | Or do we just need post-hoc threshold tuning? | `BCEWithLogitsLoss(pos_weight)` | natural shuffle |
| 3 | `three_class_natural_weighted.ipynb` | Does the SF subsampling cap matter for the 3-way DNA/LTR/LINE problem? | weighted CE (`inv_sqrt`) | no `max_per_sf` cap |

All three:
- exclude benchmark genomes `{mOrnAna, bTaeGut, rAllMis}`,
- use `FIXED_LENGTH=20_000` (max real sequence is 19,907 bp — verified lossless),
- run 5-fold rotating CV with a held-out 20 % test split,
- save top-5 checkpoints by combined validation score,
- log fusion gate weights (CNN vs GNN) per epoch.

Generate / regenerate with `python _generate_notebooks.py`.
