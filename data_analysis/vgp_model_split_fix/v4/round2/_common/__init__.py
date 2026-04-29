"""Shared infrastructure for Round-2 cross-species experiments.

Three competing tracks (T1 V4+invariance, T2 V3-CNN-only+invariance,
T3 V4-CompRes+invariance) share:

  * DANN species-adversarial head + GradReverse layer
  * Group-DRO loss aggregation over clades
  * Phylogenetically-rebalanced WeightedRandomSampler
  * Common data prep that returns clade ids + sf ids + species ids

See `cross_species_strategies.md` and
`/memories/session/plan.md` (Round-2 plan) for context.
"""
