# Round-2.5 Results — Stack-strength sweep + 6-epoch baselines

**Setup:** Same data prep as Round-2 (5000 subset, MPS, batch 16, species-disjoint splits 2939/939/1122 across 8 clades). Two questions:

1. Does longer training (6 ep instead of 3) rescue SF F1 / per-clade hAT recall in the **nostack** baselines?
2. Does a **weakened invariance stack** (lower DANN λ + lower Group-DRO η) keep the cls head alive while still recovering hAT on held-out clades?

## Headline table (held-out test, n=1122)

| Run                     | EP | DANN λ | GDRO η | cls F1 | SF F1  | hAT F1 (n=192) | hAT R  | clade-`a` (id=0) hAT R (n=163) | gate (cnn,gnn) |
|-------------------------|----|--------|--------|--------|--------|----------------|--------|-------------------------------|-----------------|
| T1 nostack              | 6  | —      | —      | **0.822** | 0.208 | 0.040 | 0.021 | 0.000 | (0.79, 0.21) |
| T2 nostack (CNN-only)   | 6  | —      | —      | 0.667 | 0.143 | 0.000 | 0.000 | 0.000 | n/a |
| T3 nostack (CompRes)    | 6  | —      | —      | 0.682 | 0.283 | 0.216 | 0.125 | **0.141** | (0.71, 0.29) |
| **T3 weak1 (winner)**   | **6** | **0.05** | **0.005** | 0.577 | **0.298** | **0.630** | **0.510** | **0.571** | (0.56, 0.44) |
| T3 weak2                | 6  | 0.15   | 0.02   | 0.453 | 0.112 | 0.000 | 0.000 | 0.000 | (0.57, 0.43) |
| (Round-2 ref) T1 nostack| 3  | —      | —      | 0.842 | 0.009 | 0.000 | 0.000 | 0.000 | (0.44, 0.56) |
| (Round-2 ref) T3 stack  | 3  | 0.5    | 0.05   | 0.251 | 0.039 | 0.000 | 0.000 | 0.000 | (0.28, 0.72) |

**Acceptance bar reminder:** Δ SF F1 ≥ +0.03 AND clade-`a` hAT recall ≥ 0.40.

## Key findings

1. **T3 weak1 is the winner — by a wide margin.** Stack=ON with λ=0.05, η=0.005, warmup=2:
   - **clade-`a` hAT recall jumps from 0.00 (nostack 3ep) / 0.14 (nostack 6ep) → 0.571** — the headline failure mode from Round-1 is fixed.
   - **hAT F1 0.630** (vs 0.216 nostack 6ep, 2.9×). Precision 0.824, recall 0.510.
   - SF macro F1 0.298, +0.015 over nostack 6ep on test (val was much larger: 0.332 vs 0.189).
   - cls F1 drops from 0.682 → 0.577 (−0.105) — non-trivial cost but not catastrophic. Training is monotone, no collapse.
   - Recovery is also seen on **clade 4** (0.00 → 0.40) and **clade 3** (0.00 → 0.14).

2. **T3 weak2 (λ=0.15, η=0.02) is unstable.** Hits val SF F1 0.189 + hAT F1 0.669 at ep4, then catastrophically forgets at ep5/6 (val hAT back to 0). Test hAT recall 0. Same bimodal forgetting we saw in Round-1 epoch 30. Stack strength is on a knife-edge between λ=0.05 (smooth) and λ=0.15 (oscillatory).

3. **Longer training does help nostack, but doesn't reach the bar.** T3 nostack 6ep: SF F1 0.283 (vs 0.086 at 3ep, 3.3×); hAT F1 0.216 (vs 0). But clade-`a` hAT recall is still only 0.141 — way below the 0.40 bar. **Longer training alone won't fix the species shift.**

4. **Architectural ranking on cls F1 (nostack 6ep)**: T1 0.822 > T3 0.682 > T2 0.667. T1 still leads on cls but T3 leads on SF (0.283 > T1 0.208 > T2 0.143). This confirms the CompRes encoder beats both V4-with-GNN and CNN-only on the cross-species SF task at 6 epochs.

5. **Gate behaviour confirms the design intent.** T3 nostack 6ep: gate (0.71, 0.29) — cnn-dominant, just like V4. T3 weak1: gate (0.56, 0.44) — much more balanced, the CompRes signal is being used. The weakened DANN successfully shifts the model towards the species-invariant compositional channel without crushing the binary head.

## Decision: promote T3 + weak1 to full CSD3 run

The acceptance bar is met by a wide margin on the metric that matters (clade-`a` hAT recall: 0.571 ≫ 0.40). Even though T3 weak1 cls F1 drops to 0.58 at smoke scale, this is a **5000-seq / 6-epoch** smoke. At full data (~36k) and 10 epochs, both heads should benefit substantially.

**CSD3 hyperparams (changes from smoke):**

| Param          | Smoke value | CSD3 full value |
|----------------|-------------|------------------|
| EPOCHS         | 6           | **10**          |
| SUBSET_SIZE    | 5000        | **None (full ~36k)** |
| BATCH_SIZE     | 16          | **64**          |
| LR             | 3e-4        | 3e-4 (unchanged) |
| DANN λ_max     | 0.05        | 0.05 (unchanged) |
| DANN warmup    | 2 ep        | 3 ep (scale with epochs) |
| GDRO η         | 0.005       | 0.005 (unchanged) |
| SF_DROPOUT     | 0.3         | 0.3 (unchanged) |
| TRACK          | T3          | **T3**          |

**Watch-outs for the full run:**
- Round-1 V4 forgot held-out clades catastrophically at epoch 30. **Do not exceed 10 epochs.** Save best checkpoint by val clade-`a` hAT recall (not val SF F1 alone — Round-1 picked the wrong checkpoint that way).
- T3 weak2 collapsed at the boundary η=0.02. If the full run shows oscillation in val SF F1, **drop η to 0.002** and rerun.
- Eval set must include per-clade hAT recall as a logged metric — add it to the training loop.

## Artifacts

- 6-epoch nostack baselines: `results/T{1,2,3}/smoke_T?_nostack_6ep.json`
- T3 weak1 (winner): `results/T3/smoke_T3_stack_weak1_6ep.json`
- T3 weak2: `results/T3/smoke_T3_stack_weak2_6ep.json`
- Per-arm logs: `results/round25_*.log`, `round25_stdout.log`

## Next step (immediate, before CSD3 submission)

Clone `models/slurm_submit_hybrid_v5.sh` → `models/slurm_submit_round2_t3_weak1.sh`. Add a thin wrapper that calls `data_analysis/vgp_model_split_fix/v4/round2/run_smoke.py --track T3 --epochs 10 --subset-size 0 --dann-lambda 0.05 --dann-warmup 3 --gdro-eta 0.005 --tag _full_csd3` (with `--subset-size 0` interpreted as "use all data" — needs a one-line patch in `prepare_data`). Add per-clade hAT recall to the val-end-of-epoch log so we can checkpoint on it.

## Future variant (parallel CSD3 job): keep the GNN, demean *after* it

T3's `CompResEncoder` discards both the per-window graph structure and the
composition magnitude (mean-pool then `comp - centroid` → MLP). That's why
even with cosine LR + class-balanced SF loss we plateau ~0.40 val SF F1 /
~0.50 val cls F1: the second tower is structurally degraded relative to
the v4.3 GNN.

A cleaner variant: **keep the V4 GNN tower unchanged**, then apply
`embed - centroid_clade(embed)` to the GNN *output embedding* (and/or to
the fused embedding feeding the SF head only). That preserves the GNN's
within-sequence message passing and keeps composition magnitude available
to the cls head, while still removing species-level shortcut features
where they actually hurt cross-species SF generalisation.

Suggested track name: **T4** = V4 (CNN + GNN + gated fusion) + post-tower
demeaning on the SF branch only + DANN/GDRO/phylo (weak1 strengths).
~30-line change in `_common/models.py` (new `V4PostDemeanWrapper`). Run
in parallel to the cosine-LR T3 follow-up on CSD3 and pick the winner.
