# Round-2 Smoke Results

**Setup:** 5000-seq subset, 3 epochs, MPS, species-disjoint splits (train 2939 / val 939 / test 1122; 268 train species, 90 test species, 8 clades {a,b,e,f,k,m,r,s}). Per-track A/B = invariance stack (DANN + Group-DRO + phylo sampler) ON vs OFF.

**Acceptance bar:** Δ SF macro F1 ≥ +0.03 on val AND clade-`a` (aPelLes-style) hAT recall ≥ 0.40 on test.

## Headline table (held-out test, species-disjoint, n=1122)

| Track | Arm     | cls F1 | SF F1  | gate (cnn, gnn) | clade-`a` hAT R | Δ SF F1 (stack − nostack) |
|-------|---------|--------|--------|------------------|-----------------|---------------------------|
| T1 (V4 + stack)        | nostack | **0.842** | 0.0091 | (0.44, 0.56) | 0.000 | — |
| T1                     | stack   | 0.251     | 0.0385 | (0.05, 0.95) | 0.000 | **+0.0294** |
| T2 (CNN-only + stack)  | nostack | 0.580     | **0.0713** | n/a | 0.000 | — |
| T2                     | stack   | 0.251     | 0.0090 | n/a | 0.000 | −0.0623 |
| T3 (CompRes + stack)   | nostack | 0.696     | **0.0863** | (0.87, 0.13) | 0.000 | — |
| T3                     | stack   | 0.251     | 0.0388 | (0.28, 0.72) | 0.000 | −0.0476 |

Per-clade hAT recall is **0.000 across every (clade, arm)** pair — the SF head never learned hAT in 3 epochs at any config.

## Reading the data

1. **The invariance stack collapses the binary head.** Every stack=ON arm (T1, T2, T3) hits cls F1 = 0.2505 — exactly the majority-class baseline. DANN at λ_max=0.5 + Group-DRO η=0.05 + phylo sampler is too aggressive simultaneously; the binary signal can't survive.
2. **No track meets the acceptance bar.** T1 stack is the only one with a positive Δ SF F1 (+0.029 — a hair below the +0.03 bar) but it pays a catastrophic −0.59 cls F1 cost, so the gain is meaningless.
3. **3 epochs is too short for the SF head full stop.** Best SF F1 across all 6 arms is 0.086 (T3 nostack) — no arm got close to even Round-1's 0.41 SF macro F1. Per-clade hAT recall is 0 everywhere because hAT just never lit up.
4. **Architectural ranking on cls F1 (nostack baseline)**: T1 (V4) 0.842 > T3 (CompRes) 0.696 > T2 (CNN-only) 0.580. The V4 hybrid retains its cls-head advantage; CompRes is a viable GNN replacement; CNN-only loses noticeable ground.
5. **Gate behaviour:** T3 nostack already shifts cnn-dominant (0.87) at 3 epochs — supports the Round-1 finding that the GNN tower contributes little. T1 stack's gate flip to gnn-dominant (0.95) co-occurs with the cls-head collapse — the model is using the GNN compositional signal as a shortcut once the species-discriminative features are stripped.

## Conclusion

**Round-2 smokes are NEGATIVE — do not promote any track to a full CSD3 run as-is.**

Two coupled problems:
- 3 epochs at 5000 seqs is below the floor where SF macro F1 stabilises (Round-1 needed 8 epochs at 36k to hit 0.41).
- The invariance stack at the prescribed strength collapses the cls head. We cannot tell whether the stack helps SF generalisation because the underlying classifier breaks first.

## Recommendation: Round-2.5 (cheap, before any CSD3 run)

Two follow-up smokes (still local, MPS, ≤2h total) before deciding what to ship to CSD3:

1. **Stack-strength sweep** on T3 only (best nostack SF F1):
   - λ_max ∈ {0.05, 0.15} (down from 0.5), DANN warmup 2 ep
   - GroupDRO η ∈ {0.005, 0.02} (down from 0.05)
   - Keep phylo sampler always on
   - Goal: find a setting where stack=ON cls F1 stays within −0.05 of nostack
2. **Longer-baseline smoke** at 5000×6ep, nostack only, all three tracks. Confirm whether the SF F1 ranking (T3 > T2 > T1) holds with more training, and whether per-clade hAT recall ever leaves zero in 6 epochs.

If (1) finds a non-collapsing stack config and (2) confirms T3 leads on SF F1, ship T3 + reduced-strength stack to CSD3 at 10 epochs / full data / batch 64.

If neither smoke shows life, escalate to Round-3: drop the invariance stack entirely, focus on data-side fixes (clade-balanced sampling + stronger SF reweighting only), and consider a pretrained DNA backbone (Nucleotide Transformer / DNABERT-2) as the cnn_tower replacement.

## Artifacts

- `results/T1/smoke_T1_{stack,nostack,delta}.json`, `results/T1/run_all.log`
- `results/T2/smoke_T2_{stack,nostack,delta}.json`, `results/T2/run_all.log`
- `results/T3/smoke_T3_{stack,nostack,delta}.json`, `results/T3/run_all.log`
- `all_smokes_stdout.log` (top-level driver log)

Each per-arm JSON contains full `cls_pred / cls_true / sf_pred / sf_true` arrays plus `per_sf` per-class metrics and `per_clade_hat` for downstream re-analysis.
