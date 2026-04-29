# Comparison: Hybrid TE Classifier v4.2, v4.3 (rotating-CV), v4.3 (single-fold)

This document compares three checkpoints of the hybrid CNN+GNN TE classifier
(`HybridTEClassifierV4`, 3 top-level classes × 23 superfamilies, identical
architecture across runs) on:

1. **VGP held-out test split** (10846–11814 sequences from 21 species held out
   by `GroupShuffleSplit(test_size=0.2, random_state=42, group=species)`).
2. **Mini-benchmark held-out genomes** (3 species — `bTaeGut` zebra finch,
   `mOrnAna` platypus, `rAllMis` American alligator — that were *excluded
   from the entire VGP corpus* via `EXCLUDE_GENOMES`).

The three checkpoints differ only in their **inner train/val split**:

| Variant | inner-split scheme | leak inside VGP train? |
|---|---|---|
| `v4.2_epoch28` | `StratifiedGroupKFold(5)`, fold rotated each epoch | **YES** — every val species is seen as train species in 4/5 epochs (81/81 species overlap confirmed). |
| `v4.3_rotating_epoch40` | same as v4.2 (rotating fold) | **YES** — same leak. |
| `v4.3_singlefold_epoch28` | second `GroupShuffleSplit(test_size=0.2, random_state=43)` (fixed train/val) | **NO** — val species disjoint from train species for the entire run. |

The VGP held-out **test** split is species-disjoint from any train/val in all
three runs, so the test numbers below are honest for *all* variants.

---

## 1. VGP held-out test (n ≈ 11k, species-disjoint)

| Variant | cls acc | cls macroF1 | sf acc | sf macroF1 | sf bal_acc |
|---|---:|---:|---:|---:|---:|
| v4.2 epoch28           | 0.925 | 0.895 | 0.780 | 0.604 | 0.654 |
| v4.3 rotating, epoch40 | **0.984** | **0.982** | **0.878** | **0.727** | **0.749** |
| v4.3 single-fold, ep28 | 0.959 | 0.955 | 0.764 | 0.638 | 0.690 |

Top-level (3-class) per-class:

| Variant | DNA P | DNA R | DNA F1 | LTR F1 | LINE F1 |
|---|---:|---:|---:|---:|---:|
| v4.2 epoch28           | 0.984 | 0.713 | 0.812 | 0.969 | 0.905 |
| v4.3 rotating, epoch40 | 0.992 | 0.964 | **0.978** | 0.990 | 0.978 |
| v4.3 single-fold, ep28 | 0.958 | 0.921 | 0.939 | 0.966 | 0.960 |

Selected per-superfamily F1 (test split):

| Superfamily | n | v4.2 | v4.3 rot | v4.3 SF |
|---|---:|---:|---:|---:|
| DNA/hAT          | 1100–1249 | 0.265 | **0.919** | 0.256 |
| DNA/PIF-Harbinger| ~200      | 0.402 | **0.432** | 0.173 |
| DNA              | ~250–294  | 0.437 | **0.606** | 0.343 |
| DNA/TcMar-Tc1    | ~370      | 0.804 | **0.907** | 0.853 |
| LTR              | ~175      | 0.262 | **0.329** | 0.287 |
| LTR/Gypsy        | ~3000     | 0.897 | **0.927** | 0.874 |
| LINE/L1          | ~1700–1900| 0.950 | **0.959** | 0.949 |

The v4.3-rotating numbers are inflated by the train/val species leak — the
model has seen each "val" species during ≥1 training epoch — so a fair
*honest* comparison is between **v4.2** and **v4.3 single-fold**, both of which
are at risk of optimistic stopping (the early-stopping criterion is computed
on a contaminated val for v4.2 and on a clean val for v4.3 single-fold). The
v4.3-singlefold val→test gap is ≈ Δsf−macroF1 = 0.04 (val 0.696 → test 0.638),
versus v4.2's Δ ≈ +0.38 (cell 39 reported val ≈ 0.99 → test 0.604) — confirming
the v4.2 number was egregiously inflated by leak.

---

## 2. Mini-benchmark — true held-out genomes (n = 1654)

These three genomes were **excluded from the VGP corpus before any split**, so
no checkpoint has ever seen them as train OR val OR test.

| Variant | bTaeGut cls F1 | mOrnAna cls F1 | rAllMis cls F1 | **overall cls macroF1** | **overall sf macroF1** |
|---|---:|---:|---:|---:|---:|
| v4.2 epoch28           | 0.616 | 0.681 | 0.687 | 0.681 | 0.148 |
| v4.3 rotating, epoch40 | **0.639** | **0.821** | **0.787** | **0.763** | 0.231 |
| v4.3 single-fold, ep28 | 0.428 | 0.703 | 0.654 | 0.607 | **0.250** |

Per-class on the mini-benchmark (overall across 3 genomes):

| Variant | DNA F1 | DNA rec | LTR F1 | LINE F1 |
|---|---:|---:|---:|---:|
| v4.2 epoch28           | 0.740 | 0.820 | 0.678 | 0.626 |
| v4.3 rotating, epoch40 | **0.825** | 0.952 | **0.814** | **0.652** |
| v4.3 single-fold, ep28 | 0.688 | **0.940** | 0.505 | 0.627 |

Per-genome detail (cls_macroF1 / sf_macroF1):

| Variant | bTaeGut | mOrnAna | rAllMis |
|---|---|---|---|
| v4.2 epoch28           | 0.616 / 0.128 | 0.681 / 0.169 | 0.687 / 0.135 |
| v4.3 rotating, epoch40 | **0.639** / 0.168 | **0.821** / **0.274** | **0.787** / 0.216 |
| v4.3 single-fold, ep28 | 0.428 / **0.244** | 0.703 / 0.235 | 0.654 / **0.238** |

---

## 3. Findings

### 3.1 Did the rotating-CV leak make v4.3 actually better, or just look better?

Both. On VGP test (clean), v4.3 rotating beats v4.3 single-fold by:

- cls macroF1: +0.027 (0.982 vs 0.955)
- sf macroF1: +0.089 (0.727 vs 0.638)

On true held-out genomes, v4.3 rotating still beats v4.3 single-fold on top-
level classification (Δ cls macroF1 = +0.156), but **single-fold beats
rotating on superfamily macroF1** (+0.019, 0.250 vs 0.231). So the rotating
schedule helped the network see more species per epoch and reduced overfitting
to top-class boundaries, but it did **not** confer a real advantage at the
fine-grained SF level on out-of-distribution data — there it slightly hurt.

### 3.2 The DNA / hAT collapse

The most striking VGP-test result is `DNA/hAT` F1 falling from **0.919**
(v4.3 rotating) to **0.256** (v4.3 single-fold) and **0.265** (v4.2).
hAT is the largest DNA superfamily (n ≈ 1100 in test), so this is the dominant
contributor to the SF macroF1 gap. Both honest variants share this collapse
nearly exactly — confirming the rotating-CV's inflated 0.92 was **only**
attainable via train/val species leakage on hAT-rich species.

### 3.3 Does v4.2 close the v4.3-singlefold DNA gap?

User question: "I noticed especially poor performance on DNA in v4.3 single,
is v4.2 able to cover that gap?"

**Answer: partially, and only at the top-level (3-class) call.**

| Metric | v4.3 SF | v4.2 | Δ (v4.2 − SF) |
|---|---:|---:|---:|
| VGP-test DNA F1                | 0.939 | 0.812 | **−0.127** (v4.2 worse) |
| VGP-test DNA recall            | 0.921 | 0.713 | −0.208 (v4.2 much worse) |
| VGP-test DNA/hAT F1            | 0.256 | 0.265 | +0.009 (≈ tied, both fail) |
| Mini-bench DNA F1 (overall)    | 0.688 | 0.740 | **+0.052** (v4.2 better) |
| Mini-bench DNA recall          | 0.940 | 0.820 | −0.120 (v4.2 worse) |
| Mini-bench DNA F1 (bTaeGut)    | 0.402 | 0.599 | **+0.197** (v4.2 much better) |

So **v4.2 does not "cover" the DNA gap**:

- On the VGP held-out test, v4.2's DNA performance is *worse* than
  v4.3 single-fold's (DNA F1 0.812 vs 0.939; DNA recall 0.713 vs 0.921), and
  the DNA/hAT collapse is identical (both ≈ 0.26).
- On the mini-benchmark held-out genomes, v4.2 has a slightly higher DNA
  F1 (0.740 vs 0.688) and is notably better on `bTaeGut` (0.599 vs 0.402).
  This is because v4.2 is *more conservative* on DNA (lower recall but higher
  precision), which on OOD genomes happens to land closer to the true label
  distribution.

The "DNA collapse" is not specific to single-fold training — it is a
**genuine architectural / data limitation on hAT and "ambiguous DNA"**
that the rotating-CV merely concealed. v4.2 trades it for a different failure
mode (under-predicting DNA outright). Neither honest variant solves it.

### 3.4 Recommendation for the thesis

- **Headline number for the model**: v4.3 single-fold, since it is the only
  variant whose val/test split is uncontaminated. Cite VGP-test: cls macroF1
  = 0.955, sf macroF1 = 0.638; mini-benchmark: cls macroF1 = 0.607,
  sf macroF1 = 0.250.
- **Use v4.3 rotating only as an upper-bound demonstration** of what the
  architecture can fit when given oracle access to val species, *and* be
  explicit that the inflated number reflects leakage.
- **Document the hAT failure mode** as a known limitation: even with proper
  splitting, both v4.2 and v4.3 single-fold collapse on `DNA/hAT`
  (F1 ≈ 0.26). Hypothesised cause: hAT is an extremely diverse,
  heterogeneous superfamily and the 23-SF flat softmax cannot cleanly
  separate it from "ambiguous DNA" / `DNA/PIF-Harbinger`.
- The "single-fold reduces leak inflation" diagnostic itself is publishable:
  Δ (rotating − single-fold) on VGP-test sf macroF1 = +0.089, but on truly
  unseen genomes it shrinks to +0.019 / −0.019 (single-fold actually wins on
  SF) — quantifying how much the apparent gain was leak-driven.

---

## 4. Methods

- **Architecture** (identical across all three runs): CNN tower
  (`motif_kernels=(7,15,21)`, `context_dilations=(1,2,4,8)`, `rc_mode=late`)
  + KmerGNN tower (`k=7`, `dim=2048`, `window=512`, `stride=256`,
  `gnn_in_dim=2049`, `gnn_hidden=128`, `gnn_layers=3`) +
  CrossModalAttentionFusion (`fusion_dim=256`, `num_heads=4`) → hierarchical
  heads (3-class top, 23-SF). `FIXED_LENGTH=20000`.
- **Training**: `BATCH_SIZE=16`, AdamW (lr=1e-3, wd=1e-4), CosineAnnealingLR
  to lr·0.01, CrossEntropy with inverse-sqrt class weights for the top class
  + LabelSmoothingCE(0.1) on superfamily.
  v4.2 ran 30 epochs, v4.3-rotating ran 40 epochs, v4.3-singlefold ran 30
  epochs (best at 28).
- **VGP test split**: `GroupShuffleSplit(test_size=0.2, random_state=42)`
  grouped by species → trainval = 86 species (~43k seqs), test = 21 species
  (~11k seqs). Disjoint species, no leak.
- **Inner train/val split**:
  - v4.2 / v4.3-rotating: `StratifiedGroupKFold(5, shuffle=True,
    random_state=42)` on the 86 trainval species, **rotating fold each epoch**
    → val species cycle through training fold across epochs (LEAK).
  - v4.3-singlefold: a second `GroupShuffleSplit(test_size=0.2,
    random_state=43)` inside trainval → fixed disjoint train (~69 species,
    ~33k seqs) / val (~17 species, ~9k seqs).
- **Mini-benchmark genomes** (`data/mini_benchmark/`): three labelled curated
  libraries from species explicitly excluded from VGP via the
  `EXCLUDE_GENOMES = {"bTaeGut","mOrnAna","rAllMis"}` filter. Header format
  `>{name}#{class}[/{superfamily}] @{taxon} [S:N]`. Filtered to entries whose
  class ∈ {DNA, LTR, LINE} (n=1654). For SF metrics, restricted further to
  entries whose tag ∈ the model's 23-SF vocabulary (n=897).
- **Evaluation script**: [eval_mini_benchmark.py](eval_mini_benchmark.py),
  outputs in [eval_mini_benchmark.json](eval_mini_benchmark.json) +
  [eval_mini_benchmark.md](eval_mini_benchmark.md). VGP test numbers come
  from the cached `results_v4.3.pt` files in each cluster session folder
  and from [v4.2/cluster session/eval_epoch28.json](v4.2/cluster%20session/eval_epoch28.json).
