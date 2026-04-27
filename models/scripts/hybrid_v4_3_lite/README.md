# hybrid_v4_3_lite

Refined version of the v4.3 hybrid CNN+GNN classifier, designed to fight the overfitting
observed in v4.2/v4.3 (train acc → 1.0 while held-out / mini-benchmark drops).

**Strategy** (full plan in `/memories/session/plan.md`):
1. *Capacity* — slim the trunk (~1.9 M → ~0.4–0.5 M params).
2. *Regularisation* — RC flip + canvas-shift + N-noise augmentations, gradient clipping,
   stronger weight decay, embedding mixup on the top-level head.
3. *Data ablations* — train three variants (3-class balanced, 3-class unbalanced,
   binary DNA-vs-nonDNA) on the same trunk to disentangle task complexity from sampling.

Carried from v4.4: 4 sinusoidal positional-encoding channels appended to the CNN input.
**Dropped**: boundary regression head (broken — pool is position-invariant), v4.4
segmentation head, RC early-fusion (kept only late-fusion test-time averaging).

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_model_definition.ipynb` | Slim architecture, parameter-count check, smoke forward pass on dummy batch. **Run this first** to validate the model. |
| 02 | `02_train.ipynb` *(coming next)* | Data pipeline + 5-fold CV training loop. Top-of-notebook `CONFIG` dict selects one of three variants: `three_class_balanced`, `three_class_unbalanced`, `binary_dna`. Outputs go to `results/<variant>/`. |
| 03 | `03_eval_and_compare.ipynb` *(coming next)* | Held-out VGP test + 3-species mini-benchmark for each trained variant; produces `results/summary.md`. |

## Variants (run via `02_train.ipynb`)

| Variant | Top-level head | Per-SF cap | Purpose |
|---------|---------------|------------|---------|
| `three_class_balanced` | DNA / LTR / LINE | 100 ≤ N ≤ 3000 | Direct lite vs v4.3 baseline |
| `three_class_unbalanced` | DNA / LTR / LINE | none | Isolate the manual subsampling effect |
| `binary_dna` | DNA vs non-DNA | 100 ≤ N ≤ 3000 | Apples-to-apples vs original binary v4 overfit gap |

All variants withhold `{mOrnAna, bTaeGut, rAllMis}` from training (same as v4.3).

## Decision rule
Lite variant replaces v4.3 only if (a) test macro-F1 within 0.02 of v4.3,
**and** (b) train→test SF-F1 gap < 0.05, **and** (c) mini-benchmark macro-F1
improves by ≥ 0.03 on at least 2 of 3 held-out species.
