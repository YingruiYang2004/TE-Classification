# Cross-Species Generalisation Strategies for VGP Hybrid V4 SF Head

**Target problem:** Species-disjoint SF macro F1 = 0.41 (epoch 8) / 0.36 (epoch 30) vs. 0.86 on random split.  
**Goal:** Identify ≤ 5 concrete experiments with realistic chance of +0.05 SF macro F1 within budget.

---

## 0. Diagnostic Plan: Per-Species hAT Breakdown

Before designing any fix, run the per-species hAT diagnostic to determine whether the collapse
is concentrated in 1–2 outlier species (sampling artefact) or spread uniformly (architectural problem).

### What to add to `eval_epoch8.py` (~30 lines)

```python
# After collecting (headers, bin_true, sf_true, sf_pred, gate_cnn, gate_gnn):

from collections import defaultdict
species_stats = defaultdict(lambda: {'hat_tp':0,'hat_fp':0,'hat_fn':0,'n_total':0,'gate_cnn':[]})

HAT_ID = superfamily_to_id.get('hAT', None)   # look up from checkpoint

for h, bt, st, sp, gc in zip(headers, bin_true, sf_true, sf_pred, gate_cnn):
    if bt == 0:  # non-DNA sample
        continue
    genome = h.split('#')[0].rsplit('-', 1)[-1]  # e.g. "aAnoBae"
    species_stats[genome]['n_total'] += 1
    species_stats[genome]['gate_cnn'].append(gc)
    if st == HAT_ID:
        if sp == HAT_ID:
            species_stats[genome]['hat_tp'] += 1
        else:
            species_stats[genome]['hat_fn'] += 1
    elif sp == HAT_ID:
        species_stats[genome]['hat_fp'] += 1

print(f"\n{'Species':<15} {'N_DNA':>6} {'hAT_P':>6} {'hAT_R':>6} {'gate_CNN':>9}")
for sp, d in sorted(species_stats.items(), key=lambda x: -x[1]['n_total']):
    tp, fp, fn = d['hat_tp'], d['hat_fp'], d['hat_fn']
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    mean_gate = sum(d['gate_cnn']) / len(d['gate_cnn'])
    print(f"{sp:<15} {d['n_total']:>6} {prec:>6.2f} {rec:>6.2f} {mean_gate:>9.3f}")
```

**Interpreting results:**
- If hAT recall < 0.10 for only 1–2 species with large sample counts → sampling/label noise, not architecture.  
  Fix: phylogenetic re-balancing of training data (cheap, ~1 hour).  
- If hAT recall < 0.30 uniformly across ≥5 species → architecture memorises species-specific GNN features.  
  Fix: regularisation or domain adversarial training (harder).

**Gate check:** If species with low hAT recall also have gate_CNN > 0.95, the model has learned to
*not* trust the GNN for these species at all — confirming GNN carries species-specific information
that does not transfer.

---

## 1. Prioritised Shortlist

### Experiment 1 (Rank #1 · EV: HIGH): Domain-Adversarial Species Classifier (DANN)

**Rationale.**
The gate collapse from 0.40/0.60 (random split) to 0.91/0.09 (species-disjoint test) reveals that
the GNN tower has memorised species-specific 7-mer composition: on seen species it contributes
heavily; on unseen species the gate adapts to ignore it. The SF head's weights, trained with a
GNN-heavy fused embedding, cannot generalise from a CNN-only embedding at test time. DANN attacks
this directly: a species-classifier head reads from the *fused* embedding through a
Gradient-Reversal Layer (GRL). Minimising species-classification accuracy forces the fused
representation to discard species-discriminative features, making the gate stable and the SF head
generalise. Because the binary head already generalises (F1 = 0.93), the GRL pressure only needs to
remove SF-irrelevant species variance, not all taxonomy signal.

**Implementation sketch.**

Add to the existing training script (e.g. `hybrid_v4_3_train.py`) or the GPU notebook cell by cell:

```python
# --- 1. Gradient Reversal Layer (add near loss definitions) ---
import torch, torch.nn as nn, torch.nn.functional as F

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.clone()
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None

class GradRevLayer(nn.Module):
    def __init__(self, lam=1.0): super().__init__(); self.lam = lam
    def forward(self, x): return _GRL.apply(x, self.lam)

# --- 2. Species adversary head (add to model or as standalone module) ---
class SpeciesAdversary(nn.Module):
    def __init__(self, fusion_dim: int, n_species: int, dropout: float = 0.3):
        super().__init__()
        self.grl = GradRevLayer(lam=1.0)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, n_species),
        )
    def set_lam(self, lam: float): self.grl.lam = lam
    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.head(self.grl(fused))   # gradients reversed through GRL

# --- 3. Build species vocab once before training ---
# genome_ids = [extract_genome_id(h) for h in train_headers]
# species_names = sorted(set(genome_ids))
# species_to_id = {s: i for i, s in enumerate(species_names)}
# n_species = len(species_names)

# --- 4. Instantiate adversary ---
# adversary = SpeciesAdversary(FUSION_DIM, n_species).to(device)
# adv_opt = torch.optim.AdamW(adversary.parameters(), lr=lr, weight_decay=1e-4)
# adv_loss_fn = nn.CrossEntropyLoss()
# DANN_ALPHA = 0.1   # relative weight; start small, schedule up

# --- 5. In training loop, after forward pass (model returns fused embedding) ---
# Extend HybridTEClassifierV4.forward() to also return `fused`:
#   return binary_logits, sf_logits, gate_weights, fused

# Then in the loop:
# p = (epoch - 1) / epochs                       # 0 -> 1
# lam = 2. / (1 + math.exp(-10 * p)) - 1         # DANN schedule
# adversary.set_lam(lam)
#
# species_ids = torch.tensor([species_to_id[extract_genome_id(h)]
#                             for h in batch_headers], device=device)
# adv_logits = adversary(fused.detach())   # <-- detach: adversary trains separately
# adv_loss   = adv_loss_fn(adv_logits, species_ids)
# adv_opt.zero_grad(); adv_loss.backward(); adv_opt.step()
#
# # Main model loss includes adversarial term via GRL (no .detach() here):
# adv_logits_grl = adversary(fused)            # GRL reverses gradients into encoder
# adv_loss_enc   = adv_loss_fn(adv_logits_grl, species_ids)
#
# loss = binary_weight*bin_loss + sf_weight*sf_loss + DANN_ALPHA*adv_loss_enc
```

**A complete standalone smoke-test is provided in `dann_smoke_test.py`.**

**Evaluation plan.**
- Primary: species-disjoint SF macro F1 at epochs 5, 10, 20.
- Secondary: gate weights mean on held-out species (expect stabilisation toward 0.60–0.70 CNN).
- Success: +0.05 SF macro F1 vs. epoch 8 baseline (0.41 → 0.46+).
- Wall-clock: ~3–4 hours on CSD3 A100 for 20 epochs on full dataset; ~45 min on MPS for smoke test.

**Failure modes.**
- *Adversary trivially wins* (species F1 stays high): increase DANN_ALPHA or lam_max. Check
  whether `n_species` is large enough that the task is hard.
- *SF F1 drops below binary baseline*: DANN_ALPHA too large; reduce by 10×.
- *Diagnostic*: plot gate weight distribution on held-out test at epoch 8 vs. DANN-epoch 8.
  If gate is still bimodal, the GRL signal is not reaching the fusion layer (check lam schedule).

---

### Experiment 2 (Rank #2 · EV: MEDIUM-HIGH): Two-Stage Training — Freeze GNN, Finetune SF Head

**Rationale.**
The random-split checkpoint (epoch 39) has good SF representations (macro F1 = 0.86) but a
species-contaminated SF head. Freezing the GNN tower (the main memoriser) and the fusion module,
then finetuning *only* the SF head on a species-disjoint CV fold, forces the head to learn
SF-discriminative features from the CNN embedding alone — which already transfers (binary F1 = 0.93).
This is a zero-risk warm-start: if it fails, we learn that the CNN embedding is not SF-discriminative
enough without GNN support.

**Implementation sketch.**

```python
# Load random-split checkpoint
ckpt = torch.load("hybrid_v4_epoch39.pt", weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])

# Freeze GNN tower and fusion; unfreeze SF head only
for name, p in model.named_parameters():
    if 'gnn_tower' in name or 'fusion' in name:
        p.requires_grad_(False)

# Optionally also freeze CNN tower for a stricter test:
# for name, p in model.named_parameters():
#     if 'cnn_tower' in name: p.requires_grad_(False)

# Use species-disjoint split for finetuning
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_train, idx_test = next(gss.split(all_h, groups=species_ids))

# Finetune: only SF head parameters are updated
opt_ft = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4, weight_decay=1e-3   # lower LR to avoid catastrophic forgetting
)
# SF head loss with heavier label smoothing (0.15) to prevent re-memorisation
sf_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.15, weight=sf_weights_t)

# Training loop identical to existing loop, but only the SF head sees gradients
```

**Evaluation plan.**
- Train for 10–15 epochs on species-disjoint CV fold 1, then evaluate on held-out test set.
- Compare: (a) frozen GNN + finetune SF head, (b) frozen GNN + frozen CNN + finetune SF head.
- Success: SF macro F1 on held-out ≥ 0.46 after 10 epochs.
- Wall-clock: ~1 hour on CSD3 (frozen backbone = cheap forward pass), ~20 min on MPS.

**Failure modes.**
- SF F1 ≤ 0.41 after 15 epochs: the CNN-only fused embedding is not SF-discriminative. Next step:
  unfreeze the CNN tower too and retrain with DANN (Experiment 1).
- SF F1 improves on validation but not on held-out test: head is re-memorising from the
  species-disjoint training fold. Add k-mer dropout (Experiment 3).

---

### Experiment 3 (Rank #3 · EV: MEDIUM): Strong k-mer Dropout + Sequence Mutation Augmentation

**Rationale.**
Species-specific memorisation by the GNN tower is enabled by highly reproducible k-mer frequency
profiles for each species (e.g., GC content, CpG suppression, repeat biases). Stochastic k-mer
feature dropout at training time forces the GNN to learn from the *structure* of the k-mer graph
(co-occurrence patterns = motifs) rather than from the absolute frequencies of individual species-
characteristic k-mers. Combined with a low-rate random mutation on the CNN input (simulating
polymorphism across taxa), both towers are nudged toward motif-based rather than composition-based
representations.

**Implementation sketch.**

```python
# --- In HybridDataset.__getitem__, after kmer_feat is computed ---
# K-mer dropout: zero-out a random fraction of k-mer bins per window
KMER_DROPOUT_P = 0.15  # fraction of bins to drop per window at training time

if self.augment.enabled:
    # 1. K-mer dropout on GNN features
    drop_mask = np.random.rand(*kmer_feat[:, :-1].shape) < KMER_DROPOUT_P
    kmer_feat = kmer_feat.copy()
    kmer_feat[:, :-1][drop_mask] = 0.0
    # Re-normalize each window after dropout
    norms = np.linalg.norm(kmer_feat[:, :-1], axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    kmer_feat[:, :-1] /= norms

    # 2. Low-rate random mutations on sequence for CNN (simulates ~1% substitution)
    MUTATION_RATE = 0.01
    mut_positions = np.random.rand(seq_len) < MUTATION_RATE
    if mut_positions.any():
        seq_idx = seq_idx.copy()
        seq_idx[mut_positions] = np.random.randint(0, 4, size=mut_positions.sum())
```

**Evaluation plan.**
- Train from scratch with species-disjoint split for 15 epochs, compare against Experiment 1.
- Track SF F1 at each epoch; expect the gap between val SF F1 and held-out SF F1 to narrow.
- Success: held-out SF macro F1 ≥ 0.46 and val–test F1 gap reduced by ≥ 50% vs. baseline.
- Wall-clock: ~30 min extra per epoch on MPS (augmentation is in-CPU, cheap); ~5 hours total.

**Failure modes.**
- Val SF F1 drops below 0.36: KMER_DROPOUT_P or MUTATION_RATE too high; halve both.
- No improvement in test generalisation: augmentation alone insufficient without representation-level
  constraint (pair with Experiment 1 or 2).

---

### Experiment 4 (Rank #4 · EV: MEDIUM): Stronger SF Head Regularisation Only

**Rationale.**
The cheapest experiment: keep the architecture identical but dramatically increase regularisation
pressure on the SF head specifically (heavier dropout 0.4→0.5, higher weight decay 1e-4→1e-3,
label smoothing 0.1→0.2, early stopping on held-out species F1 not validation F1). The epoch-8
vs. epoch-30 gap (0.41 vs. 0.36) suggests the SF head continues to fit species-specific patterns
after the representations are well-formed. Stopping earlier and regularising harder may arrest
this without any structural change. This is a floor-raising experiment, not a ceiling-raising one —
it's unlikely to reach +0.05 alone but is worth doing as a cheap first check.

**Implementation sketch.**

```python
# Change superfamily_head in HybridTEClassifierV4 constructor:
self.superfamily_head = nn.Sequential(
    nn.Linear(fusion_dim, 256), nn.GELU(),
    nn.Dropout(0.45),           # was 0.20
    nn.Linear(256, 128), nn.GELU(),
    nn.Dropout(0.45),           # extra layer
    nn.Linear(128, num_superfamilies)
)

# Change SF loss label smoothing: 0.1 -> 0.20
superfamily_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.20, weight=sf_weights_t)

# Change optimizer weight decay for SF head parameters only:
optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if 'superfamily_head' not in n],
     'weight_decay': 1e-4},
    {'params': model.superfamily_head.parameters(),
     'weight_decay': 5e-3},    # 50× heavier on SF head
], lr=lr)

# Change early stopping criterion to per-species-disjoint SF F1, not combined score
```

**Evaluation plan.**
- Train 20 epochs species-disjoint; track val SF F1 and held-out SF F1 per epoch.
- Success: held-out SF macro F1 at epoch where val SF F1 peaks ≥ 0.44 (vs. 0.41 baseline).
- Wall-clock: identical to baseline run, ~3 hours on CSD3.

**Failure modes.**
- Val SF F1 drops below 0.30: over-regularised; reduce dropout to 0.35.
- No improvement at held-out: regularisation alone insufficient; proceed to Experiment 1.

---

### Experiment 5 (Rank #5 · EV: MEDIUM-LOW): K-mer Ablation — CNN-Only Baseline

**Rationale.**
Before spending GPU budget on DANN, it is worth confirming that the GNN tower is the source of
species memorisation. If the CNN-only model achieves SF macro F1 ≥ 0.41 on the species-disjoint
test (matching or beating the full hybrid), then the GNN tower is *actively hurting* and the
cheapest fix is simply to drop it. This is a diagnostic experiment, not a primary fix. The gate
weight evidence (collapse to 0.91 CNN on held-out) already suggests the CNN carries the signal,
but a controlled ablation provides the causal evidence needed for the thesis.

**Implementation sketch.**

```python
# Simplest: run the existing v3 CNN-only model on the species-disjoint split
# (it already exists at data_analysis/vgp_model_clustering/).
# Or: zero-out GNN contribution in HybridTEClassifierV4 at test time.

class CNNOnlyWrapper(nn.Module):
    """Drop-in eval wrapper that forces gate to CNN=1.0, GNN=0.0."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x_cnn, mask, x_gnn, edge_index, batch_vec):
        cnn_embed = self.model.cnn_tower(x_cnn, mask)
        # Project CNN embed directly, bypass GNN
        fused = self.model.fusion.out_proj(
            self.model.fusion.ln1(self.model.fusion.cnn_proj(cnn_embed))
        )
        binary_logits = self.model.binary_head(fused)
        sf_logits = self.model.superfamily_head(fused)
        gate_weights = torch.tensor([[1.0, 0.0]]).expand(x_cnn.size(0), -1)
        return binary_logits, sf_logits, gate_weights

# Load epoch-8 checkpoint, wrap, evaluate on held-out test.
```

**Evaluation plan.**
- Evaluate epoch-8 checkpoint in CNN-only mode on the held-out test set.
- If CNN-only SF macro F1 ≥ 0.41: GNN is hurting; subsequent experiments should down-weight
  or regularise the GNN tower more aggressively.
- If CNN-only SF macro F1 < 0.35: CNN lacks SF-discriminative features alone; the representation
  *needs* the GNN but the GNN needs to be species-agnostic (→ DANN is essential).
- Wall-clock: ~5 minutes (inference only, no training).

---

## 2. Priority Ranking and EV Summary

| Rank | Experiment | EV | Effort | Risk |
|------|-----------|-----|--------|------|
| 1 | DANN (gradient reversal on species) | HIGH | Medium (30 lines, 1 new head) | GRL schedule tuning |
| 2 | Two-stage (freeze GNN, finetune SF head) | MEDIUM-HIGH | Low (10 lines) | Backbone may be species-contaminated |
| 3 | k-mer dropout + mutation augmentation | MEDIUM | Low (15 lines in Dataset) | May need DANN to complement |
| 4 | Stronger SF head regularisation | MEDIUM | Very Low (5 lines) | Unlikely to reach +0.05 alone |
| 5 | CNN-only ablation (diagnostic) | — | Trivial (eval only) | Diagnostic, not a fix |

**Recommended sequence:**
1. Run Experiment 5 first (5 min, diagnostic).
2. Run Experiment 4 while Experiment 1 is being set up (cheap sanity check).
3. Run Experiment 1 (DANN) — this is the primary bet.
4. If DANN gives +0.05, layer in Experiment 3 (augmentation) to see if it compounds.
5. If DANN alone gives < +0.03, run Experiment 2 (two-stage) as a backup.

---

## 3. Explicitly Deprioritised

| Idea | Reason |
|------|--------|
| Full re-train with phylogenetically stratified CV (5-fold × 80 species) | ~40+ hours GPU; out of budget |
| Nucleotide Transformer / DNABERT fine-tuning | Model size, GPU memory, weeks to pretrain; out of scope |
| Per-species embedding centering at test time | Requires held-out data from each test species (unavailable) |
| Prototypical network SF head | Interesting but requires from-scratch metric-learning setup; 2-3 days engineering |
| Hyperparameter sweep without a clear hypothesis | Too noisy; insufficient held-out budget |
| Label relabelling (fix RepeatModeler noise) | Explicitly out of scope per problem constraints |
| Collect more data | Explicitly out of scope per problem constraints |
