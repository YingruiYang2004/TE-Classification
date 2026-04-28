# Cross-species generalisation strategies for the v4 hybrid TE classifier

**Audience:** future-me, supervisor, and any agent implementing follow-ups.
**Scope:** narrow the species-disjoint test gap on the **superfamily (SF)
head only**. Binary head untouched. Random-split headline numbers
already reported. Time budget: a few days, ≤8h GPU per experiment on
CSD3.

---

## 1. Diagnosis recap

On the species-disjoint split (`GroupShuffleSplit`, seed 42), the
v4 hybrid checkpoint behaves as follows on the held-out test set
(n=7586):

| metric             | epoch 8 (early) | epoch 30 (best-by-val) |
|--------------------|-----------------|------------------------|
| Binary macro F1    | 0.93            | 0.90                   |
| **SF macro F1**    | **0.41**        | **0.36**               |
| hAT F1             | 0.53            | 0.32                   |
| TcMar-Tc1 F1       | 0.88            | 0.88                   |
| Gate (CNN/GNN)     | bimodal         | 0.91 / 0.09            |

On the random split, the *same architecture* gets SF macro F1 = 0.86
and gate 0.40 / 0.60. Three signals matter:

1. The **binary head transfers**; only the **SF head collapses**.
2. **Late epochs hurt held-out species** (epoch 8 generalises better
   than epoch 30 despite worse rolling val score).
3. The fusion **gate collapses to CNN-dominant** late in training
   (from 0.40/0.60 in random-split to 0.91/0.09 here).

These three together point to **the SF head latching onto species-specific
k-mer signatures via the GNN tower**, then over time reweighting the gate
toward the CNN tower while the SF head's softmax classifier overfits the
training species manifold.

The fixes below all attack the species-specific memorisation pathway —
not generic overfitting (which would be addressed by ordinary L2 or
dropout sweeps that we are explicitly not doing).

---

## 2. Mandatory pre-fix diagnostic — **EXECUTED**

Output of [`eval_per_species.py`](eval_per_species.py) (`per_species_hybrid_v4_epoch{8,30}.csv` + console report in `eval_per_species.log`) on the seed-42 species-disjoint test set:

| SF (epoch 8)        | macro recall | worst-2-species share of misses | worst species (n_true, recall) |
|---------------------|-------------:|--------------------------------:|---------------------------------|
| DNA/hAT             | 0.38         | **89%** (664 / 745)             | aAnoBae (218, **0.055**); aPelLes (627, **0.270**) |
| DNA/PiggyBac        | 0.58         | **100%** (55/55)                | kTriCli (63, 0.397); aPelLes (33, 0.485) |
| DNA/PIF-Harbinger   | 0.54         | 80% (61/76)                     | aAnoBae (12, 0.167); aPelLes (27, 0.185) |
| DNA/CMC             | 0.50         | 78% (7/9)                       | kTriCli (5, 0.20)                |
| DNA/TcMar-Tc1       | 0.82         | 67% (35/52)                     | rThaEle (5, 0.20); kTriCli (47, 0.34) |
| DNA (unspec.)       | 0.71         | 39% (24/62)                     | mostly uniform                   |

Epoch 30 (best-by-val) is **worse on every clade-driven SF**: hAT recall on aAnoBae drops 0.055 → 0.009, aPelLes 0.270 → 0.057. Late training memorises the train-species hAT manifold and forgets aPelLes/aAnoBae-style hAT.

**Conclusion: outcome is _concentrated_, not uniform.** ≥80% of misses on five of seven non-trivial SFs come from ≤2 species — almost always **amphibians/reptiles** (`aAnoBae`, `aPelLes`, `aHylSar`, `aRanTem`, `kTriCli`). The seed-42 split sent four amphibians and a turtle into the test set with no closely-related amphibian/reptile in train; the model has never seen their TE compositional signature.

**Reranking (effective post-diagnostic):**
1. **Augmentation (strategy 1, ★★★★★)** — still top: it forces invariance to compositional shifts, which is exactly the failure mode (aPelLes-style hAT differs from train-species hAT mainly in low-level sequence statistics).
2. **Phylo-rebalanced sampling (strategy 5, promoted to ★★★★)** — was ★★, now ★★★★. Up-weighting under-represented clades during training partially mimics seeing aPelLes-style sequences. Cheap; safe to combine with strategy 1.
3. Two-stage SF-head finetune from epoch-8 with stronger regularisation (strategy 2, ★★★) — still useful as a **complement** because epoch 30 is provably overfit; it does not fix the unseen-clade problem on its own.
4. DANN species-adversarial (strategy 3, ★★★) — still attractive but heavier. Defer until strategy 1+5 plateau.
5. Prototypical SF head (strategy 4, ★★) — demoted: clade memorisation is the bottleneck; replacing the classifier head won't help if the **features** are clade-specific. Try only after representations are de-biased by 1+5+3.

The strategy 1 smoke below is the right first move regardless. The natural follow-up is a 1+5 stack.

If
concentrated, swap target to strategy 5.

---

## 3. Ranked shortlist (5 strategies)

EV ranking is *impact × tractability*. "Wall-clock" assumes CSD3 A100
unless stated; "≈baseline" means the same train cost as the existing
v4 run.

### Strategy 1 — Strong sequence augmentation (training-time only)  ★★★★★

**Rationale.** The GNN tower is a k=7, 2048-dim hash that treats
sequence as a bag of exact 7-mers per window. Across vertebrate species,
the *exact* 7-mer composition of (e.g.) hAT TIRs is not conserved
beyond 70-90% identity, but the *consensus* motif is. Therefore
training k-mer-hash features on raw sequences encourages memorisation
of **species-specific 7-mer instances** rather than the conserved
motif. Augmenting training sequences with random reverse complement,
random crop within `FIXED_LENGTH`, low-rate point mutations
(0.5–1%), and random window dropout in the GNN graph **destroys exact
7-mer matches without destroying the underlying motif**, forcing both
towers to rely on signals that survive perturbation. The CNN tower is
already RC-invariant so RC augmentation costs it nothing; mutations
slightly redistribute its input but it has BatchNorm and dropout to
absorb that. The cost lives entirely on the data side: re-running
`KmerWindowFeaturizer` on the augmented sequence each epoch. At full
data this is ~5 minutes of CPU; on a 5000-sample smoke it is seconds.

**Implementation sketch.** Modify `HybridDataset.__getitem__` (cell 13
of `vgp_hybrid_v4_gpu.ipynb`) to apply augmentation when
`self.training` is true, then re-featurize. Plus a `WindowDropout`
applied inside `collate_hybrid` (cell 15) that masks out a fraction of
windows in the GNN edge_index.

```python
# in HybridDataset
def __getitem__(self, idx):
    seq = self.sequences[idx]
    if self.training and self.rng is not None:
        seq = augment_sequence(seq, self.rng,
                               p_rc=0.5, p_mut=0.005,
                               crop_to=FIXED_LENGTH)
        x_kmer, _ = self.featurizer.featurize_sequence(seq)
    else:
        x_kmer = self.kmer_pre[idx]   # cached float16
    return (self.headers[idx], seq, self.toplevel[idx],
            self.sf[idx], x_kmer)

def augment_sequence(seq, rng, p_rc, p_mut, crop_to):
    if rng.random() < p_rc:
        seq = reverse_complement(seq)
    if len(seq) > crop_to:
        start = rng.integers(0, len(seq) - crop_to + 1)
        seq = seq[start:start + crop_to]
    if p_mut > 0:
        b = bytearray(seq, "ascii")
        n_mut = rng.binomial(len(b), p_mut)
        for pos in rng.choice(len(b), n_mut, replace=False):
            b[pos] = ord(rng.choice(b"ACGT".replace(bytes([b[pos]]), b"")))
        seq = b.decode()
    return seq

# in collate_hybrid, after edge_index built:
if training and rng.random() < p_window_drop_run:
    keep = rng.random(n_windows) > p_window_drop
    keep[:1] = True  # keep at least one
    x_kmer = x_kmer[keep]
    edge_index = restrict_edges(edge_index, keep)
```

**Eval plan.** Re-train v4 with augmentation enabled, identical
schedule. Track val SF macro F1 + per-species hAT recall + mean gate
weights. **Success bar: ≥+0.05 SF macro F1 vs. baseline epoch-8
checkpoint on the same held-out test set** (this beats the "early stop
at epoch 8" workaround). Wall-clock: ≈baseline (5–8h on A100).

**Failure mode.** If aug helps the CNN tower a bit but the gate is
still 0.9/0.1, the SF head is still overfitting the CNN representation
of training species. That outcome routes to strategy 2 (frozen-tower
finetune) or strategy 3 (DANN). If aug *hurts* binary F1 by >0.02, the
mutation rate is too high — drop `p_mut` to 0.002.

---

### Strategy 2 — Two-stage: frozen-tower SF-head finetune from epoch 8  ★★★★

**Rationale.** Epoch 8 generalises better than epoch 30 despite a
*lower* validation score. The towers (CNN + GNN + fusion) at epoch 8
already encode features that transfer; what does not transfer is the
SF softmax head's later refinement, which fits training-species
manifolds. Freezing the towers at epoch 8 and finetuning only the SF
head with strong regularisation (label smoothing, raised dropout,
weight decay) directly tests this hypothesis with the smallest possible
engineering surface and zero re-featurization cost.

**Implementation sketch.** New script
`finetune_sf_head_from_epoch8.py` in
`data_analysis/vgp_model_split_fix/v4/`. Reuse `eval_epoch8.py`'s
notebook-exec pattern.

```python
# load epoch 8
ckpt = torch.load("cluster session/hybrid_v4_epoch8.pt", weights_only=False)
model = HybridTEClassifierV4(**ckpt["arch"], dropout=DROPOUT).to(device)
model.load_state_dict(ckpt["model_state_dict"])

# freeze towers + fusion (sf_head is named `superfamily_head` in v4)
for n, p in model.named_parameters():
    if not n.startswith("superfamily_head."):
        p.requires_grad_(False)

# raise dropout in superfamily_head from 0.1 to 0.4
for m in model.superfamily_head.modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.4

opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-4, weight_decay=5e-3,
)
sf_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=sf_weights_t)

# DNA-only finetune: skip non-DNA samples entirely
for ep in range(10):
    for batch in loader_train_dna_only:
        ... # standard step, only optimize sf_head
```

**Eval plan.** Track val SF macro F1 across 10 finetune epochs.
**Success bar: ≥+0.04 SF macro F1 over the un-finetuned epoch-8
baseline.** Wall-clock: 1–2 h on A100 (DNA-only training set is
~10 k samples; tower forward pass dominates but is a single fp16 pass
since no gradient flows back).

**Failure mode.** If the finetuned head plateaus at ≤epoch-8 baseline,
the bottleneck is the towers themselves (representation lacks generic
SF features) → fall back to strategy 1 or 3 to fix the towers.

---

### Strategy 3 — DANN-style species-adversarial head  ★★★

**Rationale.** Direct attack on the memorisation pathway: add a small
species-classifier head reading from the **fused embedding**, with a
gradient-reversal layer between fused embedding and the species head.
Training minimises class+SF loss while *maximising* species-classifier
loss, pushing the fused embedding toward a species-invariant manifold.
This is the textbook fix when train and test domains differ in some
factor (here: species) and you want labels (here: SF) that should be
domain-invariant. Empirically it can over-regularise (lose useful
species-correlated signal) so warmup and λ tuning matter.

**Implementation sketch.** ~60 LOC.

```python
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

# in HybridTEClassifierV4.__init__:
self.species_head = nn.Sequential(
    nn.Linear(fusion_dim, fusion_dim),
    nn.GELU(), nn.Dropout(0.2),
    nn.Linear(fusion_dim, num_species_train),
)

# in forward, after fused = ...
species_logits = self.species_head(grad_reverse(fused, self.lambd))
return class_logits, sf_logits, gate, species_logits

# in train loop:
lambd = 2.0 / (1 + math.exp(-10 * progress)) - 1.0   # DANN schedule
total_loss = cls_loss + sf_loss + lambda_dann * species_loss
```

Need to map training species names to integer ids and build a
`y_species` per sample.

**Eval plan.** Sweep `λ_max ∈ {0.1, 0.3, 1.0}` × 3 runs, pick best on
held-out **val** species. **Success bar: gate weights move back toward
the random-split distribution (0.4/0.6) AND SF macro F1 ≥+0.05.**
Wall-clock: ~3× baseline due to mini-sweep (~24 h CSD3 total).

**Failure mode.** If too high λ → SF F1 drops globally including on
training species (over-regularised). If too low → no effect on gate
distribution. The diagnostic is the per-species spread of fused
embeddings (UMAP) before vs after.

---

### Strategy 4 — Prototypical-network SF head  ★★★

**Rationale.** Replace the SF softmax head with a **cosine-similarity
to per-class prototypes** classifier. Prototypes are mean fused
embeddings of training-set examples per SF, recomputed each epoch.
Predictions = nearest prototype. This typically generalises better
under domain shift than a parametric softmax because there is no
class-specific weight matrix to overfit to training-species scaling
quirks; the classifier's only flexibility is the prototype location,
which averages over many training species.

**Implementation sketch.** ~40 LOC, replaces `model.sf_head` only.

```python
class ProtoSFHead(nn.Module):
    def __init__(self, fusion_dim, n_classes, tau=10.0):
        super().__init__()
        self.proj = nn.Linear(fusion_dim, fusion_dim)
        self.register_buffer("prototypes",
                             torch.zeros(n_classes, fusion_dim))
        self.tau = tau
    def forward(self, fused):
        z = F.normalize(self.proj(fused), dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        return self.tau * z @ p.t()  # logits
    @torch.no_grad()
    def recompute(self, fused, y_sf, dna_mask):
        z = F.normalize(self.proj(fused[dna_mask]), dim=-1)
        for c in range(self.prototypes.size(0)):
            sel = (y_sf[dna_mask] == c)
            if sel.any():
                self.prototypes[c] = z[sel].mean(0)
```

Recompute prototypes once per epoch on a subsample of the training
set.

**Eval plan.** Same as strategies 1/2. **Success bar: ≥+0.04 SF macro
F1.** Wall-clock: ≈baseline.

**Failure mode.** If prototypes drift wildly across epochs (instability),
add EMA on prototypes (`α=0.95`). If F1 *drops* under proto head, the
SF problem is not classifier-level but representation-level —
strategies 1/3 are the right answer.

---

### Strategy 5 — Phylogenetically-rebalanced sampling  ★★

**Rationale.** Cell 22 in the GPU notebook caps each non-DNA tag to
3000 samples *globally*, and the per-fold cap subsamples DNA per fold
without species awareness. This means a few species with many TE
copies can dominate any given training class. If the per-species
diagnostic (above) shows hAT failures concentrated on 1–2 species, the
fix is to cap not just `(tag)` but `(tag, species)` so no species
contributes more than e.g. 200 copies to any SF. The expected effect
is biggest for hAT (where dominant training species may be
overrepresented), smallest for already-balanced classes like
TcMar-Tc1 (which transfers fine).

**Implementation sketch.** Modify the global cap function in cell 22.

```python
# replace the per-tag cap:
keep = []
for tag, idxs in by_tag.items():
    by_sp = defaultdict(list)
    for i in idxs:
        by_sp[species[i]].append(i)
    sub = []
    per_sp_cap = max(50, MAX_PER_SF // max(len(by_sp), 1))
    for sp, sp_idxs in by_sp.items():
        if len(sp_idxs) > per_sp_cap:
            sp_idxs = rng.choice(sp_idxs, per_sp_cap, replace=False).tolist()
        sub.extend(sp_idxs)
    if len(sub) > MAX_PER_SF:
        sub = rng.choice(sub, MAX_PER_SF, replace=False).tolist()
    keep.extend(sub)
```

Apply to both DNA (per-fold cap) and non-DNA (global cap) paths.

**Eval plan.** Re-train. **Success bar: hAT recall on the worst-2
test species improves by ≥0.10 (per-species diagnostic).** Wall-clock:
≈baseline.

**Failure mode.** If per-species diagnostic shows uniform hAT failure
(no concentration), this strategy will yield ≤0.02 improvement and
should not be a priority. That's exactly why we run the diagnostic
first.

---

## 4. Deprioritised, with reasons

- **Mixup / Manifold mixup across species.** Strategy 1 (augmentation)
  attacks the same memorisation pathway with a stronger inductive bias
  (preserving the underlying motif while perturbing exact k-mers).
  Mixup is more general but harder to reason about for biological
  sequences; only attempt if strategy 1 underperforms.
- **Per-species centroid subtraction at inference.** Test species are
  unseen, so the centroid would have to be estimated from the test
  batch itself, which leaks across-test-sample information and is
  brittle to small batches. Skip.
- **Hardest-species-out CV ensemble.** Requires N additional full
  trainings (N ≥ 4 to be informative). Busts the 8 h/experiment
  budget. Skip unless ensembling existing checkpoints from different
  seeds turns out to be cheap.
- **Larger pretrained backbones (Nucleotide Transformer, DNABERT).**
  Explicitly excluded by user — engineering cost not justified for a
  bonus chapter.
- **Hyperparameter sweep on existing config.** Explicitly excluded.
  Won't change the species-disjoint behaviour qualitatively.
- **Relabelling RepeatModeler outputs.** Out of scope per user.
- **Collecting more data.** Out of scope per user.

---

## 5. Smoke-test target

Strategy 1 (augmentation) is the smoke target because:

1. It is **drop-in additive** — no new head, no new loss.
2. It is **wall-clock equivalent** to baseline at full scale, and
   trivially cheap on a 5000-sample subset.
3. It directly addresses the gate-collapse signal (which strategies 2
   and 4 do not).
4. It needs no checkpoint to load (strategy 2 does), so the smoke
   exercises the full training loop end-to-end.

See [`smoke_aug/run_smoke_aug.py`](smoke_aug/run_smoke_aug.py) and
[`smoke_aug/README.md`](smoke_aug/README.md). The smoke runs both an
aug arm and a `--no-aug` control with `EPOCHS=3, SUBSET_SIZE=5000` so
the A/B delta on val SF macro F1 is measurable in <1h on MPS.

---

## 6. Open questions for follow-up

1. **Phase-1 diagnostic outcome may rerank.** If hAT misses concentrate
   on ≤2 species, strategy 5 outranks strategy 1 and the smoke test
   should be retargeted.
2. **Combining strategies.** 1 + 2 (aug then frozen-tower finetune) is
   the obvious next step if both individually clear +0.05.
3. **GNN tower necessity.** A control run with the GNN tower removed
   (CNN-only) on species-disjoint would establish whether the GNN is a
   net negative for cross-species generalisation. Cheap to do; not
   listed above because it's a control, not a fix.
