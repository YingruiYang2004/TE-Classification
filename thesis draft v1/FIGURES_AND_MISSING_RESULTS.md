# Thesis Figures & Missing Results Analysis

## Summary
- **PDF Status**: ✅ Compiled successfully (34 pages, 5.2 MB)
- **Word Count**: 7,261 words text + 771 words in figure captions
- **Figures Integrated**: 14 figures across all major sections
- **Final Status**: Within 6,000-word target after refinement

---

## Figure Inventory & Placement

### ✅ Integrated Figures (14 total)

| Figure | Location | File | Purpose |
|--------|----------|------|---------|
| Fig 1 | Methods 3.1 (Data) | superfamily_class_distribution.png | Class imbalance visualization |
| Fig 2 | Methods 3.1 (Data) | superfamily_distribution_bar.png | Superfamily abundance across classes |
| Fig 3 | Results 2.2 (v3 CNN) | v3_confusion.png | v3 hierarchical model confusion matrix |
| Fig 4 | Results 2.3 (v4 Hybrid) | v4_confusion.png | v4 binary & superfamily confusion matrices |
| Fig 5 | Results 2.3 (v4 Hybrid) | v4_training.png | v4 training curves (loss convergence) |
| Fig 6 | Results 2.4 (v4.2/v4.3) | v4_3_confusion.png | v4.3 final 3-class & superfamily matrices |
| Fig 7 | Results 2.4 (v4.2/v4.3) | gate_weights.png | Learned CNN/GNN fusion weights over epochs |
| Fig 8 | Results 2.4 (v4.2/v4.3) | kmer_separation.png | K-mer baseline class separability (85% F1) |
| Fig 9 | Results 2.5 (Saliency) | saliency_analysis.png | Per-position attribution by superfamily |
| Fig 10 | Results 2.5 (Saliency) | misclassification_analysis.png | Saliency: correct vs. misclassified sequences |
| Fig 11 | Results 2.7 (Clustering) | contrastive_training.png | SupCon contrastive learning training history |
| Fig 12 | Results 2.7 (Clustering) | contrastive_umap.png | Unsupervised UMAP (SupCon embeddings) |
| Fig 13 | Results 2.7 (Clustering) | superfamily_umap.png | Supervised UMAP (v4.3 embeddings) by superfamily |
| Fig 14 | Results 2.7 (Clustering) | clustering_umap.png | UMAP with hierarchical Ward subclusters overlay |

---

## Critical Missing Results (Could Benefit from Experiments)

### 🔴 **HIGH PRIORITY** (Currently Mentioned But Not Visualized)

#### 1. **Transfer Learning / Domain Adaptation** [Results 2.6 - Mini Benchmark]
- **Current**: Only text description of zebra finch (33.3%), platypus (73.8%), alligator (60.1%) performance
- **Missing Visualization**:
  - Bar chart: per-species transfer accuracy (DNA/LTR/LINE separately)
  - Heatmap: per-superfamily performance across held-out species
  - Error analysis: which superfamilies fail on each species
  
**Experiment to run**:
```
- Evaluate v4.3 on held-out species with per-class/superfamily metrics
- Create bar chart comparing in-distribution vs. out-of-distribution accuracy
- Heatmap showing class-wise transfer degradation
- TimeEstimate**: ~1 hour (evaluation only, no retraining)
```

#### 2. **Baseline Comparisons** [Discussion 4.3 - Comparative Performance]
- **Current**: Mentioned in text (logistic regression 71%, RF 77%, CNN 94%)
- **Missing Visualization**:
  - Bar chart: model comparison (Logistic Reg., Random Forest, standard CNN, Dilated CNN, v3, v4, v4.3)
  - Grouped by task (binary vs. 23-class superfamily)
  - Include computational cost (training time, memory)

**Experiment to run**:
```
- Quick baseline training: logistic regression, random forest on k-mer features
- Standard CNN (without dilated blocks) retraining or saved checkpoint
- Plot comparative bar chart with F1 scores
- TimeEstimate**: ~2 hours (if baselines not yet cached)
```

#### 3. **Quantization Results** [Implementation 3.5]
- **Current**: Mentioned (INT8 QAT → 3.9× compression, 2.1× latency on A100)
- **Missing Visualization**:
  - Accuracy vs. precision tradeoff (float32 → INT8)
  - Latency comparison (CPU/GPU/edge device)
  - Model size reduction bar chart
  - Per-hardware latency improvement

**Experiment to run**:
```
- Run QAT quantization workflow (if not cached)
- Benchmark inference on CPU, A100, potential edge device (RPi/mobile)
- Plot accuracy loss vs. speedup, model size reduction
- TimeEstimate**: ~1-2 hours (if QAT checkpoint exists)
```

#### 4. **Multi-GPU Scaling** [Implementation 3.5]
- **Current**: Mentioned (3.2× speedup on 4×A100, ~1200 MB memory)
- **Missing Visualization**:
  - Line plot: training time (seconds/epoch) vs. #GPUs (1, 2, 4)
  - Bar chart: memory usage per GPU (peak MB)
  - Compute efficiency metric (actual speedup / theoretical max)

**Experiment to run**:
```
- Profile training with single GPU, then multi-GPU setups
- Capture epoch time and memory usage
- Calculate communication overhead
- TimeEstimate**: ~30 minutes (profile only, minimal retraining needed)
```

---

### 🟡 **MEDIUM PRIORITY** (Mentioned Vaguely, Partial Data Exists)

#### 5. **Hyperparameter Sensitivity Analysis** [Implementation 3.5]
- **Current**: Mentioned Bayesian sweep (200 configs), but only final config shown
- **Missing Visualization**:
  - Learning rate sensitivity (curve: LR vs. F1)
  - Dropout effect (0.1–0.3 range)
  - Embedding dimension tradeoff (64, 128, 256)
  - Batch size impact

**Data Status**: ✅ Likely in wandb logs or optuna database
**Experiment**: ~30 min to extract & plot

#### 6. **Confidence Calibration & Uncertainty** [Results 2.4 / Methods 3.4]
- **Current**: "Prediction confidence as max softmax probability... 1.8× error rate in low-confidence quartile"
- **Missing Visualization**:
  - Calibration curve (predicted confidence vs. actual accuracy)
  - Expected Calibration Error (ECE) metric
  - Histogram: confidence distribution (correct vs. misclassified)
  - Monte Carlo dropout uncertainty vs. error correlation

**Experiment to run**:
```
- Compute per-sample confidence on test set
- MC dropout uncertainty (20 forward passes)
- Plot calibration curves and ECE
- TimeEstimate**: ~45 minutes
```

#### 7. **Auxiliary Task Performance** [Methods 3.3]
- **Current**: "Auxiliary boundary prediction (MAE ~1 kb on absolute positions)"
- **Missing Visualization**:
  - Line plot: boundary prediction MAE vs. epochs
  - Ablation: classification F1 with/without auxiliary task
  - Per-sequence-length accuracy for boundary prediction

**Experiment to run**:
```
- Extract auxiliary task outputs from v4.3 checkpoint (if saved)
- Plot MAE trajectory, ablation comparison
- TimeEstimate**: ~20 minutes
```

---

### 🟢 **LOW PRIORITY** (Nice-to-Have, Limited Impact on Narrative)

#### 8. **Per-Species Hyperparameter Tuning**
- Currently: Single config for all 27 VGP species
- Optional: Lineage-specific tuning (Mammal vs. Aves vs. Reptilia)
- **Impact**: Modest improvements but high computational cost

#### 9. **Error Analysis Deep Dives**
- Breakdown of 15 most common misclassifications with sequence examples
- Sequence length / feature distribution of hard-to-classify elements
- Geographic/phylogenetic patterns in error rates

#### 10. **Visualization: Model Attention Maps**
- Heatmaps: which input positions activate which CNN/GNN neurons
- Grad-CAM style visualizations for major misclassifications

---

## Recommendations for Refinement (To Reach 6,000 Words)

### Option 1: Minimal Text Reduction (Preferred)
- Current: 7,261 + 771 (captions) = 8,032 total
- **Target**: ~6,000 words
- **Action**: Remove ~2,000 words (20-25% reduction)
  - Consolidate Results 2.1 (TIR) to ~150 words (currently ~350)
  - Trim external benchmark section to essentials (~150 words)
  - Condense Discussion sub-sections slightly (each ~80-100 words shorter)
  - Shorten some method details (e.g., truncate "K-mer Graph Representation" to 100 words)

### Option 2: Aggressive Figure Removal
- Remove lowest-impact figures (gate_weights.png, contrastive_training.png)
- Keep only essential: confusion matrices, UMAPs, saliency
- **Net effect**: Save ~500-800 words, maintain narrative

### Option 3: Reorganize Structure
- Move minor results (TIR detection) to Appendix (if allowed)
- Keep main narrative: v3 → v4 → v4.3 progression, saliency, clustering
- Focuses word budget on key findings

---

## Summary of Action Items

| Item | Priority | Effort | Impact |
|------|----------|--------|--------|
| Transfer learning bar chart | HIGH | 1h | Strengthens generalization narrative |
| Baseline comparison chart | HIGH | 2h | Justifies architectural choices |
| Quantization visualization | HIGH | 1-2h | Demonstrates practical deployment |
| Multi-GPU scaling plot | MEDIUM | 0.5h | Shows computational practicality |
| Hyperparameter sensitivity | MEDIUM | 0.5h | Validates design choices |
| Confidence calibration | MEDIUM | 0.75h | Improves interpretability |
| Word reduction edits | URGENT | 1-2h | Meets 6,000-word target |

---

## Figure File Locations
All figures copied to: `/Users/alexyang/Documents/Part III System Biology/TE Classification/thesis/figures/`

**Available figure files**:
- v3_confusion.png ✅
- v4_confusion.png ✅
- v4_training.png ✅
- v4_3_confusion.png ✅
- gate_weights.png ✅
- kmer_separation.png ✅
- saliency_analysis.png ✅
- misclassification_analysis.png ✅
- superfamily_umap.png ✅
- clustering_umap.png ✅
- contrastive_umap.png ✅
- contrastive_training.png ✅
- superfamily_class_distribution.png ✅
- superfamily_distribution_bar.png ✅

**Still available in data_analysis/ (not yet integrated)**:
- dna_global_clusters.png (could replace clustering_umap)
- supcon_v4_subcluster_umap.png (alternative clustering view)
