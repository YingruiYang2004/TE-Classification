# Report Outline - Email Reply Points
**Yingrui Yang | Part III Systems Biology 6506D | Machine Learning a Transposable Element Classifier**
_Prepared: 17 April 2026_

---

## Copy-ready email points

- I have drafted the report structure with major and minor headings, and measured current progress using texcount from the LaTeX source.
- Current thesis source count is 2757 text words (plus 108 words in headings and 38 words in captions/outside text).
- For drafting progress, I track only text words.
- I am now working to a **5000-word main-body target** (excluding references).
- Current main-body progress is **1962 / 5000 words (39.2%)**.

### Major headings: words written vs target

- Summary: **2 / 150** (placeholder only)
- Introduction: **2 / 550** (placeholder only)
- Results: **1339 / 2500** (partial draft)
- Discussion: **2 / 1100** (placeholder only)
- Materials and Methods: **615 / 650** (near-complete first draft)
- Acknowledgements: **2 / 50** (placeholder only)
- References: started (1 citation entered)

### Minor headings: words written vs target

- Results 2.1 Terminal Inverted Repeat Detection: **339 / 350** (full draft; still needs final metrics values)
- Results 2.2 Hierarchical DNA Transposon Classification (v3): **467 / 500** (full draft)
- Results 2.3 Hybrid CNN and K-mer Graph Neural Network Model (v4): **329 / 350** (full draft)
- Results 2.4 Multi-Class Extension: Three-Class Classification (v4.2 and v4.3): **154 / 500** (scaffold only)
- Results 2.5 External Benchmark: Mini Dataset Evaluation: **50 / 400** (scaffold only)
- Results 2.6 Unsupervised Clustering of TE sequences: **0 / 250** (not drafted)
- Results 2.7 Placeholder section: **0 / 150** (not drafted)

- Methods 4.1 Data: **165 / 170** (full draft)
- Methods 4.2 Model Architectures: **255 / 255** (full draft)
- Methods 4.3 Training: **158 / 160** (full draft)
- Methods 4.4 Evaluation: **23 / 40** (stub)
- Methods 4.5 Implementation: **14 / 25** (stub)

### Draft-state summary

- Full or near-full draft: Results 2.1-2.3, Methods 4.1-4.3.
- Mostly placeholders or stubs: Summary, Introduction, Discussion, Acknowledgements, Results 2.4-2.7, Methods 4.4-4.5.

### Proposed figures and tables (working versions or placeholders)

- Fig. 1 (placeholder): RC-invariant CNN architecture diagram for TIR detection.
- Fig. 2 (working): v3 DNA superfamily confusion matrix from [data_analysis/vgp_model_data_tpase_multi/rc_cnn_multi_v3_confusion.png](data_analysis/vgp_model_data_tpase_multi/rc_cnn_multi_v3_confusion.png).
- Fig. 3 (working): hybrid training curves and gate-weight evolution from [data_analysis/vgp_model_data_tpase_multi/v4.3/hybrid_v4_training_curves.png](data_analysis/vgp_model_data_tpase_multi/v4.3/hybrid_v4_training_curves.png) and [data_analysis/vgp_model_data_tpase_multi/v4.3/gate_weights_evolution.png](data_analysis/vgp_model_data_tpase_multi/v4.3/gate_weights_evolution.png).
- Fig. 4 (working): class-level and superfamily-level confusion matrices from [data_analysis/vgp_model_data_tpase_multi/v4.3/hybrid_v4_confusion_matrices.png](data_analysis/vgp_model_data_tpase_multi/v4.3/hybrid_v4_confusion_matrices.png).
- Fig. 5 (working): class separability in k-mer space from [data_analysis/vgp_model_data_tpase_multi/v4.3/kmer_class_separation.png](data_analysis/vgp_model_data_tpase_multi/v4.3/kmer_class_separation.png).
- Fig. 6 (working): misclassification structure plot from [data_analysis/vgp_model_data_tpase_multi/v4.3/misclassification_analysis_v4.3.png](data_analysis/vgp_model_data_tpase_multi/v4.3/misclassification_analysis_v4.3.png).
- Fig. 7 (working): saliency analysis plot from [data_analysis/vgp_model_data_tpase_multi/v4.3/saliency_analysis_v4.3.png](data_analysis/vgp_model_data_tpase_multi/v4.3/saliency_analysis_v4.3.png).
- Fig. 8 (working): global embedding UMAP from [data_analysis/vgp_model_clustering/global_umap_superfamily.png](data_analysis/vgp_model_clustering/global_umap_superfamily.png).
- Fig. 9 (working): DNA subcluster map from [data_analysis/vgp_model_clustering/dna_superfamily_subclusters.png](data_analysis/vgp_model_clustering/dna_superfamily_subclusters.png).

- Table 1 (placeholder; data ready): class-level and superfamily-level performance summary from [data_analysis/vgp_model_data_tpase_multi/v4.3/all_test_predictions_v4.3.csv](data_analysis/vgp_model_data_tpase_multi/v4.3/all_test_predictions_v4.3.csv).
- Table 2 (placeholder; data ready): top misclassification patterns from [data_analysis/vgp_model_data_tpase_multi/v4.3/misclassification_patterns_v4.3.csv](data_analysis/vgp_model_data_tpase_multi/v4.3/misclassification_patterns_v4.3.csv).
- Table 3 (placeholder; data ready): per-genome error summary from [data_analysis/vgp_model_data_tpase_multi/v4.3/misclassification_by_genome_v4.3.csv](data_analysis/vgp_model_data_tpase_multi/v4.3/misclassification_by_genome_v4.3.csv).
- Table 4 (placeholder): benchmark species and library sizes (already scaffolded in Results 2.5).
- Table 5 (placeholder): benchmark species-level and overall performance (already scaffolded in Results 2.5).
- Table 6 (placeholder): hyperparameters by model version.

### Immediate next writing priorities

- Expand Results 2.4 and 2.5 from scaffold to full narrative.
- Draft Results 2.6 (clustering and saliency interpretation).
- Write full Discussion, then Summary and Introduction.
- Finalize methods stubs (Evaluation and Implementation) and populate placeholder tables.