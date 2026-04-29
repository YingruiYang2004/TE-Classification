# Mini-benchmark held-out genome evaluation
Three v4 variants evaluated on three genomes excluded from VGP training (`bTaeGut`, `mOrnAna`, `rAllMis`).
Top-level (3-class) metrics use ALL DNA/LTR/LINE entries. Superfamily metrics restrict to entries whose true SF tag is in the model's 23-SF vocabulary.

## `v4.2_epoch28`

| genome | n | n_sf | cls_macroF1 | cls_bal_acc | DNA F1 | DNA rec | LTR F1 | LINE F1 | sf_macroF1 | sf_bal_acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bTaeGut | 414 | 172 | 0.616 | 0.740 | 0.599 | 0.827 | 0.709 | 0.542 | 0.128 | 0.341 |
| mOrnAna | 208 | 125 | 0.681 | 0.697 | 0.766 | 0.720 | 0.532 | 0.744 | 0.169 | 0.312 |
| rAllMis | 1032 | 600 | 0.687 | 0.678 | 0.765 | 0.844 | 0.678 | 0.618 | 0.135 | 0.313 |
| **overall** | **1654** | **897** | **0.681** | **0.692** | **0.740** | **0.820** | **0.678** | **0.626** | **0.148** | **0.318** |

## `v4.3_rotating_epoch40`

| genome | n | n_sf | cls_macroF1 | cls_bal_acc | DNA F1 | DNA rec | LTR F1 | LINE F1 | sf_macroF1 | sf_bal_acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bTaeGut | 414 | 172 | 0.639 | 0.738 | 0.560 | 0.933 | 0.733 | 0.623 | 0.168 | 0.502 |
| mOrnAna | 208 | 125 | 0.821 | 0.811 | 0.889 | 0.972 | 0.831 | 0.743 | 0.274 | 0.391 |
| rAllMis | 1032 | 600 | 0.787 | 0.770 | 0.881 | 0.950 | 0.854 | 0.627 | 0.216 | 0.400 |
| **overall** | **1654** | **897** | **0.763** | **0.758** | **0.825** | **0.952** | **0.814** | **0.652** | **0.231** | **0.397** |

## `v4.3_singlefold_epoch28`

| genome | n | n_sf | cls_macroF1 | cls_bal_acc | DNA F1 | DNA rec | LTR F1 | LINE F1 | sf_macroF1 | sf_bal_acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bTaeGut | 414 | 172 | 0.428 | 0.613 | 0.402 | 0.880 | 0.368 | 0.515 | 0.244 | 0.421 |
| mOrnAna | 208 | 125 | 0.703 | 0.674 | 0.826 | 0.935 | 0.571 | 0.712 | 0.235 | 0.313 |
| rAllMis | 1032 | 600 | 0.654 | 0.663 | 0.744 | 0.953 | 0.572 | 0.646 | 0.238 | 0.378 |
| **overall** | **1654** | **897** | **0.607** | **0.647** | **0.688** | **0.940** | **0.505** | **0.627** | **0.250** | **0.345** |

