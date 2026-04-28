# v4.3-lite vs v4.3 — comparison summary

## 1. Held-out VGP test

| model                  |   n_params |   best_epoch |   test_top_acc |   test_top_f1 |   top_gap |   test_sf_acc |   test_sf_f1 |   sf_gap |
|:-----------------------|-----------:|-------------:|---------------:|--------------:|----------:|--------------:|-------------:|---------:|
| v4.3 baseline          |        nan |           40 |         0.9818 |        0.9814 |       nan |        0.8927 |       0.8389 |      nan |
| three_class_balanced   |     441500 |            5 |         0.2763 |        0.1443 |        -0 |        0.0498 |       0.0041 |       -0 |
| three_class_unbalanced |     441500 |            9 |         0.2085 |        0.115  |        -0 |        0.0307 |       0.0026 |       -0 |

## 2. Mini-benchmark (held-out species)

| variant                | species   |   n |   n_top_eval |   n_sf_eval |   top_macro_f1 |   sf_macro_f1 |
|:-----------------------|:----------|----:|-------------:|------------:|---------------:|--------------:|
| three_class_balanced   | bTaeGut   | 543 |          414 |         172 |         0.1022 |        0.0272 |
| three_class_unbalanced | bTaeGut   | 543 |          414 |         172 |         0.1022 |        0.0272 |

## 3. Decision rule

- (a) lite test SF-F1 within 0.02 of baseline
- (b) train→test SF-F1 gap < 0.05
- (c) mini-benchmark top-F1 +0.03 on ≥2 of 3 species (vs baseline SF-F1)

| variant                |   test_sf_f1 |   baseline_sf_f1 |   drop_vs_baseline |   sf_train_test_gap |   mini_bench_pass_count | rule_a_test_f1_close   | rule_b_gap_small   | rule_c_bench_better   | overall_pass   |
|:-----------------------|-------------:|-----------------:|-------------------:|--------------------:|------------------------:|:-----------------------|:-------------------|:----------------------|:---------------|
| three_class_balanced   |       0.0041 |           0.8389 |             0.8348 |                  -0 |                       0 | False                  | True               | False                 | False          |
| three_class_unbalanced |       0.0026 |           0.8389 |             0.8363 |                  -0 |                       0 | False                  | True               | False                 | False          |
