# Full-corpus predictions summary

- Total sequences: 135,751
- Outputs:
  - `all_predictions_v4.3_rotating_full.csv` (135,751 rows)
  - `all_predictions_v4.3_singlefold_full.csv` (135,751 rows)

## v4.3_rotating
### outer_role distribution
```
outer_role
untracked    124723
trainval       7469
test           3180
excluded        379
```
### Predicted class for `pantera_tpase_tag == None` (n=125,101)
```
pred_class
DNA     57093
LTR     40090
LINE    27918
```
### Mean `pred_class_conf` for None, by predicted class
```
            count      mean       std
pred_class                           
DNA         57093  0.995083  0.037332
LINE        27918  0.992641  0.046343
LTR         40090  0.995317  0.036535
```
## v4.3_singlefold
### outer_role distribution
```
outer_role
untracked    124723
trainval       7469
test           3180
excluded        379
```
### Predicted class for `pantera_tpase_tag == None` (n=125,101)
```
pred_class
DNA     58557
LTR     38652
LINE    27892
```
### Mean `pred_class_conf` for None, by predicted class
```
            count      mean       std
pred_class                           
DNA         58557  0.996130  0.032967
LINE        27892  0.992190  0.047195
LTR         38652  0.993978  0.041935
```
## Agreement: v4.3_rotating vs v4.3_singlefold
- overall: pred_class agree 0.9281, pred_sf agree 0.6855
- on Pantera-tag == None: pred_class agree 0.9239, pred_sf agree 0.6782
- cross-tab pred_class on None:
```
v4.3_singlefold    DNA   LINE    LTR
v4.3_rotating                       
DNA              54052   1835   1206
LINE              2071  24966    881
LTR               2434   1091  36565
```
