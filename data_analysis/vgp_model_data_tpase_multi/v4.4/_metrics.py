import csv
from collections import defaultdict

rows = []
with open("all_test_predictions_v4.3.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

total = len(rows)
class_correct = sum(1 for r in rows if r['class_correct'] == 'True')
sf_correct = sum(1 for r in rows if r['sf_correct'] == 'True')

print("=== TOP-LEVEL CLASS (DNA/LTR/LINE) ===")
print(f"Total test samples: {total}")
print(f"Class accuracy: {class_correct/total*100:.1f}%  ({class_correct}/{total})")
print(f"\n=== SUPERFAMILY ===")
print(f"Superfamily accuracy: {sf_correct/total*100:.1f}%  ({sf_correct}/{total})")

per_class = defaultdict(lambda: {'total':0,'correct':0,'sf_correct':0})
for r in rows:
    c = r['true_class']
    per_class[c]['total'] += 1
    if r['class_correct'] == 'True': per_class[c]['correct'] += 1
    if r['sf_correct'] == 'True': per_class[c]['sf_correct'] += 1

print("\n=== PER TOP-LEVEL CLASS ===")
for cls in sorted(per_class.keys()):
    d = per_class[cls]
    print(f"  {cls}: n={d['total']}, class_acc={d['correct']/d['total']*100:.1f}%, sf_acc={d['sf_correct']/d['total']*100:.1f}%")

sf_stats = defaultdict(lambda: {'tp':0,'fp':0,'fn':0,'cls':''})
for r in rows:
    t = r['true_superfamily']
    p = r['pred_superfamily']
    sf_stats[t]['cls'] = r['true_class']
    if t == p:
        sf_stats[t]['tp'] += 1
    else:
        sf_stats[t]['fn'] += 1
        sf_stats[p]['fp'] += 1

print("\n=== PER-SUPERFAMILY F1 ===")
f1s = []
for sf in sorted(sf_stats.keys()):
    s = sf_stats[sf]
    tp,fp,fn = s['tp'],s['fp'],s['fn']
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    support = tp+fn
    f1s.append(f1)
    print(f"  {sf:32s} F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  n={support}")

macro_f1 = sum(f1s)/len(f1s)
print(f"\nMacro F1 (superfamily): {macro_f1:.3f}")

# Top-level class F1
print("\n=== TOP-LEVEL CLASS F1 ===")
cls_stats = defaultdict(lambda: {'tp':0,'fp':0,'fn':0})
for r in rows:
    t = r['true_class']
    p = r['pred_class']
    if t == p:
        cls_stats[t]['tp'] += 1
    else:
        cls_stats[t]['fn'] += 1
        cls_stats[p]['fp'] += 1

cls_f1s = []
for cls in sorted(cls_stats.keys()):
    s = cls_stats[cls]
    tp,fp,fn = s['tp'],s['fp'],s['fn']
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    support = tp+fn
    cls_f1s.append(f1)
    print(f"  {cls:10s}: F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  n={support}")

print(f"\nMacro F1 (top-level class): {sum(cls_f1s)/len(cls_f1s):.3f}")

# misclassification patterns
print("\n=== TOP 15 MISCLASSIFICATION PATTERNS ===")
patterns = defaultdict(int)
for r in rows:
    if r['sf_correct'] != 'True':
        patterns[(r['true_superfamily'], r['pred_superfamily'])] += 1
for (t,p), cnt in sorted(patterns.items(), key=lambda x: -x[1])[:15]:
    print(f"  {t:32s} -> {p:32s}  n={cnt}")
