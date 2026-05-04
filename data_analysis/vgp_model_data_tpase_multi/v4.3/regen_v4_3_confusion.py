"""Regenerate thesis/figures/v4_3_confusion.png with square panels.

Loads test predictions saved during v4.3 (Stage 2 rotating) training and
draws the 3-class and top-10 superfamily confusion matrices, both forced
to a 1:1 box aspect via ax.set_box_aspect(1).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).parent / "results_v4.3.pt"
OUT = ROOT / "thesis/figures/v4_3_confusion.png"

results = torch.load(RESULTS, map_location="cpu", weights_only=False)
class_names = results["class_names"]
sf_names = results["superfamily_names"]

cm_cls = confusion_matrix(results["test_class_true"], results["test_class_pred"])
cm_sf = confusion_matrix(results["test_sf_true"], results["test_sf_pred"])
class_support = cm_sf.sum(axis=1)
top_classes = np.argsort(class_support)[::-1][:10]
cm_sf_top = cm_sf[np.ix_(top_classes, top_classes)]
top_names = [sf_names[i] for i in top_classes]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax1 = axes[0]
sns.heatmap(cm_cls, annot=True, fmt="d", cmap="Blues", ax=ax1,
            xticklabels=class_names, yticklabels=class_names)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("True")
ax1.set_title("Class Classification Confusion Matrix")
ax1.set_box_aspect(1)

ax2 = axes[1]
sns.heatmap(cm_sf_top, annot=True, fmt="d", cmap="Blues", ax=ax2,
            xticklabels=top_names, yticklabels=top_names)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("True")
ax2.set_title("Superfamily Confusion Matrix (Top 10 Classes)")
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
plt.setp(ax2.get_yticklabels(), rotation=0)
ax2.set_box_aspect(1)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
