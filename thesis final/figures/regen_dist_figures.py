"""
Regenerate superfamily_class_distribution.png as a single "All Data" pie chart
with a legend instead of inline labels to avoid label overlap for minor classes.
Run from: thesis/figures/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LABEL_PATH = os.path.join(os.path.dirname(__file__), '../../data/vgp/20260120_features_sf')

def analyze_labels(label_path):
    superfamilies = Counter()
    top_classes = Counter()
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tag = parts[1]
            superfamilies[tag] += 1
            top_class = tag.split('/')[0]
            top_classes[top_class] += 1
    return superfamilies, top_classes

sf_counts, class_counts = analyze_labels(LABEL_PATH)
total = sum(class_counts.values())

# Sort by count descending
classes = [cls for cls, _ in sorted(class_counts.items(), key=lambda x: -x[1])]
counts  = [class_counts[cls] for cls in classes]

# Use Set2 colours
cmap = plt.cm.Set2
colors = [cmap(i / len(classes)) for i in range(len(classes))]

# --- Single "All Data" pie with legend ---
fig, ax = plt.subplots(figsize=(7, 6))

# For very small slices (< 1%) suppress inline pct label to avoid overlap;
# show them only in the legend
SMALL_THRESHOLD = 1.0  # percent

def autopct_fn(pct):
    return f'{pct:.1f}%' if pct >= SMALL_THRESHOLD else ''

wedges, texts, autotexts = ax.pie(
    counts,
    labels=None,          # labels go into legend
    autopct=autopct_fn,
    colors=colors,
    explode=[0.03] * len(classes),
    startangle=90,
    pctdistance=0.75,
)

# Style the percentage text
for at in autotexts:
    at.set_fontsize(9)

# Build legend with class name + count + pct
legend_labels = [
    f'{cls}  ({class_counts[cls]:,}, {100*class_counts[cls]/total:.1f}%)'
    for cls in classes
]
ax.legend(wedges, legend_labels, title='Class', loc='lower left',
          bbox_to_anchor=(0.0, -0.18), fontsize=9, title_fontsize=9,
          ncol=2, frameon=True)

ax.set_title('Top-Level Class Distribution (All Data)', fontsize=12, pad=12)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), 'superfamily_class_distribution.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved {out_path}')
plt.close()

# --- Bar chart: top 30 superfamilies (excluding None), portrait aspect to match pie height ---
sf_sorted = sorted(sf_counts.items(), key=lambda x: -x[1])
sf_names  = [sf for sf, _ in sf_sorted if sf != 'None'][:30]
sf_values = [sf_counts[sf] for sf in sf_names]

class_colors = {
    'DNA':  '#1f77b4', 'LTR':  '#ff7f0e', 'LINE': '#2ca02c',
    'SINE': '#d62728', 'PLE':  '#9467bd', 'RC':   '#8c564b',
}
bar_colors = [class_colors.get(sf.split('/')[0], '#7f7f7f') for sf in sf_names]

# figsize=(11, 10) → aspect ratio 1.10:1 so that at 0.55\textwidth the rendered
# height ≈ 0.50\textwidth, matching the portrait pie chart at 0.42\textwidth.
fig, ax = plt.subplots(figsize=(11, 10))
ax.barh(range(len(sf_names)), sf_values, color=bar_colors)
ax.set_yticks(range(len(sf_names)))
ax.set_yticklabels(sf_names, fontsize=16)
ax.tick_params(axis='x', labelsize=14)
ax.invert_yaxis()
ax.set_xlabel('Count', fontsize=18)
ax.set_title('Top 30 Superfamilies (excluding None)', fontsize=20)

# Annotate bar values
for i, v in enumerate(sf_values):
    ax.text(v + 60, i, str(v), va='center', fontsize=14)

# Legend
import matplotlib.patches as mpatches
legend_elements = [mpatches.Patch(facecolor=c, label=cls) for cls, c in class_colors.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=16, title='Class',
          title_fontsize=16, frameon=True)

plt.tight_layout()
bar_path = os.path.join(os.path.dirname(__file__), 'superfamily_distribution_bar.png')
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
print(f'Saved {bar_path}')
plt.close()
