#!/usr/bin/env python3
"""Analyze the v4 contrastive training results."""

import torch
import pandas as pd
from collections import defaultdict

# Load the final checkpoint
ckpt = torch.load('supcon_v4_final.pt', map_location='cpu', weights_only=False)
history = ckpt['history']
sf_names = ckpt['sf_names']
subcluster_names = ckpt.get('subcluster_names', {})
config = ckpt['config']

print('=== Training Configuration ===')
for k, v in config.items():
    print(f'  {k}: {v}')

print(f'\n=== Superfamilies ({len(sf_names)}) ===')
for sf in sf_names:
    print(f'  - {sf}')

print(f'\n=== Training History ===')
print(f'Total epochs: {len(history["train_loss"])}')
print(f'Final train loss: {history["train_loss"][-1]:.4f}')
print(f'Final val loss: {history["val_loss"][-1]:.4f}')

if 'best_epoch' in history:
    print(f'Best epoch: {history["best_epoch"]}')

# Print clustering metrics if available
if 'val_diagnostics' in history and len(history['val_diagnostics']) > 0:
    last_diag = history['val_diagnostics'][-1]
    print('\n=== Final Validation Diagnostics ===')
    for k, v in last_diag.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')

# Print epoch-by-epoch metrics
print('\n=== Epoch Metrics (first 10 and last 5) ===')
print('Epoch | Train Loss | Val Loss | Val Silhouette')
print('-' * 55)

val_diags = history.get('val_diagnostics', [])
for i in range(min(10, len(history['train_loss']))):
    sil = val_diags[i].get('silhouette', 'N/A') if i < len(val_diags) else 'N/A'
    sil_str = f'{sil:.4f}' if isinstance(sil, float) else str(sil)
    print(f'{i+1:5d} | {history["train_loss"][i]:10.4f} | {history["val_loss"][i]:8.4f} | {sil_str}')

if len(history['train_loss']) > 10:
    print('  ... ')
    for i in range(max(10, len(history['train_loss'])-5), len(history['train_loss'])):
        sil = val_diags[i].get('silhouette', 'N/A') if i < len(val_diags) else 'N/A'
        sil_str = f'{sil:.4f}' if isinstance(sil, float) else str(sil)
        print(f'{i+1:5d} | {history["train_loss"][i]:10.4f} | {history["val_loss"][i]:8.4f} | {sil_str}')

print(f'\n=== Subclusters ({len(subcluster_names)}) ===')
# Group by superfamily
sf_subs = defaultdict(list)
for sub_id, sub_name in subcluster_names.items():
    sf = sub_name.split('::')[0]
    sf_subs[sf].append(sub_name)

for sf in sf_names:
    subs = sf_subs.get(sf, [])
    print(f'  {sf}: {len(subs)} subclusters')

# Print detailed subcluster info
print('\n=== Detailed Subcluster Summary ===')
df = pd.read_csv('supcon_v4_subcluster_summary.csv')
print(df.to_string())
