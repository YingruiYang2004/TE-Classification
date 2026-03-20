#!/usr/bin/env python3
"""Analyze V4.1 model training results from checkpoint."""
import torch
from pathlib import Path

script_dir = Path(__file__).parent

# Check all available checkpoints
print("=== Available Checkpoints ===")
for f in sorted(script_dir.glob("hybrid_v4.1_epoch*.pt")):
    ckpt_temp = torch.load(f, map_location='cpu', weights_only=False)
    score = ckpt_temp.get('score', 'N/A')
    score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
    print(f"  {f.name}: score = {score_str}")

# Load epoch 59 checkpoint
ckpt = torch.load(script_dir / 'hybrid_v4.1_epoch59.pt', map_location='cpu', weights_only=False)

print("=== V4.1 Model Training Results ===\n")
print(f"Keys: {list(ckpt.keys())}")
print(f"Best Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"Score: {ckpt.get('score', 'N/A'):.4f}" if ckpt.get('score') else "Score: N/A")

print(f"\nSuperfamilies ({len(ckpt['superfamily_names'])}):")
for sf in ckpt['superfamily_names']:
    print(f"  - {sf}")

print("\n=== Architecture ===")
arch = ckpt.get('arch', {})
for k, v in arch.items():
    print(f"  {k}: {v}")

history = ckpt.get('history', {})
if history:
    print("\n=== Training History (last 5 epochs) ===")
    n_epochs = len(history.get('train_loss', []))
    print(f"Total epochs trained: {n_epochs}")
    
    start = max(0, n_epochs - 5)
    print(f"\nEpoch | Train Loss | Val Bin F1 | Val SF F1 | Combined | Fold")
    print("-" * 70)
    for i in range(start, n_epochs):
        fold = history['fold_used'][i] if 'fold_used' in history else 'N/A'
        combined = 0.5 * history['val_binary_f1'][i] + 0.5 * history['val_sf_f1'][i]
        print(f"{i+1:5d} | {history['train_loss'][i]:.4f}     | {history['val_binary_f1'][i]:.4f}     | {history['val_sf_f1'][i]:.4f}    | {combined:.4f}   | {fold}")

    print("\n=== Best Metrics from History ===")
    best_bin_f1 = max(history.get('val_binary_f1', [0]))
    best_sf_f1 = max(history.get('val_sf_f1', [0]))
    best_bin_idx = history['val_binary_f1'].index(best_bin_f1) if best_bin_f1 > 0 else 0
    best_sf_idx = history['val_sf_f1'].index(best_sf_f1) if best_sf_f1 > 0 else 0
    
    print(f"Best Binary F1: {best_bin_f1:.4f} (epoch {best_bin_idx + 1})")
    print(f"Best SF F1: {best_sf_f1:.4f} (epoch {best_sf_idx + 1})")
    
    # Average metrics over all epochs
    print("\n=== Average Metrics (across all epochs) ===")
    avg_bin_f1 = sum(history['val_binary_f1']) / len(history['val_binary_f1'])
    avg_sf_f1 = sum(history['val_sf_f1']) / len(history['val_sf_f1'])
    print(f"Avg Binary F1: {avg_bin_f1:.4f}")
    print(f"Avg SF F1: {avg_sf_f1:.4f}")
    
    # Gate weights trend
    print("\n=== Gate Weight Trend ===")
    avg_cnn = sum(history['gate_weights_cnn']) / len(history['gate_weights_cnn'])
    avg_gnn = sum(history['gate_weights_gnn']) / len(history['gate_weights_gnn'])
    print(f"Avg CNN weight: {avg_cnn:.4f}")
    print(f"Avg GNN weight: {avg_gnn:.4f}")
