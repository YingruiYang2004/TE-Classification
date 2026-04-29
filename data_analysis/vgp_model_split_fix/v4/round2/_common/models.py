"""Model wrappers for the three Round-2 tracks.

All three expose the same forward signature::

    out = wrapper(X_cnn, mask, x_gnn, edge_index, batch_vec)
    # out["class"] : (B, n_classes)
    # out["sf"]    : (B, n_sf)
    # out["gate"]  : (B, 2)  or  None for T2 (no fusion)
    # out["fused"] : (B, fusion_dim)  the embedding that the SF head reads

The species head is owned by the *training script*, not the wrapper, so
that turning DANN off (``--no-stack``) is a single-line change.

T1 = unmodified V4 (just exposes fused).
T2 = CNN-only (uses V4 CNNTower + heads, no GNN, no fusion gate).
T3 = V4 with GNN replaced by the CompositionalResidualEncoder.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ----------------------------------------------------------------- T1 ---
class V4Wrapper(nn.Module):
    """Thin wrapper that runs the un-modified V4 model and returns a dict."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        # The fused embedding is recomputed here (the V4 forward does not
        # expose it directly) so we mirror the same calls.
        # NOTE: this assumes base.fusion(cnn, gnn) -> (fused, gate).

    def forward(self, x_cnn, mask, x_gnn, edge_index, batch_vec):
        cnn_embed = self.base.cnn_tower(x_cnn, mask)
        gnn_embed = self.base.gnn_tower(x_gnn, edge_index, batch_vec)
        fused, gate = self.base.fusion(cnn_embed, gnn_embed)
        return {
            "class": self.base.class_head(fused),
            "sf": self.base.superfamily_head(fused),
            "gate": gate,
            "fused": fused,
        }


# ----------------------------------------------------------------- T2 ---
class CnnOnlyClassifier(nn.Module):
    """V3-style: CNN tower + class/sf heads. No GNN, no fusion gate.

    `cnn_tower` is the V4 `CNNTower` (RC-invariant). The SF head sits
    directly on the CNN embedding via a small projection, matching the
    fusion_dim used by V4 so the DANN species head has the same width.
    """

    def __init__(self, cnn_tower, n_classes: int, n_sf: int,
                 cnn_dim: int, fusion_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        self.cnn_tower = cnn_tower
        self.proj = nn.Sequential(
            nn.Linear(cnn_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )
        self.superfamily_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, n_sf),
        )

    def forward(self, x_cnn, mask, x_gnn, edge_index, batch_vec):
        # x_gnn / edge_index / batch_vec ignored.
        cnn_embed = self.cnn_tower(x_cnn, mask)
        fused = self.proj(cnn_embed)
        return {
            "class": self.class_head(fused),
            "sf": self.superfamily_head(fused),
            "gate": None,
            "fused": fused,
        }


# ----------------------------------------------------------------- T3 ---
class CompResEncoder(nn.Module):
    """Compositional-residual encoder.

    Takes the same `(x_gnn, batch_vec)` tensors that the GNN tower uses,
    averages window k-mer features per graph (= per sequence), subtracts
    a centroid (per-batch mean for smoke; per-clade EMA option for full
    runs), and projects through a small MLP. Output dim matches the V4
    `gnn_hidden` so it slots into the existing fusion module unchanged.

    Args:
        in_dim: width of x_gnn (kmer_dim + 1 for position channel).
        hidden: output dim (matches gnn_hidden in V4).
        clade_centroid_momentum: if >0 and `clade_ids` provided in forward,
                                 maintain per-clade EMA centroid and use it
                                 instead of per-batch mean. (Default 0.0
                                 = always per-batch mean.)
        n_clades: required if clade_centroid_momentum > 0.
        dropout: dropout in the MLP.
    """

    def __init__(self, in_dim: int, hidden: int = 128, *,
                 clade_centroid_momentum: float = 0.0,
                 n_clades: Optional[int] = None,
                 dropout: float = 0.15):
        super().__init__()
        self.out_dim = hidden
        self.in_dim = in_dim
        self.momentum = float(clade_centroid_momentum)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        if self.momentum > 0:
            assert n_clades is not None, "n_clades required when momentum>0"
            self.register_buffer(
                "clade_centroid", torch.zeros(n_clades, in_dim)
            )
            self.register_buffer(
                "clade_seen", torch.zeros(n_clades, dtype=torch.bool)
            )

    def _scatter_mean(self, x: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
        out = torch.zeros((dim_size, x.size(1)), device=x.device, dtype=x.dtype)
        out.index_add_(0, idx, x)
        cnt = torch.bincount(idx, minlength=dim_size).clamp_min(1).to(x.device).to(x.dtype).unsqueeze(1)
        return out / cnt

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch_vec: torch.Tensor,
                clade_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`edge_index` is accepted for signature parity with GNNTower; ignored."""
        del edge_index  # unused
        B = int(batch_vec.max().item()) + 1 if batch_vec.numel() else 0
        comp = self._scatter_mean(x, batch_vec, dim_size=B)  # (B, in_dim)
        if self.momentum > 0 and clade_ids is not None and self.training:
            with torch.no_grad():
                # Update per-clade EMA from this batch.
                for cid in clade_ids.unique().tolist():
                    sel = clade_ids == cid
                    new = comp[sel].mean(0).detach()
                    if not bool(self.clade_seen[cid]):
                        self.clade_centroid[cid] = new
                        self.clade_seen[cid] = True
                    else:
                        self.clade_centroid[cid].mul_(self.momentum).add_(
                            new, alpha=(1 - self.momentum)
                        )
            centroid = self.clade_centroid[clade_ids]
        elif self.momentum > 0 and clade_ids is not None:
            # Eval: use frozen EMA. Fall back to per-batch mean for unseen clades.
            seen = self.clade_seen[clade_ids]
            centroid = self.clade_centroid[clade_ids].clone()
            if (~seen).any():
                centroid[~seen] = comp[~seen].mean(0, keepdim=True).expand((~seen).sum(), -1)
        else:
            centroid = comp.mean(0, keepdim=True).expand_as(comp)
        residual = comp - centroid
        return self.mlp(residual)


class V4CompResWrapper(nn.Module):
    """V4 with the GNN tower replaced by `CompResEncoder`.

    `clade_ids_buffer` is a transient slot the training loop sets each
    iteration so the encoder can address per-clade centroids without
    changing the standard 5-arg forward signature.
    """

    def __init__(self, base, compres: CompResEncoder):
        super().__init__()
        self.base = base
        self.compres = compres
        self.clade_ids_buffer: Optional[torch.Tensor] = None

    def set_clade_ids(self, clade_ids: Optional[torch.Tensor]) -> None:
        self.clade_ids_buffer = clade_ids

    def forward(self, x_cnn, mask, x_gnn, edge_index, batch_vec):
        cnn_embed = self.base.cnn_tower(x_cnn, mask)
        gnn_embed = self.compres(x_gnn, edge_index, batch_vec, self.clade_ids_buffer)
        fused, gate = self.base.fusion(cnn_embed, gnn_embed)
        return {
            "class": self.base.class_head(fused),
            "sf": self.base.superfamily_head(fused),
            "gate": gate,
            "fused": fused,
        }
