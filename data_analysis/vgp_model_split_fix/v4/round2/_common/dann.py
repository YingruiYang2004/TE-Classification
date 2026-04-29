"""DANN gradient-reversal + species classifier head.

Used in all three Round-2 tracks. Plugged onto the *fused* embedding
(post-fusion, pre-SF-head) so the species-invariance pressure shapes
the representation that the SF head reads.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Identity in forward, multiplies grad by -lambda_ in backward."""
    return _GradReverse.apply(x, lambda_)


class SpeciesHead(nn.Module):
    """2-layer MLP that classifies species from the fused embedding.

    The forward applies grad-reverse with `lambda_` (set externally by the
    training loop's schedule) so backprop pushes the *upstream* representation
    toward species-invariance.
    """

    def __init__(self, in_dim: int, n_species: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lambda_ = 0.0
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_species),
        )

    def set_lambda(self, lam: float) -> None:
        self.lambda_ = float(lam)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(grad_reverse(fused, self.lambda_))


def lambda_warmup(epoch: int, max_lambda: float = 0.5, warmup_epochs: int = 1) -> float:
    """Linear warmup from 0 to `max_lambda` over `warmup_epochs` epochs.

    `epoch` is 1-indexed. After warmup, returns `max_lambda` constant.
    """
    if warmup_epochs <= 0:
        return float(max_lambda)
    frac = min(1.0, max(0.0, epoch / warmup_epochs))
    return float(max_lambda) * frac
