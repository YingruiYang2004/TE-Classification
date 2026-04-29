"""Group-DRO: weight per-sample losses by adversarially-tracked group means.

Sagawa et al. 2020. Light implementation: maintain a per-group EMA of
the mean loss; the group with the largest EMA gets the largest weight
via a softmax with adversarial step size `eta`. The reported loss is a
weighted mean over groups.

Used here to up-weight worst-clade losses during SF training.
"""
from __future__ import annotations

import torch


class GroupDROLoss:
    """Adversarial reweighting of per-sample losses across groups.

    Args:
        n_groups: number of groups (e.g. number of clade prefixes).
        eta: adversarial step size on the log-weight space.
        device: device for the running adv weights tensor.

    Use:
        gdro = GroupDROLoss(n_groups=7, eta=0.05, device=device)
        for batch in loader:
            per_sample_loss = ce_loss(...)  # shape (B,), reduction='none'
            group_ids = ...                 # shape (B,) longs in [0, n_groups)
            loss = gdro.weighted_mean(per_sample_loss, group_ids)
            loss.backward()
    """

    def __init__(self, n_groups: int, eta: float = 0.05, device=None):
        self.n_groups = int(n_groups)
        self.eta = float(eta)
        self.device = device
        # Adversarial group weights live on the log scale; init uniform.
        self.adv = torch.zeros(self.n_groups, device=device)

    def state_dict(self) -> dict:
        return {"adv": self.adv.detach().cpu(), "eta": self.eta, "n_groups": self.n_groups}

    def weighted_mean(self, per_sample_loss: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """Compute Group-DRO weighted mean of per-sample loss.

        `per_sample_loss`: (B,) float tensor, no reduction.
        `group_ids`:       (B,) long tensor in [0, n_groups). Samples whose
                           group has zero count in the batch are silently
                           skipped from the adv update *for this step*.
        """
        if per_sample_loss.numel() == 0:
            return per_sample_loss.sum()  # zero, on correct device

        device = per_sample_loss.device
        if self.adv.device != device:
            self.adv = self.adv.to(device)

        # Per-group mean loss in this batch.
        group_loss = torch.zeros(self.n_groups, device=device)
        group_cnt = torch.zeros(self.n_groups, device=device)
        group_loss.index_add_(0, group_ids, per_sample_loss.detach())
        ones = torch.ones_like(per_sample_loss)
        group_cnt.index_add_(0, group_ids, ones)
        present = group_cnt > 0
        # Avoid div-by-zero; absent groups will contribute 0 to adv update.
        safe_cnt = group_cnt.clamp_min(1.0)
        mean_per_group = group_loss / safe_cnt

        # Adversarial step on log-weights for present groups.
        with torch.no_grad():
            self.adv = self.adv + self.eta * mean_per_group * present.float()

        # Softmax over groups -> distribution.
        weights = torch.softmax(self.adv, dim=0)

        # Forward-side: weighted sum of *batch* group means using adv weights.
        # Absent groups contribute 0 (their mean is 0 by construction above).
        # Renormalise weights over present groups so the loss has stable scale.
        present_w = weights * present.float()
        denom = present_w.sum().clamp_min(1e-12)
        present_w = present_w / denom
        # Recompute differentiable mean_per_group from per_sample_loss without detach.
        diff_group_loss = torch.zeros(self.n_groups, device=device)
        diff_group_loss.index_add_(0, group_ids, per_sample_loss)
        diff_mean = diff_group_loss / safe_cnt
        return (present_w * diff_mean).sum()
