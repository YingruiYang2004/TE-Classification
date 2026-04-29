"""Phylogenetically-rebalanced WeightedRandomSampler.

Inverse-frequency weights over `(clade_prefix, sf_with_nondna_sentinel)`
pairs. Targets the failure mode shown in the diagnostic: ~89% of hAT
misses come from amphib/turtle clades that are under-represented in
training.

`clade_prefix` = first letter of VGP species code (the species code is
the part of the FASTA header after the last '-' and before '#'). VGP
prefixes a/b/f/k/m/r/s correspond 1:1 with major vertebrate clades.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from torch.utils.data import WeightedRandomSampler


def species_clade(species_code: str) -> str:
    """Return the single-letter clade prefix of a VGP species code."""
    return species_code[:1] if species_code else "?"


def make_clade_sf_sampler(
    clades: np.ndarray,
    sfs: np.ndarray,
    *,
    binary_labels: np.ndarray | None = None,
    nondna_sentinel: int = 0,
    smoothing: float = 1.0,
    num_samples: int | None = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler over (clade, sf) groups.

    For non-DNA samples (binary_labels == 0) the SF id is collapsed to a
    single bucket so non-DNA balance follows clade-only frequency. DNA
    samples are bucketed by (clade, sf).

    weight_i = 1 / (count(group_i) + smoothing)

    Args:
        clades:       (N,) array of clade prefix strings.
        sfs:          (N,) array of SF ids (used only for DNA samples).
        binary_labels:(N,) optional 0/1; if None, all samples treated as DNA.
        nondna_sentinel: SF bucket id for non-DNA in the (clade, sf) tuple.
        smoothing:    Laplace smoothing on the count to avoid huge weights
                      on rare groups (default 1.0).
        num_samples:  draws per epoch; defaults to N.
        replacement:  passed to WeightedRandomSampler.

    Returns:
        WeightedRandomSampler suitable for DataLoader(sampler=...).
    """
    n = len(clades)
    if binary_labels is None:
        binary_labels = np.ones(n, dtype=np.int64)

    keys: list[tuple[str, int]] = []
    for c, s, b in zip(clades, sfs, binary_labels):
        if int(b) == 1:
            keys.append((str(c), int(s)))
        else:
            keys.append((str(c), int(nondna_sentinel) - 1))  # distinct from any sf id
    counts = Counter(keys)
    weights = np.empty(n, dtype=np.float64)
    for i, k in enumerate(keys):
        weights[i] = 1.0 / (counts[k] + smoothing)

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=int(num_samples) if num_samples else n,
        replacement=bool(replacement),
    )
