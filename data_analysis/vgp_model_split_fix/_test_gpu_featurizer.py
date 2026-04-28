"""Correctness + speed test for the GPU k-mer featurizer.

Extracts the CPU class and the new GPU class from the v4 notebooks,
runs both on a few real sequences, and reports max-abs-diff + per-seq
time on MPS (or CPU fallback).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
GPU_NB = ROOT / "v4" / "vgp_hybrid_v4_gpu.ipynb"


def _exec_cells(nb_path: Path, predicates):
    """Exec cells whose joined source contains any of `predicates` strings."""
    nb = json.loads(nb_path.read_text())
    g = {"__name__": "__main__"}
    sys.modules.setdefault("__main__", type(sys)("__main__"))
    exec("import numpy as np\nimport torch\nfrom dataclasses import dataclass\n"
         "from typing import Tuple, List, Optional\n", g)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if any(p in src for p in predicates):
            exec(src, g)
    return g


def main() -> None:
    g = _exec_cells(GPU_NB, [
        "_ASCII_MAP", "class KmerWindowFeaturizer:",
        "class KmerWindowFeaturizerGPU",
    ])
    Cpu = g["KmerWindowFeaturizer"]
    Gpu = g["KmerWindowFeaturizerGPU"]

    rng = np.random.default_rng(0)
    bases = np.array(list("ACGT"))
    seqs = []
    for L in [500, 2000, 8000, 20000, 20000]:
        seq = "".join(rng.choice(bases, size=L).tolist())
        # Sprinkle some N's
        if L > 100:
            for pos in rng.integers(0, L, size=L // 200):
                seq = seq[:pos] + "N" + seq[pos + 1:]
        seqs.append(seq)

    cpu = Cpu(k=7, dim=2048, window=512, stride=256,
              add_pos=True, l2_normalize=True)

    # GPU device autoselect
    import torch
    dev = ("cuda" if torch.cuda.is_available()
           else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"GPU featurizer device: {dev}")
    gpu = Gpu(k=7, dim=2048, window=512, stride=256,
              add_pos=True, l2_normalize=True, device=dev)

    print("\n--- Correctness ---")
    for i, s in enumerate(seqs):
        Xc, sc = cpu.featurize_sequence(s)
        Xg, sg = gpu.featurize_sequence(s)
        max_abs = float(np.abs(Xc - Xg).max()) if Xc.shape == Xg.shape else float("inf")
        # Decompose: histogram (cols [:-1]) vs position (col [-1])
        d_hist = float(np.abs(Xc[:, :-1] - Xg[:, :-1]).max())
        d_pos = float(np.abs(Xc[:, -1] - Xg[:, -1]).max())
        # Where the largest hist diff is
        flat = np.abs(Xc[:, :-1] - Xg[:, :-1])
        if flat.size:
            wi, bi = np.unravel_index(flat.argmax(), flat.shape)
            print(f"  seq{i} L={len(s):5d} shape={Xc.shape}  max|Δ|hist={d_hist:.3e} "
                  f"max|Δ|pos={d_pos:.3e}  worst_cell=(w{wi},b{bi}) "
                  f"cpu={Xc[wi, bi]:.4f} gpu={Xg[wi, bi]:.4f}")

    # Benchmark on N seqs of length 20000
    N = 200
    bench_seqs = ["".join(rng.choice(bases, size=20000).tolist()) for _ in range(N)]

    print(f"\n--- Speed (N={N} seqs of length 20k) ---")
    t0 = time.time()
    for s in bench_seqs:
        cpu.featurize_sequence(s)
    t_cpu = time.time() - t0
    print(f"  CPU NumPy: {t_cpu:.2f}s ({N/t_cpu:.1f} seq/s)")

    # Warm-up GPU
    gpu.featurize_sequence(bench_seqs[0])
    t0 = time.time()
    for s in bench_seqs:
        gpu.featurize_sequence(s)
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()
    t_gpu = time.time() - t0
    print(f"  GPU ({dev}): {t_gpu:.2f}s ({N/t_gpu:.1f} seq/s)  speedup={t_cpu/t_gpu:.1f}x")


if __name__ == "__main__":
    main()
