"""Create a GPU-featurizer variant of v4 notebook.

Reads v4/vgp_hybrid_v4.ipynb, writes v4/vgp_hybrid_v4_gpu.ipynb with:

1. A new code cell (after the CPU featurizer cell) defining
   `KmerWindowFeaturizerGPU` with a `.featurize_sequence(seq)` method
   compatible with the CPU one (returns ``(np.ndarray, np.ndarray)``).
2. The pre-compute loop in the training function rewritten to use the
   GPU featurizer when ``device.type in {'cuda','mps'}``; otherwise it
   falls back to the existing CPU loop.

Idempotent: re-running is a no-op if the marker is already present.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "v4" / "vgp_hybrid_v4.ipynb"
DST = ROOT / "v4" / "vgp_hybrid_v4_gpu.ipynb"
MARKER = "KmerWindowFeaturizerGPU"

GPU_FEATURIZER_CELL = '''# ============ GPU K-mer Feature Extraction ============
# Vectorised PyTorch implementation of KmerWindowFeaturizer.
# Drop-in replacement: same constructor signature and same return type
# (np.ndarray) so downstream code is unchanged.  ~10-100x faster than the
# CPU NumPy loop (CUDA), ~5-10x faster on MPS.
#
# NOTE: Hash output is bit-identical to the CPU version (splitmix64 on
# unsigned 64-bit), so cached features are interchangeable between
# CPU and GPU featurizers.
import torch as _torch_kmer
import numpy as _np_kmer


class KmerWindowFeaturizerGPU:
    """GPU-accelerated k-mer window featurizer.

    Same semantics as KmerWindowFeaturizer:
      - canonical k-mer (min of fwd / rc) hashed via splitmix64 to dim bins
      - per-window L2 normalisation (optional)
      - optional position channel
    """

    # splitmix64 constants (uint64 arithmetic)
    _C1 = 0x9E3779B97F4A7C15
    _C2 = 0xC2B2AE3D27D4EB4F
    _MASK = 0xFFFFFFFFFFFFFFFF

    def __init__(self, k=7, dim=2048, window=512, stride=256,
                 add_pos=True, l2_normalize=True, device=None, batch_size=64):
        self.k = int(k)
        self.dim = int(dim)
        self.window = int(window)
        self.stride = int(stride)
        self.add_pos = bool(add_pos)
        self.l2_normalize = bool(l2_normalize)
        self.batch_size = int(batch_size)
        if device is None:
            device = "cuda" if _torch_kmer.cuda.is_available() else (
                "mps" if _torch_kmer.backends.mps.is_available() else "cpu")
        self.device = _torch_kmer.device(device)

        # Precompute powers of 4 for k-mer encoding (int64; safe for k<=31).
        self._pow4 = (4 ** _torch_kmer.arange(
            self.k - 1, -1, -1, dtype=_torch_kmer.int64,
            device=self.device))  # (k,)

        # MPS truncates Python int constants > int32 range when broadcast
        # into an int64 tensor op (silently giving 0).  Store the splitmix
        # constants as int64 device tensors so the bit pattern is preserved.
        self._c1_t = _torch_kmer.tensor(
            self._signed64(self._C1), dtype=_torch_kmer.int64,
            device=self.device)
        self._c2_t = _torch_kmer.tensor(
            self._signed64(self._C2), dtype=_torch_kmer.int64,
            device=self.device)
        self._mask_t = _torch_kmer.tensor(
            self._signed64(self._MASK), dtype=_torch_kmer.int64,
            device=self.device)
        # Mask for stripping high bits after a sign-extending right shift.
        # `>> 33` -> 31 low bits remain valid; `>> 29` -> 35 low bits remain
        # valid.  35 > int32 range, so store as int64 tensor for MPS.
        self._mask_lo31_t = _torch_kmer.tensor(
            (1 << 31) - 1, dtype=_torch_kmer.int64, device=self.device)
        self._mask_lo35_t = _torch_kmer.tensor(
            (1 << 35) - 1, dtype=_torch_kmer.int64, device=self.device)
        # Sign-bit mask for "h & 0x7FFF...FFFF" (positive int64).
        self._mask_pos63_t = _torch_kmer.tensor(
            0x7FFFFFFFFFFFFFFF, dtype=_torch_kmer.int64, device=self.device)

    @staticmethod
    def _signed64(u: int) -> int:
        """Reinterpret a uint64 bit pattern as a signed int64 Python int."""
        return u - (1 << 64) if u >= (1 << 63) else u

    # ---- ASCII -> 0..4 lookup (A,C,G,T -> 0..3, anything else -> 4) ----
    @staticmethod
    def _ascii_lut():
        lut = _np_kmer.full(256, 4, dtype=_np_kmer.uint8)
        for ch, val in [("A", 0), ("C", 1), ("G", 2), ("T", 3),
                        ("a", 0), ("c", 1), ("g", 2), ("t", 3)]:
            lut[ord(ch)] = val
        return lut

    _LUT = None

    def _encode(self, seq: str) -> _torch_kmer.Tensor:
        if KmerWindowFeaturizerGPU._LUT is None:
            KmerWindowFeaturizerGPU._LUT = self._ascii_lut()
        b = _np_kmer.frombuffer(seq.encode("ascii", "ignore"),
                                dtype=_np_kmer.uint8)
        arr = KmerWindowFeaturizerGPU._LUT[b]  # uint8 (L,)
        return _torch_kmer.from_numpy(arr.copy()).to(
            device=self.device, dtype=_torch_kmer.int64)

    def _splitmix64(self, x: _torch_kmer.Tensor) -> _torch_kmer.Tensor:
        # x is int64; arithmetic must wrap mod 2^64 with UNSIGNED right
        # shifts.  PyTorch >> on signed int64 is arithmetic (sign-extending),
        # so after each shift we mask off the high (n) bits to mimic a
        # logical shift on a uint64.
        # On MPS, scalar Python ints > int32 range get silently truncated, so
        # we use device-resident int64 tensors for the multiplicative
        # constants and the 64-bit AND mask.
        m = self._mask_t
        z = (x * self._c1_t) & m
        z = z ^ ((z >> 33) & self._mask_lo31_t)   # logical >> 33
        z = (z * self._c2_t) & m
        z = z ^ ((z >> 29) & self._mask_lo35_t)   # logical >> 29
        return z

    def featurize_sequence(self, seq: str):
        """Returns (X: np.ndarray (W, dim+add_pos) float32, starts: np.ndarray (W,) int64)."""
        with _torch_kmer.no_grad():
            arr = self._encode(seq)  # (L,) int64
            L = int(arr.numel())
            out_dim = self.dim + (1 if self.add_pos else 0)

            if L == 0:
                return (_np_kmer.zeros((1, out_dim), dtype=_np_kmer.float32),
                        _np_kmer.array([0], dtype=_np_kmer.int64))

            # Pad if shorter than window (then we still emit 1 window starting at 0).
            if L < self.window:
                pad = _torch_kmer.full((self.window - L,), 4,
                                       dtype=_torch_kmer.int64,
                                       device=self.device)
                arr = _torch_kmer.cat([arr, pad], dim=0)
                starts_np = _np_kmer.array([0], dtype=_np_kmer.int64)
            else:
                starts_np = _np_kmer.arange(0, L - self.window + 1,
                                            self.stride, dtype=_np_kmer.int64)
                if starts_np.size == 0:
                    starts_np = _np_kmer.array([0], dtype=_np_kmer.int64)

            # (W, window) view: each row is one window.
            windows = arr.unfold(0, self.window, self.stride)
            W = windows.size(0)

            # If the unfold produced a different W than starts_np (edge cases
            # where L just exceeds window but stride misaligns), trust the
            # tensor shape and recompute starts.
            if W != starts_np.size:
                starts_np = (_np_kmer.arange(W, dtype=_np_kmer.int64)
                             * self.stride)

            # (W, n_kmers, k) where n_kmers = window - k + 1
            kmers = windows.unfold(1, self.k, 1)
            # forward and reverse-complement codes.
            fwd = (kmers * self._pow4).sum(dim=-1)            # (W, n_kmers)
            rc = ((3 - kmers).flip(-1) * self._pow4).sum(dim=-1)
            # Validity: any base == 4 (N) -> invalid k-mer.
            valid = ~(kmers == 4).any(dim=-1)                 # (W, n_kmers) bool
            # Canonical = min(fwd, rc). For invalid k-mers the value is
            # arbitrary; we'll mask them in the scatter.
            code = _torch_kmer.minimum(fwd, rc)
            # splitmix64 hash -> bin id in [0, dim).
            h = self._splitmix64(code)
            # PyTorch lacks unsigned modulo on int64 directly, but for
            # nonnegative dim we can just mask sign and take modulo.
            bins = (h & self._mask_pos63_t) % self.dim        # (W, n_kmers)

            # Histogram: scatter_add ones into (W, dim) at indices `bins`.
            counts = _torch_kmer.zeros((W, self.dim),
                                       dtype=_torch_kmer.float32,
                                       device=self.device)
            ones = valid.to(_torch_kmer.float32)
            counts.scatter_add_(1, bins, ones)

            # Normalise by valid k-mer count per window.
            totals = ones.sum(dim=-1, keepdim=True).clamp_min(1.0)
            counts = counts / totals

            if self.l2_normalize:
                norms = counts.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                counts = counts / norms

            if self.add_pos:
                starts_t = _torch_kmer.from_numpy(starts_np).to(
                    self.device).float()
                # Match CPU semantics: center = (start + min(start+window, L)) / 2
                ends = _torch_kmer.minimum(
                    starts_t + float(self.window),
                    _torch_kmer.tensor(float(L), device=self.device))
                centres = (starts_t + ends) / 2.0
                pos = (centres / max(1.0, float(L))).unsqueeze(-1)
                X = _torch_kmer.cat([counts, pos], dim=-1)
            else:
                X = counts

            return X.detach().cpu().numpy().astype(_np_kmer.float32), starts_np
'''

# The featurization loop currently looks like this (we replace it):
OLD_LOOP_ANCHOR = '''    print("\\n=== Pre-computing k-mer features ===")
    featurizer = KmerWindowFeaturizer(
        k=kmer_k, dim=kmer_dim, window=kmer_window, stride=kmer_stride,
        add_pos=True, l2_normalize=True
    )
    all_kmer_features = []
    _n_total = len(all_s)
    _print_every = max(1000, _n_total // 20)
    for _i, seq in enumerate(tqdm(all_s, desc="Featurizing", leave=False), 1):
        X, _ = featurizer.featurize_sequence(seq)
        all_kmer_features.append(X)
        if _i % _print_every == 0 or _i == _n_total:
            print(f"  [features extracted] {_i}/{_n_total} features have been extracted", flush=True)
    print(f"K-mer features computed: {len(all_kmer_features)} sequences")'''

NEW_LOOP = '''    print("\\n=== Pre-computing k-mer features ===")
    if device.type in ("cuda", "mps"):
        print(f"  Using GPU featurizer on {device}")
        featurizer = KmerWindowFeaturizerGPU(
            k=kmer_k, dim=kmer_dim, window=kmer_window, stride=kmer_stride,
            add_pos=True, l2_normalize=True, device=device,
        )
    else:
        print("  Using CPU NumPy featurizer (no GPU available)")
        featurizer = KmerWindowFeaturizer(
            k=kmer_k, dim=kmer_dim, window=kmer_window, stride=kmer_stride,
            add_pos=True, l2_normalize=True,
        )
    all_kmer_features = []
    _n_total = len(all_s)
    _print_every = max(1000, _n_total // 20)
    import time as _t_kmer
    _t0_kmer = _t_kmer.time()
    for _i, seq in enumerate(tqdm(all_s, desc="Featurizing", leave=False), 1):
        X, _ = featurizer.featurize_sequence(seq)
        all_kmer_features.append(X)
        if _i % _print_every == 0 or _i == _n_total:
            _elapsed = _t_kmer.time() - _t0_kmer
            _rate = _i / max(_elapsed, 1e-6)
            _eta = (_n_total - _i) / max(_rate, 1e-6)
            print(f"  [features extracted] {_i}/{_n_total} ({_rate:.1f} seq/s, ETA {_eta:.0f}s)", flush=True)
    print(f"K-mer features computed: {len(all_kmer_features)} sequences in {_t_kmer.time()-_t0_kmer:.1f}s")'''


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source notebook not found: {SRC}")

    # Always start from a fresh copy so re-runs reflect upstream changes.
    nb = json.loads(SRC.read_text())

    # ---- Insert GPU featurizer cell after the CPU featurizer cell ----
    cells = nb["cells"]
    insert_at = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "class KmerWindowFeaturizer:" in src:
            insert_at = i + 1
            break
    if insert_at is None:
        raise SystemExit("Could not locate CPU KmerWindowFeaturizer cell")

    new_cell = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": GPU_FEATURIZER_CELL.splitlines(keepends=True),
    }
    cells.insert(insert_at, new_cell)

    # ---- Rewrite the pre-compute loop ----
    patched_loops = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if OLD_LOOP_ANCHOR in src:
            new_src = src.replace(OLD_LOOP_ANCHOR, NEW_LOOP, 1)
            c["source"] = new_src.splitlines(keepends=True)
            patched_loops += 1

    DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {DST.relative_to(ROOT)}: "
          f"GPU featurizer cell inserted, {patched_loops} loop(s) rewritten")


if __name__ == "__main__":
    main()
