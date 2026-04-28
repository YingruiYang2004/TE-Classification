"""Idempotent patcher: insert a global per-tag cap on NON-DNA sequences in v4
notebook BEFORE k-mer featurization, so we don't waste compute on the ~130k
sequences kept at natural prevalence.

DNA samples are NOT touched.  The cap uses ``max_per_sf`` (the same parameter
already passed in from MAX_PER_SF in the config cell), grouped by the original
``all_tags`` string.
"""
from __future__ import annotations
import json
from pathlib import Path

NB = Path(__file__).parent / "v4" / "vgp_hybrid_v4.ipynb"
MARKER = "[v4 global non-DNA cap inserted]"

ANCHOR = (
    '    print(f"\\nAfter filtering: {len(all_h)} sequences  '
    '(DNA={n_dna}, non-DNA={n_neg})")\n'
)

INSERT = (
    "\n"
    "    # ---- Global per-tag cap on NON-DNA sequences (compute budget) ----\n"
    "    # " + MARKER + "\n"
    "    # The per-fold MAX_PER_SF cap only subsamples DNA training indices.\n"
    "    # Non-DNA at full natural prevalence (~130k) is too large to\n"
    "    # featurize.  Apply the same cap GLOBALLY to non-DNA sequences\n"
    "    # (per original tag) BEFORE k-mer featurization.  DNA is untouched.\n"
    "    if max_per_sf is not None:\n"
    "        rng_cap = np.random.RandomState(random_state)\n"
    "        nondna_idx = np.where(all_toplevel == 0)[0]\n"
    "        dna_idx = np.where(all_toplevel == 1)[0]\n"
    "        by_tag = {}\n"
    "        for i in nondna_idx:\n"
    "            by_tag.setdefault(all_tags[i], []).append(int(i))\n"
    "        keep_nondna = []\n"
    "        for tag, idxs in by_tag.items():\n"
    "            if len(idxs) > max_per_sf:\n"
    "                idxs = rng_cap.choice(idxs, max_per_sf, replace=False).tolist()\n"
    "            keep_nondna.extend(idxs)\n"
    "        keep = sorted(dna_idx.tolist() + keep_nondna)\n"
    "        all_h = [all_h[i] for i in keep]\n"
    "        all_s = [all_s[i] for i in keep]\n"
    "        all_tags = [all_tags[i] for i in keep]\n"
    "        all_toplevel = all_toplevel[keep]\n"
    "        all_sf = all_sf[keep]\n"
    "        n_dna = int((all_toplevel == 1).sum())\n"
    "        n_neg = int((all_toplevel == 0).sum())\n"
    "        print(f\"After global non-DNA cap (max_per_sf={max_per_sf}): \"\n"
    "              f\"{len(all_h)} sequences  (DNA={n_dna}, non-DNA={n_neg})\")\n"
    "        gc.collect()\n"
)


def patch() -> None:
    nb = json.loads(NB.read_text())
    patched = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if MARKER in src:
            continue
        if ANCHOR not in src:
            continue
        new_src = src.replace(ANCHOR, ANCHOR + INSERT, 1)
        cell["source"] = new_src.splitlines(keepends=True)
        patched += 1
    if patched == 0:
        print(f"[skip] {NB.name}: anchor not found or already patched")
        return
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"[ok] {NB.name}: {patched} cell(s) patched")


if __name__ == "__main__":
    patch()
