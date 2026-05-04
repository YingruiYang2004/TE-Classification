#!/usr/bin/env python3
"""Regenerate mini-benchmark per-superfamily figures from eval_mini_benchmark.json.

Produces:
  - mini_benchmark_perSF_bar.png  (grouped bars: top SFs by support, v4.3 vs v4.3-singlefold, faceted by species)
  - mini_benchmark_perSF_heatmap.png  (two-panel heatmap: SF x species, F1 cells)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
JSON = ROOT / "data_analysis/vgp_model_split_fix/eval_mini_benchmark.json"
OUT_DIR = Path(__file__).resolve().parent

CK_ROT = "v4.3_rotating_epoch40"   # Stage 2 (= thesis "v4.3")
CK_SF  = "v4.3_singlefold_epoch28" # Stage 3 (= thesis "v4.3-singlefold")
SPECIES = [("bTaeGut", "Zebra finch"), ("mOrnAna", "Platypus"), ("rAllMis", "Alligator")]


def load():
    return json.loads(JSON.read_text())["checkpoints"]


def collect_sfs(data, ck):
    sup = {}
    for sp, _ in SPECIES:
        per = data[ck]["per_genome"][sp].get("sf_per", {})
        for sf, m in per.items():
            sup[sf] = sup.get(sf, 0) + m.get("support", 0)
    return sorted(sup.items(), key=lambda x: -x[1])


def get_f1(data, ck, sp, sf):
    per = data[ck]["per_genome"][sp].get("sf_per", {})
    return per.get(sf, {}).get("f1", np.nan), per.get(sf, {}).get("support", 0)


def fig_bar(data, top_n=10):
    merged = {}
    for ck in (CK_ROT, CK_SF):
        for sf, s in collect_sfs(data, ck):
            merged[sf] = merged.get(sf, 0) + s
    sfs = [s for s, _ in sorted(merged.items(), key=lambda x: -x[1])[:top_n]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    x = np.arange(len(sfs))
    w = 0.38
    for ax, (sp_id, sp_name) in zip(axes, SPECIES):
        rot = [get_f1(data, CK_ROT, sp_id, sf)[0] for sf in sfs]
        sing = [get_f1(data, CK_SF, sp_id, sf)[0] for sf in sfs]
        ax.bar(x - w / 2, rot, w, label="v4.3 (Stage 2)", color="#4C72B0")
        ax.bar(x + w / 2, sing, w, label="v4.3-singlefold (Stage 3)", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(sfs, rotation=45, ha="right", fontsize=8)
        ax.set_title(sp_name, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Per-superfamily F$_1$")
    axes[-1].legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle(
        f"Mini-benchmark: per-superfamily F$_1$ on top {top_n} superfamilies (by total support)",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT_DIR / "mini_benchmark_perSF_bar.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_heatmap(data, top_n=14):
    merged = {}
    for ck in (CK_ROT, CK_SF):
        for sf, s in collect_sfs(data, ck):
            merged[sf] = merged.get(sf, 0) + s
    sfs = [s for s, _ in sorted(merged.items(), key=lambda x: -x[1])[:top_n]]

    def matrix(ck):
        M = np.full((len(sfs), len(SPECIES)), np.nan)
        for i, sf in enumerate(sfs):
            for j, (sp_id, _) in enumerate(SPECIES):
                f1, sup = get_f1(data, ck, sp_id, sf)
                if sup > 0:
                    M[i, j] = f1
        return M

    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(sfs) + 1.5))
    titles = ["v4.3 (Stage 2; rotating-CV)", "v4.3-singlefold (Stage 3; disjoint val)"]
    for ax, ck, title in zip(axes, [CK_ROT, CK_SF], titles):
        M = matrix(ck)
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(SPECIES)))
        ax.set_xticklabels([n for _, n in SPECIES], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(sfs)))
        ax.set_yticklabels(sfs, fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                _, sup = get_f1(data, ck, SPECIES[j][0], sfs[i])
                if not np.isnan(v):
                    ax.text(
                        j, i, f"{v:.2f}\n(n={sup})",
                        ha="center", va="center",
                        color="white" if v < 0.55 else "black",
                        fontsize=7,
                    )
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="F$_1$")
    fig.suptitle(
        f"Mini-benchmark: per-superfamily F$_1$ across the three excised amniote genomes (top {top_n} by support)",
        fontsize=10,
    )
    out = OUT_DIR / "mini_benchmark_perSF_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    data = load()
    fig_bar(data)
    fig_heatmap(data)
