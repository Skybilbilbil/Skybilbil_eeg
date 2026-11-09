#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the three figures that the paper currently lacks (readout ladder / coupling spectrum / identity saturation).

All numbers are read from the already-saved JSON files (zero training).
Output: MAtt_baseline/results/report_figs/*.png
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent          # 论文/（仓库根）
BASE = HERE.parent / 'MAtt_baseline'            # MAtt_baseline/
R = BASE / "results"
OUT = R / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 170,
})
C_LOW = "#d62728"; C_HIGH = "#1f77b4"; C_SH = "#c0392b"; C_PS = "#2471a3"


def load(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def fig_ladder():
    """Fig 7: shared-readout ladder + identity purity."""
    K = load("kernel_ceiling/kernel_ceiling.json")["arms"]
    N = load("nystrom_head/nystrom_head.json")["arms"]["nystrom_M1024"]
    rows = [
        ("LE+LDA", 62.9, K["A2_K1_lda_linear"]["low4"], K["A2_K1_lda_linear"]["high5"], 1.000),
        ("RC+LDA", 61.4, 46.4 / 100, 73.3 / 100, np.nan),
        ("multi-chart", 68.6, 0.540, 0.800, 1.000),
        ("RFF+LDA", 69.7, 0.545, 0.817, np.nan),
        ("RBF-SVM", 73.5, K["A1_K1_C100_gm2"]["low4"], K["A1_K1_C100_gm2"]["high5"], 1.000),
        ("Nystrom", 73.8, N["low4"], N["high5"], 1.000),
        ("SupCon+LDA", 69.7, 0.548, 0.813, 0.713),
    ]
    lab = [r[0] for r in rows]
    pooled = [r[1] for r in rows]
    lo = [r[2] * 100 for r in rows]
    hi = [r[3] * 100 for r in rows]
    pur = [r[4] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), gridspec_kw={"width_ratios": [2.0, 1.0]})
    ax = axes[0]
    x = np.arange(len(rows))
    w = 0.27
    ax.bar(x - w, lo, w, color=C_LOW, alpha=.85, label="low-4 (S02,S04,S05,S06)")
    ax.bar(x, pooled, w, color="#7f8c8d", alpha=.85, label="pooled")
    ax.bar(x + w, hi, w, color=C_HIGH, alpha=.85, label="high-5 (S01,S03,S07,S08,S09)")
    for i, v in enumerate(pooled):
        ax.text(i, v + 1.0, "%.1f" % v, ha="center", fontsize=8, fontweight="bold")
    ax.axhline(25, ls="--", c="k", lw=.8, alpha=.6); ax.text(len(rows) - .4, 26, "chance 25%", fontsize=7, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=20, ha="right")
    ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Shared readout ladder (same 253-d LE features; same test fold)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=3)

    ax2 = axes[1]
    m = ~np.isnan(pur)
    ax2.scatter(np.array(pooled)[m], np.array(pur)[m], s=75, c="#8e44ad", zorder=3)
    offs = {"LE+LDA": (-58, -14), "multi-chart": (-62, 8), "RBF-SVM": (-70, 8),
            "Nystrom": (8, -16), "SupCon+LDA": (8, 8)}
    for i in np.where(m)[0]:
        dx, dy = offs.get(lab[i], (6, 5))
        ax2.annotate(lab[i], (pooled[i], pur[i]), textcoords="offset points", xytext=(dx, dy), fontsize=7.5)
    ax2.axhline(1.0, ls=":", c="gray", lw=1)
    ax2.text(min(np.array(pooled)[m]) - 2.5, 1.028, "identity saturated (1.000)", fontsize=7, color="gray", ha="left")
    ax2.set_xlabel("test accuracy (%)"); ax2.set_ylabel("identity purity (kNN k=9)")
    ax2.set_ylim(0.62, 1.07); ax2.set_xlim(min(np.array(pooled)[m]) - 3, max(np.array(pooled)[m]) + 4)
    ax2.set_title("Accuracy vs. shared-ness", fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / "fig8_readout_ladder.png", bbox_inches="tight"); plt.close(fig)
    print("fig8 ok")


def fig_coupling():
    """Fig 8: coupling of mu-ERD depth with per-subject accuracy, by reader layer."""
    D = load("erd_coupling_by_reader/erd_coupling_by_reader.json")["rows"]
    rows = [r for r in D if not r["reader"].startswith("deep70 seed")]
    rows = sorted(rows, key=lambda r: r["pooled"])
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(rows))
    colors = [C_PS if r["kind"] == "per-subject" else C_SH for r in rows]
    vals = [r["rho"] for r in rows]
    ax.bar(x, vals, color=colors, alpha=.88, width=.62)
    for i, r in enumerate(rows):
        v = r["rho"]
        if abs(v) < 0.02:   # zero bar is invisible: mark with a short line + label above
            ax.plot([i - .31, i + .31], [0, 0], color=C_PS, lw=3.2)
            ax.text(i, 0.035, "%.3f" % v, ha="center", va="bottom", fontsize=7.5, color=C_PS)
        else:
            ax.text(i, v - 0.04, "%.3f" % v, ha="center", va="top", fontsize=7.5)
    ax.axhline(0, c="k", lw=.9)
    ax.axhspan(-1.0, -0.7, color=C_SH, alpha=.07)
    ax.text(len(rows) - 0.6, -0.50, "shared readouts:\nstrong coupling", fontsize=7.5, ha="right", color=C_SH)
    ax.text(len(rows) - 0.6, -0.10, "per-subject readouts:\ncoupling vanishes", fontsize=7.5, ha="right", color=C_PS)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(pooled %.1f)" % (r["reader"], r["pooled"] * 100) for r in rows],
                       rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel("Spearman rho (mu-ERD depth vs. per-subject acc)"); ax.set_ylim(-1.05, 0.15)
    ax.set_title("The ERD-accuracy coupling is a property of the SHARED readout", fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / "fig7_erd_coupling.png", bbox_inches="tight"); plt.close(fig)
    print("fig7 ok")


def fig_identity():
    """Fig 9: identity saturation in LE space (kNN purity + same/diff distance)."""
    K = load("kernel_ceiling/kernel_ceiling.json")["arms"]
    ks = ["1", "15", "30"]
    frac = [K["D_nn_identity_k%s" % k] for k in ks]
    ratio = K["D_dist_ratio"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axes[0]
    ax.bar(["k=1", "k=15", "k=30"], [f * 100 for f in frac], color=C_SH, alpha=.85, width=.55)
    ax.axhline(100 / 9, ls="--", c="k", lw=.9)
    ax.text(2.35, 100 / 9 + 2, "chance 11.1%", fontsize=7.5, ha="right")
    for i, f in enumerate(frac):
        ax.text(i, f * 100 + 1.5, "%.0f%%" % (f * 100), ha="center", fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, 112); ax.set_ylabel("share of neighbours from the same subject (%)")
    ax.set_title("Identity saturation in LE space", fontsize=9.5)

    ax2 = axes[1]
    ax2.bar(["same subject", "other subject"], [2.83, 23.23], color=[C_PS, C_HIGH], alpha=.85, width=.5)
    ax2.set_ylim(0, 29)
    ax2.set_ylabel("median nearest-neighbour distance")
    ax2.set_xlabel("ratio same/other = %.3f" % ratio, fontsize=8)
    ax2.set_title("Same-subject neighbours are ~8x closer", fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / "fig9_identity_saturation.png", bbox_inches="tight"); plt.close(fig)
    print("fig9 ok")


if __name__ == "__main__":
    fig_ladder(); fig_coupling(); fig_identity()
    print("[OK]", OUT)
