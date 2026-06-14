"""Quantitative diagnostic figure for the four SAM4Tun structural constraints.

Replaces the cartoonish error_mechanism.pdf with four data-derived panels
(Opus-4.6 m+s+k, 30 tunnels), each tied to one constraint:
  (a) C1 non-uniform density  -> per-ring density vs per-ring block accuracy
  (b) C2 moving K-anchor       -> per-ring K-offset vs accuracy (aligned vs rotated)
  (c) C3 fixed-offset template  -> recall vs sector-distance from K (K-aligned rings)
  (d) C4 hard-coded walk dir.   -> rotation / flip / other ring counts

Inputs (produced by analyze_constraints.py --all):
  methods/reviews/v2/analysis/per_ring.csv
  methods/reviews/v2/analysis/aggregate.json
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = "methods/reviews/v2/analysis"
OUT = "methods/reviews/v2/figs/constraint_diagnostics.pdf"

C_REG = "#2166ac"
C_CPX = "#b2182b"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})


def main():
    pr = pd.read_csv(f"{ANALYSIS}/per_ring.csv")
    with open(f"{ANALYSIS}/aggregate.json") as f:
        agg = json.load(f)

    reg = pr[pr["category"] == "regular"]
    cpx = pr[pr["category"] == "complex"]

    fig, axes = plt.subplots(1, 5, figsize=(18.5, 3.6), constrained_layout=True)

    # ---- (a) C1: density vs accuracy ----
    ax = axes[0]
    ax.scatter(reg["density"], reg["acc"], s=14, c=C_REG, alpha=0.6,
               edgecolors="none", label="Regular")
    ax.scatter(cpx["density"], cpx["acc"], s=14, c=C_CPX, alpha=0.6,
               edgecolors="none", label="Complex")
    ax.set_xscale("log")
    ax.set_xlabel("per-ring point density (log)")
    ax.set_ylabel("per-ring block accuracy")
    ax.set_title("(a) C1 Non-uniform density")
    ax.text(0.04, 0.93,
            f"corr (reg)= +{agg['regular']['c1_corr_dens_acc']:.2f}\n"
            f"count max/min: {agg['regular']['c1_count_ratio']:.0f}x / "
            f"{agg['complex']['c1_count_ratio']:.0f}x",
            transform=ax.transAxes, va="top", fontsize=7.5)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.set_ylim(-0.03, 1.03)

    # ---- (b) C2: K-offset vs accuracy ----
    ax = axes[1]
    ax.scatter(reg["k_off_sectors"], reg["acc"], s=14, c=C_REG, alpha=0.6,
               edgecolors="none", label="Regular")
    ax.scatter(cpx["k_off_sectors"], cpx["acc"], s=14, c=C_CPX, alpha=0.6,
               edgecolors="none", label="Complex")
    ax.axvline(0.5, color="#444444", ls="--", lw=1)
    ax.text(0.52, 0.02, "K mislocated \u2192", fontsize=7.5, color="#444444")
    ax.set_xlabel("per-ring K offset (sector units)")
    ax.set_ylabel("per-ring block accuracy")
    ax.set_title("(b) C2 Moving K-anchor")
    ax.text(0.96, 0.95,
            f"mislocated: {agg['regular']['c2_misloc_frac']*100:.0f}% reg / "
            f"{agg['complex']['c2_misloc_frac']*100:.0f}% cpx",
            transform=ax.transAxes, va="top", ha="right", fontsize=7.5)
    ax.set_ylim(-0.03, 1.03)

    # ---- (c) C3: recall vs distance from K ----
    ax = axes[2]
    dists = [0, 1, 2, 3]
    rr = [agg["regular"]["c3_recall_by_dist"].get(str(d), np.nan) for d in dists]
    rc = [agg["complex"]["c3_recall_by_dist"].get(str(d), np.nan) for d in dists]
    x = np.arange(len(dists))
    w = 0.38
    ax.bar(x - w / 2, rr, w, color=C_REG, label="Regular")
    ax.bar(x + w / 2, rc, w, color=C_CPX, label="Complex")
    ax.set_xticks(x)
    ax.set_xticklabels(["K", "1", "2", "3 (far)"])
    ax.set_xlabel("sector distance from K (K-aligned rings)")
    ax.set_ylabel("class recall")
    ax.set_title("(c) C3 Fixed-offset template")
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    ax.set_ylim(0, 1.0)

    # ---- (d) C4: ordering outcome counts ----
    ax = axes[3]
    cats = ["Regular", "Complex"]
    rot = [agg["regular"]["c4_rot"], agg["complex"]["c4_rot"]]
    flip = [agg["regular"]["c4_flip"], agg["complex"]["c4_flip"]]
    other = [agg["regular"]["c4_other"], agg["complex"]["c4_other"]]
    xb = np.arange(len(cats))
    ax.bar(xb, rot, 0.55, color="#cccccc", label="rotation-only")
    ax.bar(xb, flip, 0.55, bottom=rot, color=C_CPX, label="direction flip")
    ax.bar(xb, other, 0.55, bottom=np.array(rot) + np.array(flip),
           color="#f4a300", label="other / mixed")
    ax.set_xticks(xb)
    ax.set_xticklabels(cats)
    ax.set_ylabel("rings")
    ax.set_title("(d) C4 Walk direction")
    ax.text(0.5, 0.5,
            f"flip-ring acc \u2248 {agg['regular']['c4_flip_acc']:.2f}",
            transform=ax.transAxes, ha="center", fontsize=7.5)
    ax.legend(loc="upper center", fontsize=7.5, frameon=False, ncol=1)

    # ---- (e) FP/FN/swap composition (baseline vs m+s+k) ----
    ax = axes[4]
    cols = {"correct": "#cccccc", "fn": "#2166ac", "fp": "#b2182b",
            "swap": "#f4a300", "unmapped": "#efe8d6"}
    labels = ["Reg\nbase", "Reg\nm+s+k", "Cpx\nbase", "Cpx\nm+s+k"]
    keys = [("regular", "base_"), ("regular", "msk_"),
            ("complex", "base_"), ("complex", "msk_")]
    xb = np.arange(len(keys))
    bottom = np.zeros(len(keys))
    for comp, color in cols.items():
        vals = np.array([agg[cat][f"{pre}{comp}"] * 100 for cat, pre in keys])
        ax.bar(xb, vals, 0.6, bottom=bottom, color=color,
               label={"fn": "FN", "fp": "FP", "swap": "swap"}.get(comp, comp))
        bottom += vals
    ax.set_xticks(xb)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("% of GT points")
    ax.set_title("(e) Error composition")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower center", fontsize=6.8, frameon=False, ncol=3,
              bbox_to_anchor=(0.5, -0.02))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
