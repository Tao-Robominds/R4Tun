#!/usr/bin/env python3
"""
Generate rules-comparison bar chart: 5 conditions (sam4tun, rules, m, m+s, m+s+k)
sorted by mean mIoU within each panel (Regular, Complex).
LLM conditions use 3-model averages; rules uses n=30 with 3 failures as 0.

Output: methods/papers/figs/rules_comparison.pdf
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA = {
    "Regular": {
        "sam4tun":  0.291,
        "rules":    0.286,
        "m":        0.297,
        "m+s":      0.524,
        "m+s+k":    0.513,
    },
    "Complex": {
        "sam4tun":  0.042,
        "rules":    0.137,
        "m":        0.082,
        "m+s":      0.141,
        "m+s+k":    0.184,
    },
}

COUNTS = {"Regular": 13, "Complex": 17}

COLORS = {
    "sam4tun": "#888888",
    "rules":   "#DAA520",
    "m":       "#7BAFD4",
    "m+s":     "#4A90D9",
    "m+s+k":   "#2C5F9E",
}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)

    for ax_idx, (fam, values) in enumerate(DATA.items()):
        ax = axes[ax_idx]
        sorted_items = sorted(values.items(), key=lambda kv: kv[1])
        labels = [k for k, _ in sorted_items]
        vals = [v for _, v in sorted_items]
        colors = [COLORS[k] for k in labels]

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, vals, color=colors, edgecolor="white",
                       linewidth=0.6, height=0.6)

        for bar, v in zip(bars, vals):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", ha="left", fontsize=10,
                    color="#333333", fontweight="medium")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Mean mIoU (3-model avg)", fontsize=11)
        n = COUNTS[fam]
        ax.set_title(f"{fam} tunnels ($n = {n}$)", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        x_max = max(vals) * 1.22
        ax.set_xlim(0, x_max)

    plt.tight_layout()
    out_path = REPO_ROOT / "methods" / "papers" / "figs" / "rules_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
