"""Intuitive schematic of the class-confusion mechanism and its failure modes.

2x2 layout:
  (a) how blocks are labelled: detect K, then place A/B by FIXED offsets
  (b) ring-to-ring point density (real data) -> one preprocessing setting cannot
      fit both sparse and dense rings -> unreliable K/boundary detection
  (c) ring-to-ring K circumferential position (real data) -> the anchor moves,
      so K must be re-found every ring (stagger; worst on complex tunnels)
  (d) resulting failure modes: K mis-detection rotates the whole ring; the fixed
      sector drifts for far blocks; reversed handedness mirrors the labels
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = "/media/boringtao/Expansion/R4Tun-AIC!/methods/papers/figs/error_mechanism.pdf"
F_REG = "/media/boringtao/Expansion/R4Tun-AIC!/data/sam4tun/1-1/final.csv"
F_CPX = "/media/boringtao/Expansion/R4Tun-AIC!/data/sam4tun/4-1/final.csv"

COL = {
    "K": "#f4a300", "B1": "#4575b4", "B2": "#4575b4",
    "A1": "#5ab4ac", "A2": "#2c7a73", "A3": "#5ab4ac",
}
# one ring, bottom->top by circumferential angle; (label, height)
RING = [("A1", 3), ("B1", 3), ("K", 1), ("B2", 3), ("A3", 3), ("A2", 3)]


def draw_strip(ax, x0, width, blocks, labels=None, edge="black", alpha=1.0,
               y0=0.0, fontsize=8, mark_k=True, wrong=None):
    """Draw a vertical stack of labelled blocks. `labels` overrides displayed text."""
    y = y0
    wrong = wrong or set()
    for i, (name, h) in enumerate(blocks):
        disp = name if labels is None else labels[i]
        fc = COL.get(name, "#dddddd")
        rect = Rectangle((x0, y), width, h, facecolor=fc, edgecolor=edge,
                         linewidth=1.0, alpha=alpha)
        ax.add_patch(rect)
        txt = disp
        tc = "white" if name in ("K", "A2") else "black"
        if disp in wrong:
            tc = "red"
            txt = disp + "\u2717"
        ax.text(x0 + width / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fontsize, color=tc, fontweight="bold")
        if mark_k and name == "K":
            ax.annotate("K detected", xy=(x0, y + h / 2),
                        xytext=(x0 - 1.7 * width, y + h / 2),
                        ha="right", va="center", fontsize=fontsize,
                        color="#b35900",
                        arrowprops=dict(arrowstyle="->", color="#b35900"))
        y += h
    return y


def panel_a(ax):
    ax.set_title("(a) Labelling rule: detect K, place A/B by fixed offsets",
                 fontsize=10, loc="left")
    top = draw_strip(ax, 0.0, 1.4, RING, fontsize=8)
    # fixed-offset arrows from K
    k_center = 3 + 3 + 0.5
    for dy, lab in [(3, "+1 sector"), (6, "+2"), (9, "+3"),
                    (-2, "-1 sector"), (-5, "-2")]:
        ax.add_patch(FancyArrowPatch((1.7, k_center), (1.7, k_center + dy),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#888", lw=1))
    ax.text(2.0, k_center + 4.5, "fixed\nA/B sector\nheight",
            fontsize=8, va="center", color="#555")
    ax.text(2.0, k_center - 3.5, "(same for\nevery ring)",
            fontsize=8, va="center", color="#555")
    ax.set_xlim(-2.6, 4.2)
    ax.set_ylim(-0.5, 16.5)
    ax.set_ylabel("circumferential angle \u03b8")
    ax.set_xticks([])
    ax.set_yticks([])


def panel_b(ax):
    ax.set_title("(b) Ring-to-ring point density varies 17-40x in one tunnel",
                 fontsize=10, loc="left")
    for path, lab, c in [(F_REG, "regular (1-1)", "#1f77b4"),
                         (F_CPX, "complex (4-1)", "#d62728")]:
        df = pd.read_csv(path, usecols=["ring"])
        df = df[np.isfinite(df["ring"])]
        vc = df["ring"].value_counts().sort_index()
        x = np.linspace(0, 1, len(vc))
        ax.plot(x, vc.values / 1000.0, "-o", color=c, label=lab, ms=4)
    ax.set_xlabel("ring index (normalised, near\u2192far from scanner)")
    ax.set_ylabel("points per ring (thousands)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.text(0.03, 0.97, "one global preprocessing setting\ncannot fit sparse + dense rings",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="#555", style="italic")


def panel_c(ax):
    ax.set_title("(c) K circumferential position jumps ring-to-ring (stagger)",
                 fontsize=10, loc="left")
    for path, lab, c in [(F_REG, "regular (1-1)", "#1f77b4"),
                         (F_CPX, "complex (4-1)", "#d62728")]:
        df = pd.read_csv(path, usecols=["segment", "ring", "theta"])
        df = df[(df["segment"] == 1) & np.isfinite(df["ring"])]
        kth = df.groupby("ring")["theta"].median()
        x = np.linspace(0, 1, len(kth))
        ax.plot(x, kth.values, "-o", color=c, label=lab, ms=4)
    ax.set_xlabel("ring index (normalised)")
    ax.set_ylabel("K position \u03b8")
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.5, 0.18, "anchor must be re-found every ring;\nmoves up to 3/4 of the arc (complex)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            color="#555", style="italic")


def panel_d(ax):
    ax.set_title("(d) How that breaks the labels", fontsize=10, loc="left")
    names = [n for n, _ in RING]

    # 1) correct
    draw_strip(ax, 0.0, 1.3, RING, fontsize=7, mark_k=False)
    ax.text(0.65, -1.2, "K correct\n\u2713 aligned", ha="center", fontsize=8)

    # 2) K mis-detected -> whole ring rotated by one sector
    rot_labels = names[1:] + names[:1]  # shift labels up by one
    wrong = set(n for n, r in zip(names, rot_labels) if n != r)
    draw_strip(ax, 3.0, 1.3, list(zip(rot_labels, [h for _, h in RING])),
               fontsize=7, mark_k=False, wrong=wrong)
    ax.text(3.65, -1.2, "K mis-located\n(sparse/shifted)\n\u2192 ring rotated",
            ha="center", fontsize=8, color="#b30000")

    # 3) reversed handedness -> mirrored
    flip = list(reversed(RING))
    flip_disp = list(reversed(names))
    wrong3 = set(n for n, d in zip(names, flip_disp) if n != d)
    draw_strip(ax, 6.0, 1.3, [(n, h) for n, (_, h) in zip(flip_disp, flip)],
               fontsize=7, mark_k=False, wrong=wrong3)
    ax.text(6.65, -1.2, "reversed ring\n\u2192 mirrored labels",
            ha="center", fontsize=8, color="#b30000")

    # drift annotation on far block of strip 1
    ax.annotate("far block drifts\n(fixed sector)", xy=(1.3, 14.5),
                xytext=(1.9, 13.0), fontsize=7, color="#555",
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_xlim(-0.5, 8.0)
    ax.set_ylim(-3.0, 16.5)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
